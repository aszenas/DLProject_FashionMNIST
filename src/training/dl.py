"""Reusable training, evaluation and inference helpers for PyTorch models."""

import random
from copy import deepcopy
from dataclasses import dataclass
from time import perf_counter
from typing import Any

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class TrainingConfig:
    """Configuration shared by one training experiment."""

    epochs: int = 10
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    scheduler: str | None = None


def set_random_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch random generators."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of parameters updated by the optimizer."""
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _classification_metrics(
    labels: list[int], predictions: list[int], total_loss: float, sample_count: int
) -> dict[str, float]:
    return {
        "loss": total_loss / sample_count,
        "accuracy": accuracy_score(labels, predictions),
        "macro_f1": f1_score(labels, predictions, average="macro", zero_division=0),
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    """Train one epoch and aggregate loss by sample count."""
    model.train()
    total_loss = 0.0
    sample_count = 0
    labels: list[int] = []
    predictions: list[int] = []

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        sample_count += batch_size
        labels.extend(targets.detach().cpu().tolist())
        predictions.extend(logits.argmax(dim=1).detach().cpu().tolist())

    return _classification_metrics(labels, predictions, total_loss, sample_count)


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate a model without tracking gradients."""
    model.eval()
    total_loss = 0.0
    sample_count = 0
    labels: list[int] = []
    predictions: list[int] = []

    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        loss = criterion(logits, targets)
        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        sample_count += batch_size
        labels.extend(targets.cpu().tolist())
        predictions.extend(logits.argmax(dim=1).cpu().tolist())

    return _classification_metrics(labels, predictions, total_loss, sample_count)


def fit_model(
    model: nn.Module,
    train_loader: DataLoader,
    train_eval_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    config: TrainingConfig,
    device: torch.device,
) -> tuple[list[dict[str, Any]], int, dict[str, torch.Tensor], float]:
    """Train, select and restore the best checkpoint by validation metrics."""
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = None

    if config.scheduler == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config.epochs,
        )
    elif config.scheduler is not None:
        raise ValueError(f"Unsupported scheduler: {config.scheduler}")
    history: list[dict[str, Any]] = []
    best_key = (-float("inf"), -float("inf"), float("inf"))
    best_epoch = 0
    best_state = deepcopy(model.state_dict())
    training_start = perf_counter()

    for epoch in range(1, config.epochs + 1):
        epoch_start = perf_counter()
        train_one_epoch(model, train_loader, criterion, optimizer, device)
        train_metrics = evaluate_model(model, train_eval_loader, criterion, device)
        val_metrics = evaluate_model(model, val_loader, criterion, device)
        _synchronize(device)
        epoch_duration = perf_counter() - epoch_start

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_macro_f1": train_metrics["macro_f1"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_macro_f1": val_metrics["macro_f1"],
            "epoch_duration_s": epoch_duration,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(
            f"Epoch {epoch:02d}/{config.epochs}: "
            f"train_loss={row['train_loss']:.4f}, "
            f"train_acc={row['train_accuracy']:.4f}, "
            f"val_loss={row['val_loss']:.4f}, "
            f"val_acc={row['val_accuracy']:.4f}, "
            f"val_macro_f1={row['val_macro_f1']:.4f}, "
            f"duration={epoch_duration:.2f}s"
        )

        selection_key = (
            row["val_accuracy"],
            row["val_macro_f1"],
            -epoch,
        )
        if selection_key > best_key:
            best_key = selection_key
            best_epoch = epoch
            best_state = deepcopy(model.state_dict())

        if scheduler is not None:
            scheduler.step()

    _synchronize(device)
    total_training_time = perf_counter() - training_start
    model.load_state_dict(best_state)
    return history, best_epoch, best_state, total_training_time


@torch.no_grad()
def predict_model(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    """Return predictions from a model in loader order."""
    model.eval()
    predictions: list[int] = []
    for inputs, _ in loader:
        predictions.extend(model(inputs.to(device)).argmax(dim=1).cpu().tolist())
    return np.asarray(predictions, dtype=np.int64)


def measure_inference_time(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    repeats: int = 5,
) -> tuple[float, float]:
    """Return median DataLoader-inclusive and forward-only durations in seconds."""
    model.eval()
    device_batches = [inputs.to(device) for inputs, _ in loader]
    with torch.no_grad():
        for inputs in device_batches:
            model(inputs)
    _synchronize(device)

    total_times: list[float] = []
    forward_times: list[float] = []
    for _ in range(repeats):
        total_start = perf_counter()
        for inputs, _ in loader:
            model(inputs.to(device))
        _synchronize(device)
        total_times.append(perf_counter() - total_start)

        forward_start = perf_counter()
        for inputs in device_batches:
            model(inputs)
        _synchronize(device)
        forward_times.append(perf_counter() - forward_start)

    return float(np.median(total_times)), float(np.median(forward_times))
