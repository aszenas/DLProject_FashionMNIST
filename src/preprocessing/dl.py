"""PyTorch loaders for the shared Fashion-MNIST split."""

from collections.abc import Iterable
from operator import index
from typing import Literal

import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

DEFAULT_BATCH_SIZE = 64
DEFAULT_SEED = 42
Split = Literal["train", "val", "test"]


class _ImageSubset(Dataset):
    """Read raw samples by their original index and apply a local transform."""

    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, position):
        image, label = self.dataset[self.indices[position]]
        return self.transform(image), int(label)


def _checked_indices(indices: Iterable[int], dataset_size: int, name: str) -> list[int]:
    result = [index(value) for value in indices]
    if not result:
        raise ValueError(f"{name} must contain at least one index.")
    if len(set(result)) != len(result):
        raise ValueError(f"{name} contains duplicate indices.")
    if any(value < 0 or value >= dataset_size for value in result):
        raise ValueError(f"{name} contains an index outside the original training dataset.")
    return result


def _training_normalization(dataset, indices, to_tensor):
    total = 0.0
    squared_total = 0.0
    pixel_count = 0
    for idx in indices:
        image, _ = dataset[idx]
        pixels = to_tensor(image).to(torch.float64)
        total += pixels.sum().item()
        squared_total += pixels.square().sum().item()
        pixel_count += pixels.numel()

    mean = total / pixel_count
    variance = max(0.0, squared_total / pixel_count - mean**2)
    std = variance**0.5
    if std == 0:
        raise ValueError("The training pixels have zero standard deviation.")
    return transforms.Normalize((mean,), (std,))


def build_dataloaders(
    train_dataset,
    test_dataset,
    train_indices: Iterable[int],
    val_indices: Iterable[int],
    *,
    batch_size: int = DEFAULT_BATCH_SIZE,
    seed: int = DEFAULT_SEED,
    standardize: bool = False,
    train_augmentation=None,
) -> dict[Split, DataLoader]:
    """Build loaders from raw datasets and the shared original training indices.

    Datasets must yield untransformed grayscale PIL images and class labels.
    Optional augmentation accepts and returns a PIL image, before ToTensor.
    """
    train_indices = _checked_indices(train_indices, len(train_dataset), "train_indices")
    val_indices = _checked_indices(val_indices, len(train_dataset), "val_indices")
    if set(train_indices).intersection(val_indices):
        raise ValueError("Training and validation indices must not overlap.")

    to_tensor = transforms.ToTensor()
    common_steps = [to_tensor]
    if standardize:
        common_steps.append(_training_normalization(train_dataset, train_indices, to_tensor))

    eval_transform = transforms.Compose(common_steps)
    train_steps = list(common_steps)
    if train_augmentation is not None:
        train_steps.insert(0, train_augmentation)

    train = _ImageSubset(train_dataset, train_indices, transforms.Compose(train_steps))
    val = _ImageSubset(train_dataset, val_indices, eval_transform)
    test = _ImageSubset(test_dataset, range(len(test_dataset)), eval_transform)
    generator = torch.Generator().manual_seed(seed)

    return {
        "train": DataLoader(train, batch_size=batch_size, shuffle=True, generator=generator),
        "val": DataLoader(val, batch_size=batch_size, shuffle=False),
        "test": DataLoader(test, batch_size=batch_size, shuffle=False),
    }
