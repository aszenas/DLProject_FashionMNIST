"""Generate and load the shared Fashion-MNIST train/validation indices."""

from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPLIT_PATH = PROJECT_ROOT / "data" / "splits.npz"
SOURCE_TRAIN_SIZE = 60_000
VALIDATION_SIZE = 10_000
SPLIT_SEED = 42


def _validate_split(train_indices: np.ndarray, val_indices: np.ndarray) -> None:
    for name, indices, expected_size in (
        ("train_indices", train_indices, SOURCE_TRAIN_SIZE - VALIDATION_SIZE),
        ("val_indices", val_indices, VALIDATION_SIZE),
    ):
        if indices.ndim != 1 or not np.issubdtype(indices.dtype, np.integer):
            raise ValueError(f"{name} must be a one-dimensional integer array.")
        if len(indices) != expected_size:
            raise ValueError(f"{name} must contain {expected_size} indices.")

    # Exact coverage also rules out overlaps, duplicates and out-of-range indices.
    combined = np.concatenate((train_indices, val_indices))
    if not np.array_equal(np.sort(combined), np.arange(SOURCE_TRAIN_SIZE)):
        raise ValueError("Each original training index must occur exactly once.")


def load_split(path: str | Path = SPLIT_PATH) -> tuple[np.ndarray, np.ndarray]:
    """Return validated train and validation indices in their saved order."""
    with np.load(path, allow_pickle=False) as split:
        train_indices = split["train_indices"]
        val_indices = split["val_indices"]
    _validate_split(train_indices, val_indices)
    return train_indices, val_indices


def generate_split(labels, path: str | Path = SPLIT_PATH) -> Path:
    """Stratify the original training indices and save them for both pipelines."""
    labels = np.asarray(labels)
    if labels.shape != (SOURCE_TRAIN_SIZE,):
        raise ValueError(f"Expected {SOURCE_TRAIN_SIZE} original training labels.")

    train_indices, val_indices = train_test_split(
        np.arange(SOURCE_TRAIN_SIZE, dtype=np.int64),
        test_size=VALIDATION_SIZE,
        random_state=SPLIT_SEED,
        stratify=labels,
    )
    _validate_split(train_indices, val_indices)

    output_path = Path(path)
    if output_path.suffix != ".npz":
        raise ValueError("The split file must have an .npz extension.")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, train_indices=train_indices, val_indices=val_indices)
    return output_path


def main() -> None:
    from src.preprocessing.dataset import load_dataset

    train_dataset, _ = load_dataset()
    path = generate_split(train_dataset.targets)
    train_indices, val_indices = load_split(path)
    print(
        f"Saved {len(train_indices)} training and {len(val_indices)} validation indices to {path}"
    )


if __name__ == "__main__":
    main()
