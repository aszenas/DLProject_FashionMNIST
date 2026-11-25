"""Preview the Fashion-MNIST ML preprocessing output."""

import matplotlib.pyplot as plt
import numpy as np

from src.preprocessing.dataset import load_dataset
from src.preprocessing.ml import build_ml_datasets
from src.preprocessing.split import load_split


def preview_ml_preprocessing():
    train_dataset, test_dataset = load_dataset()

    train_indices, val_indices = load_split()

    datasets, preprocessor = build_ml_datasets(
        train_dataset,
        test_dataset,
        train_indices,
        val_indices,
        standardize=False,
        pca_components=None,
        use_hog=False,
        seed=42,
    )

    print("\n=== ML PREPROCESSING PIPELINE ===")
    print(preprocessor)

    print("\n=== DATASET INFORMATION ===")

    for split, (X, y) in datasets.items():
        print(f"\n{split.upper()}")
        print(f"Feature matrix shape: {X.shape}")
        print(f"Label vector shape: {y.shape}")
        print(f"Feature data type: {X.dtype}")
        print(f"Label data type: {y.dtype}")
        print(f"Minimum value: {X.min():.4f}")
        print(f"Maximum value: {X.max():.4f}")

    X_train, y_train = datasets["train"]

    # Inspect the first training sample
    sample = X_train[0]
    label = y_train[0]
    original_index = int(train_indices[0])

    print("\n=== SAMPLE AFTER PREPROCESSING ===")
    print(f"Original image shape: {train_dataset.data[original_index].shape}")
    print(f"Processed image shape: {sample.shape}")
    print(f"Number of features: {sample.size}")
    print(f"Label: {label}")
    print(f"Class name: {train_dataset.classes[label]}")

    print("\nFirst 30 features:")
    print(sample[:30])

    print("\nComplete feature vector:")
    with np.printoptions(precision=4, suppress=True, threshold=np.inf):
        print(sample)

        fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    original_image, _ = train_dataset[original_index]

    axes[0].imshow(original_image, cmap="gray", vmin=0, vmax=255)
    axes[0].set_title("Original image (28x28)")
    axes[0].axis("off")

    axes[1].imshow(
        sample.reshape(28, 28),
        cmap="gray",
        vmin=0,
        vmax=1,
    )
    axes[1].set_title("Scaled image (0-1)")
    axes[1].axis("off")

    axes[2].imshow(
        sample.reshape(1, -1),
        cmap="gray",
        aspect="auto",
        vmin=0,
        vmax=1,
    )
    axes[2].set_title("Flattened vector (784)")
    axes[2].set_yticks([])
    axes[2].set_xlabel("Feature index")

    fig.suptitle(f"ML Preprocessing - {train_dataset.classes[label]}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    preview_ml_preprocessing()
