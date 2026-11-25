import random

import matplotlib.pyplot as plt

from src.preprocessing.dataset import load_dataset


def preview_dataset(num_samples=10, seed=42):
    train_dataset, test_dataset = load_dataset()

    # Basic dataset information
    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Image shape: {train_dataset.data[0].shape}")
    print(f"Data type: {train_dataset.data.dtype}")
    print(f"Number of classes: {len(train_dataset.classes)}")
    print(f"Classes: {train_dataset.classes}")

    # Select random training samples
    rng = random.Random(seed)
    indices = rng.sample(range(len(train_dataset)), num_samples)

    # Create visualization
    fig, axes = plt.subplots(2, 5, figsize=(12, 6))

    for ax, idx in zip(axes.flatten(), indices):
        image, label = train_dataset[idx]
        class_name = train_dataset.classes[label]

        ax.imshow(image, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{class_name}\nLabel: {label}")
        ax.axis("off")

        print(f"Index: {idx} | Label: {label} | Class: {class_name}")

    fig.suptitle("Fashion-MNIST - Random Training Samples")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    preview_dataset()