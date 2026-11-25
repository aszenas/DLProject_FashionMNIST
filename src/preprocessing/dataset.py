from pathlib import Path

from torchvision.datasets import FashionMNIST

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "raw"


def load_dataset():
    train_dataset = FashionMNIST(
        root=DATA_DIR,
        train=True,
        download=True,
    )

    test_dataset = FashionMNIST(
        root=DATA_DIR,
        train=False,
        download=True,
    )

    return train_dataset, test_dataset


if __name__ == "__main__":
    train_dataset, test_dataset = load_dataset()

    print(f"Training samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")

    print(f"Training shape: {train_dataset.data.shape}")
    print(f"Test shape: {test_dataset.data.shape}")
