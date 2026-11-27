"""Simple convolutional neural network for Fashion-MNIST."""

import torch
from torch import nn


class SimpleCNN(nn.Module):
    """Two-block convolutional classifier for 28x28 grayscale images."""

    def __init__(self, dropout: float = 0.0, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return raw class logits without applying softmax."""
        outputs = self.pool(self.relu(self.conv1(inputs)))
        outputs = self.pool(self.relu(self.conv2(outputs)))
        outputs = self.flatten(outputs)
        outputs = self.relu(self.fc1(outputs))
        outputs = self.dropout(outputs)
        return self.fc2(outputs)


def build_simple_cnn(dropout: float = 0.0) -> SimpleCNN:
    """Build a Simple CNN with the baseline architecture."""
    return SimpleCNN(dropout=dropout)
