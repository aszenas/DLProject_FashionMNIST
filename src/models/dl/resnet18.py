"""Adapted ResNet-18 classifier for Fashion-MNIST."""

from torch import nn
from torchvision.models import resnet18


def build_resnet18(
    dropout: float = 0.0,
    num_classes: int = 10,
) -> nn.Module:
    """Build a ResNet-18 for 28x28 grayscale images."""
    model = resnet18(weights=None)

    # Adapt the input convolution to grayscale images.
    model.conv1 = nn.Conv2d(
        in_channels=1,
        out_channels=64,
        kernel_size=3,
        stride=1,
        padding=1,
        bias=False,
    )

    nn.init.kaiming_normal_(
        model.conv1.weight,
        mode="fan_out",
        nonlinearity="relu",
    )

    # Preserve spatial resolution in the initial layers.
    model.maxpool = nn.Identity()

    # Adapt the classification head.
    in_features = model.fc.in_features

    if dropout > 0:
        model.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )
    else:
        model.fc = nn.Linear(
            in_features,
            num_classes,
        )

    return model
