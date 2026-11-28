"""Unit tests for the adapted ResNet-18 model."""

import unittest

import torch
from torch import nn

from src.models.dl.resnet18 import build_resnet18


class TestResNet18(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = build_resnet18()
        self.model.eval()

    def test_input_convolution(self):
        conv = self.model.conv1

        self.assertEqual(conv.in_channels, 1)
        self.assertEqual(conv.out_channels, 64)
        self.assertEqual(conv.kernel_size, (3, 3))
        self.assertEqual(conv.stride, (1, 1))
        self.assertEqual(conv.padding, (1, 1))
        self.assertIsNone(conv.bias)

    def test_initial_pooling_removed(self):
        self.assertIsInstance(self.model.maxpool, nn.Identity)

    def test_residual_block_configuration(self):
        layers = [
            self.model.layer1,
            self.model.layer2,
            self.model.layer3,
            self.model.layer4,
        ]

        self.assertEqual([len(layer) for layer in layers], [2, 2, 2, 2])

    def test_classification_head(self):
        self.assertIsInstance(self.model.fc, nn.Linear)
        self.assertEqual(self.model.fc.in_features, 512)
        self.assertEqual(self.model.fc.out_features, 10)

    def test_parameter_count(self):
        count = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        self.assertEqual(count, 11_172_810)

    def test_forward_pass(self):
        inputs = torch.rand(2, 1, 28, 28)

        with torch.no_grad():
            outputs = self.model(inputs)

        self.assertEqual(outputs.shape, (2, 10))
        self.assertTrue(torch.isfinite(outputs).all().item())

    def test_dropout_configuration(self):
        model = build_resnet18(dropout=0.3)

        self.assertIsInstance(model.fc, nn.Sequential)
        self.assertIsInstance(model.fc[0], nn.Dropout)
        self.assertEqual(model.fc[0].p, 0.3)
        self.assertIsInstance(model.fc[1], nn.Linear)

    def test_dropout_parameter_count(self):
        model = build_resnet18(dropout=0.3)

        count = sum(p.numel() for p in model.parameters() if p.requires_grad)

        self.assertEqual(count, 11_172_810)


if __name__ == "__main__":
    unittest.main()
