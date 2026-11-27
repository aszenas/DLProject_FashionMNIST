import unittest

import torch
from torch import nn

from src.models.dl.simple_cnn import SimpleCNN, build_simple_cnn
from src.training.dl import count_trainable_parameters


class SimpleCNNTests(unittest.TestCase):
    def test_baseline_architecture_and_parameter_count(self):
        model = build_simple_cnn()

        self.assertIsInstance(model, SimpleCNN)
        self.assertEqual(count_trainable_parameters(model), 421_642)
        self.assertEqual(model.conv1.out_channels, 32)
        self.assertEqual(model.conv2.out_channels, 64)
        self.assertEqual(model.dropout.p, 0.0)

    def test_forward_returns_raw_logits_with_expected_shape(self):
        model = build_simple_cnn()
        outputs = model(torch.randn(4, 1, 28, 28))

        self.assertEqual(outputs.shape, (4, 10))
        self.assertFalse(any(isinstance(module, nn.Softmax) for module in model.modules()))

    def test_dropout_is_configurable(self):
        self.assertEqual(build_simple_cnn(dropout=0.3).dropout.p, 0.3)


if __name__ == "__main__":
    unittest.main()
