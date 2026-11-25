import unittest

import torch
from PIL import Image

from src.preprocessing.dl import build_dataloaders


def _samples(*values):
    return [(Image.new("L", (28, 28), value), label) for label, value in enumerate(values)]


class DLPreprocessingTests(unittest.TestCase):
    def test_baseline_keeps_split_membership_and_repeats_shuffle_with_same_seed(self):
        train = _samples(0, 64, 128, 255, 32)
        test = _samples(255, 0)
        args = (train, test, [4, 0, 2], [3, 1])
        first = build_dataloaders(*args, batch_size=2, seed=7)
        second = build_dataloaders(*args, batch_size=2, seed=7)
        first_labels = torch.cat([labels for _, labels in first["train"]])
        second_labels = torch.cat([labels for _, labels in second["train"]])
        self.assertTrue(torch.equal(first_labels, second_labels))
        self.assertEqual(sorted(first_labels.tolist()), [0, 2, 4])

        images, labels = next(iter(first["val"]))
        self.assertEqual(images.shape, (2, 1, 28, 28))
        self.assertEqual(images.dtype, torch.float32)
        self.assertEqual(labels.dtype, torch.long)
        self.assertEqual(labels.tolist(), [3, 1])
        torch.testing.assert_close(images[:, 0, 0, 0], torch.tensor([1.0, 64 / 255]))

        test_images, test_labels = next(iter(first["test"]))
        self.assertEqual(test_labels.tolist(), [0, 1])
        torch.testing.assert_close(test_images[:, 0, 0, 0], torch.tensor([1.0, 0.0]))

    def test_normalization_uses_raw_training_pixels_and_augmentation_only_changes_train(self):
        train = _samples(0, 128, 255)
        test = _samples(255)
        loaders = build_dataloaders(
            train,
            test,
            [0, 2],
            [1],
            batch_size=2,
            standardize=True,
            train_augmentation=lambda image: Image.new("L", image.size, 255),
        )
        train_images, _ = next(iter(loaders["train"]))
        val_images, _ = next(iter(loaders["val"]))
        test_images, _ = next(iter(loaders["test"]))
        torch.testing.assert_close(train_images, torch.ones_like(train_images))
        torch.testing.assert_close(val_images, torch.full_like(val_images, 1 / 255))
        torch.testing.assert_close(test_images, torch.ones_like(test_images))

    def test_rejects_overlap_duplicate_and_out_of_range_indices(self):
        train = _samples(0, 128, 255)
        test = _samples(255)
        cases = (
            ([0, 1], [1], "overlap"),
            ([0, 0], [1], "duplicate"),
            ([0], [3], "outside"),
        )
        for train_indices, val_indices, message in cases:
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                build_dataloaders(train, test, train_indices, val_indices)


if __name__ == "__main__":
    unittest.main()
