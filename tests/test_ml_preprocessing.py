import unittest

import numpy as np
from PIL import Image

from src.preprocessing.dl import build_dataloaders
from src.preprocessing.ml import build_ml_datasets


def _samples(*values):
    return [(Image.new("L", (28, 28), value), label) for label, value in enumerate(values)]


class MLPreprocessingTests(unittest.TestCase):
    def test_baseline_shape_dtype_scaling_and_labels(self):
        train = _samples(0, 64, 128, 255, 32)
        test = _samples(255, 0)

        datasets, _ = build_ml_datasets(train, test, [4, 0, 2], [3, 1])

        X_train, y_train = datasets["train"]
        X_val, y_val = datasets["val"]
        X_test, y_test = datasets["test"]

        self.assertEqual(X_train.shape, (3, 784))
        self.assertEqual(X_val.shape, (2, 784))
        self.assertEqual(X_test.shape, (2, 784))

        self.assertEqual(X_train.dtype, np.float32)
        self.assertEqual(X_val.dtype, np.float32)
        self.assertEqual(X_test.dtype, np.float32)
        self.assertEqual(y_train.dtype, np.int64)

        np.testing.assert_array_equal(y_train, [4, 0, 2])
        np.testing.assert_array_equal(y_val, [3, 1])
        np.testing.assert_array_equal(y_test, [0, 1])

        np.testing.assert_allclose(X_train[:, 0], [32 / 255, 0, 128 / 255])
        np.testing.assert_allclose(X_val[:, 0], [1.0, 64 / 255])
        np.testing.assert_allclose(X_test[:, 0], [1.0, 0.0])

        for X, _ in datasets.values():
            self.assertTrue(np.all(X >= 0))
            self.assertTrue(np.all(X <= 1))

    def test_standardization_is_fitted_only_on_training_data(self):
        train = _samples(0, 128, 255)
        test = _samples(255)

        datasets, preprocessor = build_ml_datasets(
            train,
            test,
            [0, 2],
            [1],
            standardize=True,
        )

        X_train, _ = datasets["train"]
        X_val, _ = datasets["val"]
        X_test, _ = datasets["test"]

        scaler = preprocessor.named_steps["standardize"]

        np.testing.assert_allclose(scaler.mean_, 0.5)
        np.testing.assert_allclose(scaler.scale_, 0.5)

        np.testing.assert_allclose(X_train[0], -1.0)
        np.testing.assert_allclose(X_train[1], 1.0)

        np.testing.assert_allclose(X_val, 1 / 255, atol=1e-7)
        np.testing.assert_allclose(X_test, 1.0)

    def test_pca_reduces_feature_dimensions(self):
        train = _samples(0, 64, 128, 255)
        test = _samples(32, 200)

        datasets, preprocessor = build_ml_datasets(
            train,
            test,
            [0, 1, 3],
            [2],
            pca_components=1,
        )

        self.assertEqual(datasets["train"][0].shape, (3, 1))
        self.assertEqual(datasets["val"][0].shape, (1, 1))
        self.assertEqual(datasets["test"][0].shape, (2, 1))

        pca = preprocessor.named_steps["pca"]

        self.assertEqual(pca.n_components_, 1)
        self.assertEqual(pca.n_samples_, 3)

        expected_mean = np.mean(np.array([0, 64, 255], dtype=np.float32) / 255)

        np.testing.assert_allclose(pca.mean_, expected_mean, atol=1e-7)

    def test_hog_extracts_expected_number_of_features(self):
        train = _samples(0, 128, 255)
        test = _samples(255)

        datasets, preprocessor = build_ml_datasets(
            train,
            test,
            [0, 2],
            [1],
            use_hog=True,
        )

        self.assertIn("hog", preprocessor.named_steps)
        self.assertNotIn("flatten", preprocessor.named_steps)

        self.assertEqual(datasets["train"][0].shape, (2, 1296))
        self.assertEqual(datasets["val"][0].shape, (1, 1296))
        self.assertEqual(datasets["test"][0].shape, (1, 1296))

        for X, _ in datasets.values():
            self.assertTrue(np.all(np.isfinite(X)))

    def test_rejects_overlap_duplicate_and_out_of_range_indices(self):
        train = _samples(0, 128, 255)
        test = _samples(255)

        cases = (
            ([0, 1], [1], "overlap"),
            ([0, 0], [1], "duplicate"),
            ([0], [3], "outside"),
            ([-1], [1], "outside"),
            ([], [1], "at least one index"),
        )

        for train_indices, val_indices, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    build_ml_datasets(train, test, train_indices, val_indices)

    def test_ml_and_dl_baseline_values_and_labels_match(self):
        pixels = np.arange(784, dtype=np.int32).reshape(28, 28)
        pixels = (pixels % 256).astype(np.uint8)

        train = [
            (Image.fromarray(pixels), 0),
            (Image.fromarray(np.flipud(pixels).copy()), 1),
            (Image.fromarray(np.fliplr(pixels).copy()), 2),
        ]

        test = [
            (Image.fromarray(pixels), 3),
        ]

        args = (train, test, [0, 2], [1])

        ml_datasets, _ = build_ml_datasets(*args)
        dl_loaders = build_dataloaders(*args, batch_size=2)

        for split in ("val", "test"):
            X_ml, y_ml = ml_datasets[split]
            X_dl, y_dl = next(iter(dl_loaders[split]))

            X_dl = X_dl.flatten(start_dim=1).numpy()

            np.testing.assert_allclose(X_ml, X_dl, atol=1e-7)
            np.testing.assert_array_equal(y_ml, y_dl.numpy())

        X_train, y_train = ml_datasets["train"]
        expected = dict(zip(y_train.tolist(), X_train))

        for images, labels in dl_loaders["train"]:
            flattened = images.flatten(start_dim=1).numpy()

            for image, label in zip(flattened, labels.tolist()):
                np.testing.assert_allclose(image, expected[label], atol=1e-7)


if __name__ == "__main__":
    unittest.main()
