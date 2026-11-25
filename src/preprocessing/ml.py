"""Scikit-learn preprocessing for the shared Fashion-MNIST split."""

from collections.abc import Iterable
from operator import index

import numpy as np
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

DEFAULT_SEED = 42
IMAGE_SHAPE = (28, 28)
NUM_FEATURES = 784


def _checked_indices(indices: Iterable[int], dataset_size: int, name: str) -> list[int]:
    """Validate indices referring to the original training dataset."""

    result = [index(value) for value in indices]

    if not result:
        raise ValueError(f"{name} must contain at least one index.")

    if len(set(result)) != len(result):
        raise ValueError(f"{name} contains duplicate indices.")

    if any(value < 0 or value >= dataset_size for value in result):
        raise ValueError(f"{name} contains an index outside the original training dataset.")

    return result


def _extract_samples(dataset, indices):
    """Extract raw grayscale images and their corresponding labels."""

    images = np.empty((len(indices), *IMAGE_SHAPE), dtype=np.uint8)
    labels = np.empty(len(indices), dtype=np.int64)

    for position, idx in enumerate(indices):
        image, label = dataset[idx]
        pixels = np.asarray(image)

        if pixels.shape != IMAGE_SHAPE:
            raise ValueError(f"Expected image shape {IMAGE_SHAPE}, got {pixels.shape}.")

        if pixels.dtype != np.uint8:
            raise ValueError("Expected raw uint8 images.")

        images[position] = pixels
        labels[position] = int(label)

    return images, labels


def _scale_pixels(images):
    """Convert raw pixel values to float32 in the [0, 1] range."""

    return images.astype(np.float32) / np.float32(255.0)


def _flatten_images(images):
    """Convert 28x28 images to 784-dimensional feature vectors."""

    return images.reshape(-1, NUM_FEATURES)


def _extract_hog_features(images):
    """Extract Histogram of Oriented Gradients features."""

    features = [
        hog(
            image,
            orientations=9,
            pixels_per_cell=(4, 4),
            cells_per_block=(2, 2),
            block_norm="L2-Hys",
            feature_vector=True,
            channel_axis=None,
        )
        for image in images
    ]

    return np.asarray(features, dtype=np.float32)


def build_preprocessor(
    *,
    standardize: bool = False,
    pca_components: int | None = None,
    use_hog: bool = False,
    seed: int = DEFAULT_SEED,
) -> Pipeline:
    """Create a configurable ML preprocessing pipeline."""

    steps = [
        (
            "pixel_scaling",
            FunctionTransformer(_scale_pixels, validate=False),
        )
    ]

    if use_hog:
        steps.append(
            (
                "hog",
                FunctionTransformer(_extract_hog_features, validate=False),
            )
        )
    else:
        steps.append(
            (
                "flatten",
                FunctionTransformer(_flatten_images, validate=False),
            )
        )

    if standardize:
        steps.append(("standardize", StandardScaler()))

    if pca_components is not None:
        steps.append(
            (
                "pca",
                PCA(n_components=pca_components, random_state=seed),
            )
        )

    return Pipeline(steps)


def build_ml_datasets(
    train_dataset,
    test_dataset,
    train_indices: Iterable[int],
    val_indices: Iterable[int],
    *,
    standardize: bool = False,
    pca_components: int | None = None,
    use_hog: bool = False,
    seed: int = DEFAULT_SEED,
):
    """Build ML datasets using the shared original training indices.

    Datasets must yield untransformed grayscale PIL images
    and integer class labels.

    The preprocessor is fitted exclusively on training samples.
    Validation and test samples are only transformed.
    """

    train_indices = _checked_indices(train_indices, len(train_dataset), "train_indices")
    val_indices = _checked_indices(val_indices, len(train_dataset), "val_indices")

    if set(train_indices).intersection(val_indices):
        raise ValueError("Training and validation indices must not overlap.")

    X_train, y_train = _extract_samples(train_dataset, train_indices)
    X_val, y_val = _extract_samples(train_dataset, val_indices)
    X_test, y_test = _extract_samples(test_dataset, range(len(test_dataset)))

    preprocessor = build_preprocessor(
        standardize=standardize,
        pca_components=pca_components,
        use_hog=use_hog,
        seed=seed,
    )

    X_train = preprocessor.fit_transform(X_train)
    X_val = preprocessor.transform(X_val)
    X_test = preprocessor.transform(X_test)

    datasets = {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }

    return datasets, preprocessor
