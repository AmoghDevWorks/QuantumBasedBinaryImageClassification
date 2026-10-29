"""Leakage-safe PCA and angle scaling."""

from dataclasses import dataclass

import numpy as np
from sklearn.decomposition import PCA

from .data_loader import DigitDataset, split_dataset


@dataclass
class PreparedData:
    x_train: np.ndarray
    x_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    train_images: np.ndarray
    test_images: np.ndarray
    pca: PCA


def prepare_data(
    dataset: DigitDataset,
    n_components: int = 4,
    test_size: float = 0.25,
    random_seed: int = 42,
) -> PreparedData:
    if n_components not in (2, 4, 8):
        raise ValueError("n_components must be 2, 4, or 8")
    train_images, test_images, y_train, y_test = split_dataset(
        dataset, test_size=test_size, random_seed=random_seed
    )
    n_components = min(n_components, train_images.shape[0], train_images.shape[1])
    pca = PCA(n_components=n_components, whiten=True, random_state=random_seed)
    x_train = pca.fit_transform(train_images.reshape(len(train_images), -1))
    x_test = pca.transform(test_images.reshape(len(test_images), -1))
    scale = np.max(np.abs(x_train), axis=0)
    scale[scale == 0] = 1.0
    x_train = np.clip(x_train / scale, -1.0, 1.0) * np.pi
    x_test = np.clip(x_test / scale, -1.0, 1.0) * np.pi
    return PreparedData(x_train, x_test, y_train, y_test, train_images, test_images, pca)
