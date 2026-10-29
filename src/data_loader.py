"""Offline loading and filtering for the scikit-learn digits dataset."""

from dataclasses import dataclass

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


@dataclass(frozen=True)
class DigitDataset:
    images: np.ndarray
    labels: np.ndarray
    target_names: tuple[int, int]


def load_binary_digits(
    class_a: int = 3,
    class_b: int = 7,
    test_size: float = 0.25,
    random_seed: int = 42,
    sample_limit: int | None = None,
) -> DigitDataset:
    if class_a == class_b or not 0 <= class_a <= 9 or not 0 <= class_b <= 9:
        raise ValueError("class_a and class_b must be different digits from 0 to 9")
    if not 0.1 <= test_size < 0.5:
        raise ValueError("test_size must be between 0.1 and 0.49")
    if sample_limit is not None and sample_limit < 4:
        raise ValueError("sample_limit must be at least 4")

    data = load_digits()
    mask = np.isin(data.target, [class_a, class_b])
    images = data.images[mask].astype(np.float64) / 16.0
    labels = data.target[mask].astype(int)
    if sample_limit is not None and sample_limit < len(labels):
        _, images, _, labels = train_test_split(
            images,
            labels,
            test_size=sample_limit,
            random_state=random_seed,
            stratify=labels,
        )
    train_images, test_images, train_labels, test_labels = train_test_split(
        images,
        labels,
        test_size=test_size,
        random_state=random_seed,
        stratify=labels,
    )
    return DigitDataset(
        images=np.concatenate([train_images, test_images]),
        labels=np.concatenate([train_labels, test_labels]),
        target_names=(class_a, class_b),
    )


def split_dataset(
    dataset: DigitDataset, test_size: float = 0.25, random_seed: int = 42
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(dataset.labels))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_seed,
        stratify=dataset.labels,
    )
    return (
        dataset.images[train_idx],
        dataset.images[test_idx],
        dataset.labels[train_idx],
        dataset.labels[test_idx],
    )
