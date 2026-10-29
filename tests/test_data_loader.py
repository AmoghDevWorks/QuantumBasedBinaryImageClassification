import numpy as np
import pytest

from src.data_loader import load_binary_digits


def test_binary_dataset_is_filtered():
    dataset = load_binary_digits(3, 7, sample_limit=80)
    assert set(np.unique(dataset.labels)) == {3, 7}
    assert dataset.images.shape[1:] == (8, 8)


def test_invalid_classes_are_rejected():
    with pytest.raises(ValueError):
        load_binary_digits(3, 3)
