import numpy as np

from src.data_loader import load_binary_digits
from src.preprocessing import prepare_data


def test_preprocessing_is_reproducible_and_scaled():
    dataset = load_binary_digits(3, 7, sample_limit=100, random_seed=11)
    first = prepare_data(dataset, 4, random_seed=11)
    second = prepare_data(dataset, 4, random_seed=11)
    np.testing.assert_allclose(first.x_train, second.x_train)
    assert np.max(np.abs(first.x_test)) <= np.pi + 1e-8


def test_feature_count_validation():
    dataset = load_binary_digits(3, 7, sample_limit=50)
    try:
        prepare_data(dataset, 3)
    except ValueError:
        pass
    else:
        raise AssertionError("invalid feature count should fail")
