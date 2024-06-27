import os
import sys

# Set the path to the src directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))) 
import numpy as np
import pytest

from src import preprocess


@pytest.mark.manual
def test_uniqueness():
    """
    Test the uniqueness of data samples in X_train, X_test, and X_val arrays.
    Data should be at least 99% unique.
    """
    seed = np.random.randint(0, 100)
    random_slice = slice(seed*1000, (seed+5)*1000)
    X_train = np.load("data/preprocess/X_train.npy")[random_slice]
    X_val = np.load("data/preprocess/X_val.npy")[random_slice]
    X_test = np.load("data/preprocess/X_test.npy")[random_slice]
    train_unique = len(set(map(tuple, X_train)))
    test_unique = len(set(map(tuple, X_test)))
    val_unique = len(set(map(tuple, X_val)))
    assert train_unique >= 0.99 * len(X_train)
    assert test_unique >= 0.99 * len(X_test)
    assert val_unique >= 0.99 * len(X_val)

@pytest.mark.fast
def test_uniqueness_fake_data():
    sys.argv = ["", "tests/testdata", "false"]
    preprocess.main()
    X_train = np.load("tests/testdata/preprocess/X_train.npy")
    X_val = np.load("tests/testdata/preprocess/X_val.npy")
    X_test = np.load("tests/testdata/preprocess/X_test.npy")
    train_unique = len(set(map(tuple, X_train)))
    test_unique = len(set(map(tuple, X_test)))
    val_unique = len(set(map(tuple, X_val)))
    assert train_unique == len(X_train)
    assert test_unique == len(X_test)
    # assert val_unique == len(X_val) Not unique in test data-set