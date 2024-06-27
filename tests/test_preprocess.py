import os
import sys

# Set the path to the src directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))) 
import json

import numpy as np
import pytest

from src import preprocess


@pytest.mark.fast
def test_fake_data():
    sys.argv = ["", "tests/testdata", "false"]
    preprocess.main()
    assert os.path.exists("tests/testdata/preprocess/X_train.npy") and os.path.exists("tests/testdata/preprocess/y_train.npy")
    assert os.path.exists("tests/testdata/preprocess/X_val.npy") and os.path.exists("tests/testdata/preprocess/y_val.npy")
    assert os.path.exists("tests/testdata/preprocess/X_test.npy") and os.path.exists("tests/testdata/preprocess/y_test.npy")

@pytest.mark.fast
def test_invalid_path():
    sys.argv = ["", "tests/testdata_invalid", "false"]
    with pytest.raises(FileNotFoundError):
        preprocess.main()
        
@pytest.mark.fast
def test_data_shape():
    X_train, y_train = np.load("tests/testdata/preprocess/X_train.npy"), np.load("tests/testdata/preprocess/y_train.npy")
    X_test, y_test = np.load("tests/testdata/preprocess/X_test.npy"), np.load("tests/testdata/preprocess/y_test.npy")
    X_val, y_val = np.load("tests/testdata/preprocess/X_val.npy"), np.load("tests/testdata/preprocess/y_val.npy")
    assert X_train.shape == (100,200) and y_train.shape == (100,)
    assert X_test.shape == (100,200) and y_test.shape == (100,)
    assert X_val.shape == (100,200) and y_val.shape == (100,)
    
@pytest.mark.fast
def test_char_index():
    with open("tests/testdata/preprocess/char_index.json", "r") as file:
        char_index = json.load(file)
    assert len(char_index) == 53
    
@pytest.mark.manual
def test_real_data():
    sys.argv = ["", "data", "false"]
    preprocess.main()
    assert os.path.exists("data/preprocess/X_train.npy") and os.path.exists("data/preprocess/y_train.npy") 
    assert os.path.exists("data/preprocess/X_val.npy") and os.path.exists("data/preprocess/y_val.npy")
    assert os.path.exists("data/preprocess/X_test.npy") and os.path.exists("data/preprocess/y_test.npy")
