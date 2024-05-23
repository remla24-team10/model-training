import sys
import os
# Set the path to the src directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from src import model_definition
from src import preprocess
import numpy as np
import json
import pytest
import keras

@pytest.mark.fast
def test_fake_data():
    # Load in the data
    sys.argv = ["", "tests/testdata"]
    preprocess.main()
    X_train, y_train = np.load("tests/testdata/preprocess/X_train.npy"), np.load("tests/testdata/preprocess/y_train.npy")
    X_test, y_test = np.load("tests/testdata/preprocess/X_test.npy"), np.load("tests/testdata/preprocess/y_test.npy")
    X_val, y_val = np.load("tests/testdata/preprocess/X_val.npy"), np.load("tests/testdata/preprocess/y_val.npy")
    # Char index
    char_index = json.load(open("tests/testdata/preprocess/char_index.json", "r"))    
    # Define the model 
    model = model_definition.build_model(char_index, ['label1', 'label2'])