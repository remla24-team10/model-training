import sys
import os

from matplotlib import pyplot as plt

# Set the path to the src directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from src import model_definition
from src import preprocess
from src import train
import numpy as np
import pytest
import yaml
import json
import keras

@pytest.mark.fast
def test_trained_model():
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
    # Read in paramaters
    file_path = os.path.join(os.path.dirname(__file__), "test_params.yaml")
    with open(file_path, "r") as file:
        params = yaml.safe_load(file)
    # Train the model
    trained_model = train.train(model, X_train, y_train, X_val, y_val, params)
    print(trained_model.__class__) # keras.src.models.sequential.Sequential
    # assert trained_model is keras.models.Model
    assert isinstance(trained_model, keras.models.Model)
