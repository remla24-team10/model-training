import sys
import os

# Set the path to the src directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from src import model_definition
from src import preprocess
from src import predict
from src import train
import numpy as np
import pytest
import yaml
import json

import keras

@pytest.mark.fast
def test_integration_small_sample():
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
    # Evaluate the model 
    prediction = predict.predict_classes(trained_model, X_test)
    evaluation_results = predict.evaluate_results(y_test, prediction)
    # Plot confusion matrix
    predict.plot_confusion_matrix(evaluation_results['confusion_matrix'])

@pytest.mark.dev
def test_integration():
    sys.argv = ["", "data"]
    preprocess.main()
    assert os.path.exists("data/preprocess/X_train.npy") and os.path.exists("data/preprocess/y_train.npy") 
    assert os.path.exists("data/preprocess/X_val.npy") and os.path.exists("data/preprocess/y_val.npy")
    assert os.path.exists("data/preprocess/X_test.npy") and os.path.exists("data/preprocess/y_test.npy")

    X_train, y_train = np.load("data/preprocess/X_train.npy"), np.load("data/preprocess/y_train.npy")
    X_test, y_test = np.load("data/preprocess/X_test.npy"), np.load("data/preprocess/y_test.npy")
    X_val, y_val = np.load("data/preprocess/X_val.npy"), np.load("data/preprocess/y_val.npy")
    char_index = json.load(open("data/preprocess/char_index.json", "r"))

    file_path = os.path.join(os.path.dirname(__file__), "test_params.yaml")
    with open(file_path, "r") as file:
        params = yaml.safe_load(file)
    model = model_definition.build_model(char_index, params['categories'])
    
    trained_model = train.train(model, X_train, y_train, X_val, y_val, params)
    prediction = predict.predict_classes(trained_model, X_test)
    evaluation_results = predict.evaluate_results(y_test, prediction)
    predict.plot_confusion_matrix(evaluation_results['confusion_matrix'])
    assert evaluation_results['accuracy'] > 0.9
