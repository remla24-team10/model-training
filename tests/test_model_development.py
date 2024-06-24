import os
import random
import sys

import numpy as np
import pytest
import tensorflow as tf
import yaml

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..' '/src')))

from lib_ml_remla import split_data

from src.model_definition import build_model
from src.predict import evaluate_results, predict_classes
from src.train import train
from src.utility_functions import load_data_from_text, load_json


@pytest.mark.manual
def test_slices():

    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    
    model = tf.keras.models.load_model(os.path.join(path, "models", "trained_model.keras"))

    X_test = np.load(os.path.join(path, "data", "preprocess", "X_test.npy"))
    y_test = np.load(os.path.join(path, "data", "preprocess", "y_test.npy"))


    test = load_data_from_text(os.path.join(path, "data", "raw", "test.txt"))
    test = [line.strip() for line in test] 
    #Fix parameter order in lib-ml? (test input is in pos2 while test output is in pos5-6)
    _, _, _, _, X_test_split, _ = split_data([], test, [])
    
    #Slice test into urls starting with 'http' and 'https' based on the first 4-5 characters per entry
    http_indices = [i for i, url in enumerate(X_test_split) if url[:4] == 'http']
    https_indices = [i for i, url in enumerate(X_test_split) if url[:5] == 'https']
    
    http_X_test = X_test[http_indices]
    https_X_test = X_test[https_indices]
    
    http_y_test = y_test[http_indices]
    https_y_test = y_test[https_indices]
    
    http_predictions = predict_classes(model, http_X_test)
    https_predictions = predict_classes(model, https_X_test)
    
    http_results = evaluate_results(http_y_test, http_predictions)
    https_results = evaluate_results(https_y_test, https_predictions)
    
    assert (http_results['accuracy'] > 0.9)
    assert (https_results['accuracy'] > 0.9)
    assert ((https_results['accuracy'] - http_results['accuracy']) < 0.05)
        
@pytest.mark.training
def test_non_determinism():
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    with open(os.path.join(path, "src", "params.yaml"), 'r') as file:
        params = yaml.safe_load(file)
    
    X_train = np.load(os.path.join(path, "data", "preprocess", "X_train.npy"))
    y_train = np.load(os.path.join(path, "data", "preprocess", "y_train.npy"))
    X_val = np.load(os.path.join(path, "data", "preprocess", "X_val.npy"))
    y_val = np.load(os.path.join(path, "data", "preprocess", "y_val.npy"))
    X_test = np.load(os.path.join(path, "data", "preprocess", "X_test.npy"))
    y_test = np.load(os.path.join(path, "data", "preprocess", "y_test.npy"))
    char_index = load_json(os.path.join(path, "data", "preprocess", "char_index.json"))
    
    #First model (pipeline output)
    trained_model1 = tf.keras.models.load_model(os.path.join(path, "models", "trained_model.keras"))
    predictions_model1 = predict_classes(trained_model1, X_test)
    results_model1 = evaluate_results(y_test, predictions_model1)
    
    
    #Set seeds
    random.seed(694201337)
    np.random.seed(694201337)
    tf.random.set_seed(694201337)
    
    #Second model (trained with fixed seed)
    model2 = build_model(char_index, params['categories'])
    trained_model2 = train(model2, X_train, y_train, X_val, y_val, params)
    predictions_model2 = predict_classes(trained_model2, X_test)
    results_model2 = evaluate_results(y_test, predictions_model2)
    
    assert (results_model1['accuracy'] - results_model2['accuracy']) < 0.05