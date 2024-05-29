import pytest
import numpy as np
import os
import sys
import yaml
import psutil
import tensorflow as tf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..' '/src')))

from src.train import train
from src.utils import load_json
from src.model_definition import build_model


@pytest.mark.integration_test
def test_memory_usage_model():
    process = psutil.Process(os.getpid())
    
    path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    with open(os.path.join(path, "src", "params.yaml"), 'r') as file:
        params = yaml.safe_load(file)
    
    X_train = np.load(os.path.join(path, "data", "preprocess", "X_train.npy"))
    y_train = np.load(os.path.join(path, "data", "preprocess", "y_train.npy"))
    X_val = np.load(os.path.join(path, "data", "preprocess", "X_val.npy"))
    y_val = np.load(os.path.join(path, "data", "preprocess", "y_val.npy"))
    char_index = load_json(os.path.join(path, "data", "preprocess", "char_index.json"))
    
    mem_info_before = process.memory_info()
    mem_usage_before = mem_info_before.rss
    
    
    model = build_model(char_index, params['categories'])
    
    trained_model = train(model, X_train, y_train, X_val, y_val, params)
    
    mem_info_after = process.memory_info()
    mem_usage_after = mem_info_after.rss
    
    mem_usage = mem_usage_after - mem_usage_before
    
    #trained_model.save(os.path.join(path, "models", "trained_model_test.keras"))
    print(trained_model.summary())
    print(mem_usage)
    
    assert mem_usage < 4000000000 #4GB