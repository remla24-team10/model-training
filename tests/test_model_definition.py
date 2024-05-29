import sys
import os
# Set the path to the src directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
from src import model_definition
import pytest
import keras


@pytest.mark.fast
def test_build_model():
    model = model_definition.build_model({'a': 1, 'b': 2, 'c': 3}, ['label1', 'label2'])
    assert isinstance(model, keras.models.Model)
    assert len(model.layers) == 21
    # Ensure that all the layers are correct
    expected_layer_types = [
        keras.layers.Embedding,
        keras.layers.Conv1D,
        keras.layers.MaxPooling1D,
        keras.layers.Dropout,
        keras.layers.Conv1D,
        keras.layers.Dropout,
        keras.layers.Conv1D,
        keras.layers.Dropout,
        keras.layers.Conv1D,
        keras.layers.MaxPooling1D,
        keras.layers.Dropout,
        keras.layers.Conv1D,
        keras.layers.Dropout,
        keras.layers.Conv1D,
        keras.layers.MaxPooling1D,
        keras.layers.Dropout,
        keras.layers.Conv1D,
        keras.layers.MaxPooling1D,
        keras.layers.Dropout,
        keras.layers.Flatten,
        keras.layers.Dense
    ]
    for i, layer in enumerate(model.layers):
        assert isinstance(layer, expected_layer_types[i])