"""
Provides functions to create the model.

"""
import os
import sys
import utils as utils
import yaml
from keras._tf_keras.keras.models import Sequential, Model
from keras._tf_keras.keras.layers import Embedding, Conv1D, MaxPooling1D, Flatten, Dense, Dropout

# Disable oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def build_model(char_index: dict, categories: list[str], dropout_rate: float = 0.2) -> Model:
    """
    Build a model for the phishing detection task
    
    Args:
        char_index: A dictionary mapping characters to their index.
        categories: A list of categories to predict.
        dropout_rate: The dropout rate.

    Returns:
        A Keras model.

    """
    voc_size = len(char_index.keys())

    model = Sequential()
    model.add(Embedding(voc_size + 1, 50))

    model.add(Conv1D(128, 3, activation='tanh'))
    model.add(MaxPooling1D(3))
    model.add(Dropout(dropout_rate))

    model.add(Conv1D(128, 7, activation='tanh', padding='same'))
    model.add(Dropout(dropout_rate))

    model.add(Conv1D(128, 5, activation='tanh', padding='same'))
    model.add(Dropout(dropout_rate))

    model.add(Conv1D(128, 3, activation='tanh', padding='same'))
    model.add(MaxPooling1D(3))
    model.add(Dropout(dropout_rate))

    model.add(Conv1D(128, 5, activation='tanh', padding='same'))
    model.add(Dropout(dropout_rate))

    model.add(Conv1D(128, 3, activation='tanh', padding='same'))
    model.add(MaxPooling1D(3))
    model.add(Dropout(dropout_rate))

    model.add(Conv1D(128, 3, activation='tanh', padding='same'))
    model.add(MaxPooling1D(3))
    model.add(Dropout(dropout_rate))

    model.add(Flatten())

    model.add(Dense(len(categories)-1, activation='sigmoid'))

    return model
