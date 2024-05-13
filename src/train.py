"""
Provides functions for training a model.

"""
import sys
import os
import yaml
import numpy as np
from model_definition import build_model
from utils import load_json
from keras._tf_keras.keras import Model
from keras._tf_keras.keras.models import load_model

# Disable oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def train(model: Model, X_train: np.array, y_train: np.array,
          X_val: np.array, y_val: np.array, params: dict) -> Model:
    """
    Train the model.

    Args:
        model: A Keras model.
        x_train: The training data.
        y_train: The training labels.
        x_val: The validation data.
        y_val: The validation labels.
        params: A dictionary containing the parameters for the model.

    Returns:
        The history object returned by model.fit().

    """
    model.compile(loss=params['loss_function'], optimizer=params['optimizer'], metrics=['accuracy'])


    model.fit(X_train, y_train,
                batch_size=params['batch_train'],
                epochs=params['epoch'],
                shuffle=True,
                validation_data=(X_val, y_val)
                )

    return model

def main():
    """
    Train and save model.

    Returns:
        None
    """
    path = sys.argv[1]
    params_file = sys.argv[2]

    X_train = np.load(os.path.join(path, "preprocess", "X_train.npy"))
    y_train = np.load(os.path.join(path, "preprocess", "y_train.npy"))
    X_val = np.load(os.path.join(path, "preprocess", "X_val.npy"))
    y_val = np.load(os.path.join(path, "preprocess", "y_val.npy"))
    char_index = load_json(os.path.join(path, "preprocess", "char_index.json"))

    with open(params_file, "r") as file:
        params = yaml.safe_load(file)

    model = build_model(char_index, params['categories'])

    trained_model = train(model, X_train, y_train, X_val, y_val, params)

    trained_model.save(os.path.join("models", "trained_model.keras"))

if __name__ == "__main__":
    main()