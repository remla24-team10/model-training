"""
This script performs phishing detection using a machine learning model. It loads data,
preprocesses it, builds a model, trains the model, evaluates its performance, and plots
a confusion matrix.

Usage:
    Ensure that the parameters are specified in the 'params.yaml' file located in the
    'phishing-detection/phishing_detection' directory. Then run this script.

Example:
    python phishing-detection/phishing_detection/run.py
"""
import os
import yaml
from train import train
from model_definition import build_model
from predict import evaluate_results, plot_confusion_matrix, predict_classes
from lib_ml_remla import split_data, preprocess_data
from utils import load_data_from_text

# Disable oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def run(params: dict, data_path: str) -> None:
    """
    Runs the model with the given parameters.

    Parameters:
        params: A dictionary containing parameters for the model training and evaluation.
        data_path: Path to data directory

    Returns:
        None

    Example:
        params = yaml.safe_load(path)
        run(params)
    """
    train = load_data_from_text(os.path.join(data_path, "train.txt"))
    test = load_data_from_text(os.path.join(data_path, "test.txt"))
    val = load_data_from_text(os.path.join(data_path, "val.txt"))

    # Split data
    raw_X_train, raw_y_train, raw_X_val, raw_y_val, raw_X_test, raw_y_test = split_data(train, test, val)
    # Preprocess data
    X_train, y_train, X_val, y_val, X_test, y_test, char_index = preprocess_data(
        raw_X_train, raw_y_train, raw_X_val, raw_y_val, raw_X_test, raw_y_test)

    # Create model
    model = build_model(char_index, params)

    # Train model
    model = train(model, X_train, y_train, X_val, y_val, params)

    # Evaluate model
    prediction = predict_classes(model, X_test)
    evaluation_results = evaluate_results(y_test, prediction)

    # plot confusion matrix
    plot_confusion_matrix(evaluation_results['confusion_matrix']) #save fig?
 

def main():
    with open(os.path.join("src","params.yaml")) as file:
        params = yaml.safe_load(file)
    data_path = params["dataset_dir"]
    run(params, data_path)

if __name__ == "__main__":
    main()