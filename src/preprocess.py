"""
Provides functions to preprocess data.

"""
import os
import sys
import utils
import numpy as np
from sklearn.preprocessing import LabelEncoder
from lib_ml_remla import split_data, preprocess_data


# Disable oneDNN custom operations
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def main():
    """
    Preprocess data and save result to file.

    Returns:
        None
    """
    path = sys.argv[1]
    
    # Load data from text files
    train = utils.load_data_from_text(os.path.join(path, "raw", "train.txt"))
    train = [line.strip() for line in train[1:]]
    test = utils.load_data_from_text(os.path.join(path, "raw", "test.txt"))
    test = [line.strip() for line in test]
    val = utils.load_data_from_text(os.path.join(path, "raw", "val.txt"))
    val = [line.strip() for line in val]


    raw_X_train, raw_y_train, raw_X_val, raw_y_val, raw_X_test, raw_y_test = split_data(train, test, val)

    X_train, y_train, X_val, y_val, X_test, y_test, char_index = preprocess_data(
        raw_X_train, raw_y_train, raw_X_val, raw_y_val, raw_X_test, raw_y_test)


    np.save(os.path.join(path, "preprocess", "X_train.npy"), X_train)
    np.save(os.path.join(path, "preprocess", "y_train.npy"), y_train)
    np.save(os.path.join(path, "preprocess", "X_val.npy"), X_val)
    np.save(os.path.join(path, "preprocess", "y_val.npy"), y_val)
    np.save(os.path.join(path, "preprocess", "X_test.npy"), X_test)
    np.save(os.path.join(path, "preprocess", "y_test.npy"), y_test)
    utils.save_json(char_index, os.path.join(path, "preprocess", "char_index.json"))


if __name__ == "__main__":
    main()