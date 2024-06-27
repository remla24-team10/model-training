"""
Provides functions for training a model.

"""

import os
import sys

import boto3
import numpy as np
import requests
import yaml
from keras._tf_keras.keras import Model
from keras._tf_keras.keras.saving import load_model

from .model_definition import build_model
from .utility_functions import load_json

# Disable oneDNN custom operations
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"


def get_next_version(bucket_name: str, model_name: str) -> str:  # pragma: no cover
    """Get the next version of the model from S3."""
    s3_client = boto3.client("s3")
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=f"{model_name}/")

    versions = []
    if "Contents" in response:
        for obj in response["Contents"]:
            key = obj["Key"]
            if key.startswith(f"{model_name}/{model_name}_v") and key.endswith(
                ".keras"
            ):
                version = key.split("_v")[-1].split(".keras")[0]
                versions.append(version)

    if not versions:
        return "v1.0"

    versions = sorted(versions, key=lambda x: list(map(int, x.split("."))))
    latest_version = versions[-1]
    major, minor = map(int, latest_version.split("."))
    next_version = f"v{major}.{minor + 1}"
    return next_version


def get_latest_version(bucket_name: str, model_name: str) -> str:  # pragma: no cover
    """Get the latest version of the model from S3."""
    s3_client = boto3.client("s3")
    response = s3_client.list_objects_v2(Bucket=bucket_name, Prefix=f"{model_name}/")

    versions = []
    if "Contents" in response:
        for obj in response["Contents"]:
            key = obj["Key"]
            if key.startswith(f"{model_name}/{model_name}_v") and key.endswith(
                ".keras"
            ):
                version = key.split("_v")[-1].split(".keras")[0]
                versions.append(version)

    if not versions:
        raise ValueError("No versions found for the specified model.")

    versions = sorted(versions, key=lambda x: list(map(int, x.split("."))))
    latest_version = versions[-1]
    return latest_version


def save_model_to_s3(
    model: Model, bucket_name: str, model_name: str, model_dir: str = "models"
):  # pragma: no cover
    """Save the model to S3."""
    # Get the next version dynamically
    version = get_next_version(bucket_name, model_name)

    # Create the model path
    model_path = os.path.join(model_dir, f"{model_name}_{version}.keras")
    os.makedirs(model_dir, exist_ok=True)

    # Save the model locally
    model.save(model_path)

    # Initialize S3 client
    s3_client = boto3.client("s3")

    # Upload the model to S3
    s3_client.upload_file(
        model_path, bucket_name, f"{model_name}/{model_name}_{version}.keras"
    )

    # Make the model file publicly accessible
    s3_client.put_object_acl(
        ACL="public-read",
        Bucket=bucket_name,
        Key=f"{model_name}/{model_name}_{version}.keras",
    )

    # Generate the public URL
    public_url = f"https://{bucket_name}.s3.amazonaws.com/{model_name}/{model_name}_{version}.keras"

    print(f"Model saved to S3: {public_url}")


def download_model_from_s3(
    bucket_name: str, model_name: str, download_dir: str = "downloaded_models"
):  # pragma: no cover
    """Download the latest model from S3."""
    # Get the latest version dynamically
    version = get_latest_version(bucket_name, model_name)

    # Create the public URL for the latest version
    public_url = (
        f"https://{bucket_name}.s3.amazonaws.com/"
        f"{model_name}/{model_name}_v{version}.keras"
    )

    # Create the download directory if it doesn't exist
    os.makedirs(download_dir, exist_ok=True)

    # Extract the file name from the URL
    file_name = public_url.split("/")[-1]
    local_path = os.path.join(download_dir, file_name)

    # Download the file
    try:
        response = requests.get(public_url, timeout=10)
    except requests.exceptions.RequestException as e:
        print(f"Failed to download model from {public_url}")
        raise SystemExit(e) from e
    response.raise_for_status()  # Ensure we notice bad responses

    # Save the file locally
    with open(local_path, "wb") as f:
        f.write(response.content)

    print(f"Model {version} downloaded from {public_url} to {local_path}")

    return local_path


def train(
    model: Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    params: dict,
) -> Model:
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
    model.compile(
        loss=params["loss_function"],
        optimizer=params["optimizer"],
        metrics=["accuracy"],
    )

    model.fit(
        X_train,
        y_train,
        batch_size=params["batch_train"],
        epochs=params["epoch"],
        shuffle=True,
        validation_data=(X_val, y_val),
    )

    return model


def main():  # pragma: no cover
    """
    Train and saves the model to models/trained_model.keras.

    Returns:
        None
    """
    path = sys.argv[1]
    params_file = sys.argv[2]
    save_model = sys.argv[3].lower() == "true"

    X_train = np.load(os.path.join(path, "preprocess", "X_train.npy"))
    y_train = np.load(os.path.join(path, "preprocess", "y_train.npy"))
    X_val = np.load(os.path.join(path, "preprocess", "X_val.npy"))
    y_val = np.load(os.path.join(path, "preprocess", "y_val.npy"))
    char_index = load_json(os.path.join(path, "preprocess", "char_index.json"))

    with open(params_file, "r", encoding="utf-8") as file:
        params = yaml.safe_load(file)

    model = build_model(char_index, params["categories"])

    trained_model = train(model, X_train, y_train, X_val, y_val, params)

    trained_model.save(os.path.join("models", "trained_model.keras"))

    trained_model = load_model(os.path.join("models", "trained_model.keras"))

    bucket_name = "remla10-phishing-detector-model"
    model_name = "phishing_detector"
    if save_model:
        model_name = "phishing_detector"
        save_model_to_s3(trained_model, bucket_name, model_name)

    # code for saving
    # downloaded_model_path = download_model_from_s3(bucket_name, model_name)
    # print(downloaded_model_path)


if __name__ == "__main__":
    main()
