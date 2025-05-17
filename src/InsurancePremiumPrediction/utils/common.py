import os
import yaml
import json
import joblib
import logging
from box import ConfigBox
from pathlib import Path
from typing import Any, Dict, List

# Get the logger directly instead of importing from the main package
# This avoids circular imports
logger = logging.getLogger("InsuranceLogger")


def read_yaml(path_to_yaml: Path) -> ConfigBox:
    """
    Reads a YAML file and returns its contents as a ConfigBox.

    Args:
        path_to_yaml (Path): Path to the YAML file

    Returns:
        ConfigBox: ConfigBox containing the YAML file contents

    Raises:
        ValueError: If the YAML file is empty
        Exception: If any other error occurs
    """
    try:
        with open(path_to_yaml) as yaml_file:
            content = yaml.safe_load(yaml_file)
            logger.info(f"YAML file: {path_to_yaml} loaded successfully")
            return ConfigBox(content)
    except Exception as e:
        logger.error(f"Error reading YAML file: {path_to_yaml}, error: {e}")
        raise e


def save_json(path: Path, data: Dict) -> None:
    """
    Saves data as a JSON file

    Args:
        path (Path): Path to the JSON file
        data (Dict): Data to be saved
    """
    try:
        with open(path, "w") as f:
            json.dump(data, f, indent=4)
        logger.info(f"JSON file saved at: {path}")
    except Exception as e:
        logger.error(f"Error saving JSON file: {path}, error: {e}")
        raise e


def load_json(path: Path) -> ConfigBox:
    """
    Loads a JSON file

    Args:
        path (Path): Path to the JSON file

    Returns:
        ConfigBox: ConfigBox containing the JSON file contents
    """
    try:
        with open(path) as f:
            content = json.load(f)
        logger.info(f"JSON file loaded from: {path}")
        return ConfigBox(content)
    except Exception as e:
        logger.error(f"Error loading JSON file: {path}, error: {e}")
        raise e


def save_model(path: Path, model: Any) -> None:
    """
    Saves a model using joblib

    Args:
        path (Path): Path to save the model
        model (Any): Model to be saved
    """
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(model, path)
        logger.info(f"Model saved at: {path}")
    except Exception as e:
        logger.error(f"Error saving model: {path}, error: {e}")
        raise e


def load_model(path: Path) -> Any:
    """
    Loads a model using joblib

    Args:
        path (Path): Path to the model

    Returns:
        Any: Loaded model
    """
    try:
        model = joblib.load(path)
        logger.info(f"Model loaded from: {path}")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {path}, error: {e}")
        raise e


def create_directories(path_to_directories: List[str], verbose=True) -> None:
    """
    Creates a list of directories

    Args:
        path_to_directories (List[str]): List of paths to create
        verbose (bool, optional): Whether to log the creation. Defaults to True.
    """
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        if verbose:
            logger.info(f"Directory created at: {path}")
