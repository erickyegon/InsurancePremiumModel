"""
Utility functions for the Insurance Premium Prediction project.
"""

import yaml
import logging

logger = logging.getLogger(__name__)


def read_yaml(path):
    """
    Read a YAML file and return its contents as a Python dictionary.

    Args:
        path (str): Path to the YAML file

    Returns:
        dict: Contents of the YAML file

    Raises:
        Exception: If the file cannot be read or parsed
    """
    try:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logger.error(f"Error reading YAML file {path}: {e}")
        raise
