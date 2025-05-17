"""
Configuration Entity Module

This module contains dataclasses that define the configuration for each component
in the ML pipeline. These classes are used to store and pass configuration parameters
between components.
"""
from dataclasses import dataclass
from pathlib import Path


@dataclass
class DataIngestionConfig:
    """Configuration for data ingestion component."""
    root_dir: Path
    source_data: Path
    local_data_file: Path
    train_data_path: Path
    test_data_path: Path
    validation_data_path: Path


@dataclass
class DataValidationConfig:
    """Configuration for data validation component."""
    root_dir: Path
    status_file: Path
    required_columns: list


@dataclass
class DataTransformationConfig:
    """Configuration for data transformation component."""
    root_dir: Path
    preprocessor_path: Path
    transformed_train_path: Path
    transformed_test_path: Path
    transformed_validation_path: Path


@dataclass
class ModelTrainerConfig:
    """Configuration for model trainer component."""
    root_dir: Path
    model_path: Path
    train_metrics_path: Path
    test_metrics_path: Path


@dataclass
class ModelEvaluationConfig:
    """Configuration for model evaluation component."""
    root_dir: Path
    metrics_path: Path
    params_path: Path


@dataclass
class ModelDeploymentConfig:
    """Configuration for model deployment component."""
    model_path: Path
    preprocessor_path: Path
    schema_path: Path
