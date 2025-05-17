"""
Configuration Manager

This module provides a configuration manager that creates configuration objects
for each component in the ML pipeline.
"""
from pathlib import Path
from InsurancePremiumPrediction.constants import CONFIG_FILE_PATH, SCHEMA_FILE_PATH
from InsurancePremiumPrediction.utils import read_yaml
from InsurancePremiumPrediction.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    ModelTrainerConfig,
    ModelEvaluationConfig,
    ModelDeploymentConfig
)


class ConfigurationManager:
    """
    Configuration manager for the ML pipeline.

    This class is responsible for creating and providing configuration objects
    for each component in the ML pipeline.
    """

    def __init__(
        self,
        config_filepath=CONFIG_FILE_PATH,
        schema_filepath=SCHEMA_FILE_PATH
    ):
        """
        Initialize the configuration manager.

        Args:
            config_filepath (Path): Path to the configuration file
            schema_filepath (Path): Path to the schema file
        """
        self.config = read_yaml(config_filepath)
        self.schema = read_yaml(schema_filepath)

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        """
        Get the configuration for data ingestion.

        Returns:
            DataIngestionConfig: Configuration for data ingestion
        """
        config = self.config.data_ingestion

        data_ingestion_config = DataIngestionConfig(
            root_dir=Path(config.root_dir),
            source_data=Path(config.source_data),
            local_data_file=Path(config.local_data_file),
            train_data_path=Path(config.train_data_path),
            test_data_path=Path(config.test_data_path),
            validation_data_path=Path(config.validation_data_path)
        )

        return data_ingestion_config

    def get_data_validation_config(self) -> DataValidationConfig:
        """
        Get the configuration for data validation.

        Returns:
            DataValidationConfig: Configuration for data validation
        """
        config = self.config.data_validation

        data_validation_config = DataValidationConfig(
            root_dir=Path(config.root_dir),
            status_file=Path(config.status_file),
            required_columns=config.required_columns
        )

        return data_validation_config

    def get_data_transformation_config(self) -> DataTransformationConfig:
        """
        Get the configuration for data transformation.

        Returns:
            DataTransformationConfig: Configuration for data transformation
        """
        config = self.config.data_transformation

        data_transformation_config = DataTransformationConfig(
            root_dir=Path(config.root_dir),
            preprocessor_path=Path(config.preprocessor_path),
            transformed_train_path=Path(config.transformed_train_path),
            transformed_test_path=Path(config.transformed_test_path),
            transformed_validation_path=Path(
                config.transformed_validation_path)
        )

        return data_transformation_config

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        """
        Get the configuration for model trainer.

        Returns:
            ModelTrainerConfig: Configuration for model trainer
        """
        config = self.config.model_trainer

        model_trainer_config = ModelTrainerConfig(
            root_dir=Path(config.root_dir),
            model_path=Path(config.model_path),
            train_metrics_path=Path(config.train_metrics_path),
            test_metrics_path=Path(config.test_metrics_path)
        )

        return model_trainer_config

    def get_model_evaluation_config(self) -> ModelEvaluationConfig:
        """
        Get the configuration for model evaluation.

        Returns:
            ModelEvaluationConfig: Configuration for model evaluation
        """
        config = self.config.model_evaluation

        model_evaluation_config = ModelEvaluationConfig(
            root_dir=Path(config.root_dir),
            metrics_path=Path(config.metrics_path),
            params_path=Path(config.params_path)
        )

        return model_evaluation_config

    def get_model_deployment_config(self) -> ModelDeploymentConfig:
        """
        Get the configuration for model deployment.

        Returns:
            ModelDeploymentConfig: Configuration for model deployment
        """
        model_trainer_config = self.config.model_trainer

        model_deployment_config = ModelDeploymentConfig(
            model_path=Path(model_trainer_config.model_path),
            preprocessor_path=Path(
                self.config.data_transformation.preprocessor_path),
            schema_path=SCHEMA_FILE_PATH
        )

        return model_deployment_config
