"""
Model Evaluation Component

This module contains the ModelEvaluation class which is responsible for:
1. Evaluating the trained model on validation data
2. Comparing model performance with previous models
3. Saving evaluation metrics
"""
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from InsurancePremiumPrediction import logger
from InsurancePremiumPrediction.entity.config_entity import ModelEvaluationConfig
from InsurancePremiumPrediction.constants import TARGET_COLUMN, PARAMS_FILE_PATH
from InsurancePremiumPrediction.utils import load_model, read_yaml, save_json


class ModelEvaluation:
    """
    Class for model evaluation operations.

    This class handles:
    1. Evaluating the trained model on validation data
    2. Comparing model performance with previous models
    3. Saving evaluation metrics
    """

    def __init__(self, config: ModelEvaluationConfig, params_filepath=PARAMS_FILE_PATH):
        """
        Initialize the ModelEvaluation class.

        Args:
            config (ModelEvaluationConfig): Configuration for model evaluation
            params_filepath (Path, optional): Path to the parameters file. Defaults to PARAMS_FILE_PATH.
        """
        self.config = config
        self.params_filepath = params_filepath

        # Create root directory
        os.makedirs(self.config.root_dir, exist_ok=True)

    def get_model_params(self):
        """
        Get model parameters from the parameters file.

        Returns:
            dict: Model parameters
        """
        params = read_yaml(self.params_filepath)
        return params

    def evaluate_model(self, model, X, y):
        """
        Evaluate the model on the given data.

        Args:
            model: The trained model
            X: Features
            y: Target

        Returns:
            dict: Evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X)

        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        # Create metrics dictionary
        metrics = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2
        }

        return metrics

    def initiate_model_evaluation(self, validation_path: Path, model_path: Path):
        """
        Initiate the model evaluation process.

        Args:
            validation_path (Path): Path to the validation dataset
            model_path (Path): Path to the trained model

        Returns:
            dict: Evaluation metrics
        """
        logger.info("Initiating model evaluation")

        try:
            # Load validation dataset
            validation_df = pd.read_csv(validation_path)
            logger.info(
                f"Loaded validation dataset with shape: {validation_df.shape}")

            # Separate features and target
            target_column = TARGET_COLUMN

            X_val = validation_df.drop(columns=[target_column])
            y_val = validation_df[target_column]

            # Load the trained model
            model = load_model(model_path)
            logger.info(f"Loaded model from {model_path}")

            # Get model parameters
            params = self.get_model_params()

            # Evaluate the model
            metrics = self.evaluate_model(model, X_val, y_val)
            logger.info(
                f"Model evaluation metrics: RMSE: {metrics['rmse']:.4f}, MAE: {metrics['mae']:.4f}, R²: {metrics['r2']:.4f}")

            # Save metrics
            save_json(self.config.metrics_path, metrics)
            logger.info(f"Metrics saved to {self.config.metrics_path}")

            # Save parameters
            save_json(self.config.params_path, params)
            logger.info(f"Parameters saved to {self.config.params_path}")

            # Return metrics
            return metrics

        except Exception as e:
            logger.error(f"Error in model evaluation: {e}")
            raise e
