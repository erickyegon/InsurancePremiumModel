"""
Prediction Pipeline

This module contains the PredictionPipeline class which is responsible for:
1. Loading the model deployment component
2. Processing input data
3. Making predictions
"""
from typing import Dict, Any

from InsurancePremiumPrediction import logger
from InsurancePremiumPrediction.config.configuration import ConfigurationManager
from InsurancePremiumPrediction.components.model_deployment import ModelDeployment


class PredictionPipeline:
    """
    Prediction pipeline for the insurance premium prediction model.

    This class is responsible for making predictions using the trained model.
    """

    def __init__(self):
        """Initialize the prediction pipeline."""
        self.config_manager = ConfigurationManager()
        self.model_deployment = None

    def predict(self, input_data: Dict[str, Any]) -> float:
        """
        Make a prediction for the input data.

        Args:
            input_data (Dict[str, Any]): Input data as a dictionary

        Returns:
            float: Predicted insurance premium
        """
        logger.info("Making prediction using prediction pipeline")

        try:
            # Initialize model deployment if not already initialized
            if self.model_deployment is None:
                model_deployment_config = self.config_manager.get_model_deployment_config()
                self.model_deployment = ModelDeployment(
                    config=model_deployment_config)
                self.model_deployment.initiate_model_deployment()

            # Make prediction
            prediction = self.model_deployment.predict(input_data)

            return prediction

        except Exception as e:
            logger.error(f"Error in prediction pipeline: {e}")
            raise e
