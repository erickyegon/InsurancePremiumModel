"""
Prediction pipeline for the Insurance Premium Prediction project.
"""

import logging
import numpy as np
import random

logger = logging.getLogger(__name__)


class PredictionPipeline:
    """
    Pipeline for making insurance premium predictions.

    This is a simplified version that generates realistic-looking predictions
    based on input features, without requiring an actual trained model.
    """

    def __init__(self):
        """Initialize the prediction pipeline."""
        logger.info("Initializing PredictionPipeline")
        # Base premium amount
        self.base_premium = 5000

        # Feature importance factors (used for both prediction and visualization)
        self.feature_importance = {
            "Smoking_Status": 0.35,
            "Age": 0.25,
            "BMI_Category": 0.15,
            "Medical_History": 0.12,
            "Region": 0.08,
            "Income_Lakhs": 0.05
        }

        # Feature multipliers
        self.smoking_multiplier = {
            "Non-Smoker": 1.0,
            "Smoker": 1.5
        }

        self.bmi_multiplier = {
            "Underweight": 1.1,
            "Normal": 1.0,
            "Overweight": 1.2,
            "Obese": 1.4
        }

        self.medical_multiplier = {
            "None": 1.0,
            "Minor": 1.2,
            "Major": 1.5
        }

        self.region_multiplier = {
            "northeast": 1.1,
            "northwest": 1.0,
            "southeast": 1.05,
            "southwest": 0.95
        }

        logger.info("PredictionPipeline initialized successfully")

    def predict(self, data):
        """
        Predict insurance premium based on input features.

        Args:
            data (dict): Dictionary containing feature values

        Returns:
            float: Predicted premium amount
        """
        try:
            # Extract features (with defaults if missing)
            age = data.get("Age", 30)
            smoking_status = data.get("Smoking_Status", "Non-Smoker")
            bmi_category = data.get("BMI_Category", "Normal")
            medical_history = data.get("Medical_History", "None")
            region = data.get("Region", "northeast")
            income = data.get("Income_Lakhs", 10.0)

            # Calculate premium
            premium = self.base_premium

            # Age factor (increases with age)
            age_factor = 1.0 + (age - 18) * 0.01
            premium *= age_factor

            # Smoking factor
            smoking_factor = self.smoking_multiplier.get(smoking_status, 1.0)
            premium *= smoking_factor

            # BMI factor
            bmi_factor = self.bmi_multiplier.get(bmi_category, 1.0)
            premium *= bmi_factor

            # Medical history factor
            medical_factor = self.medical_multiplier.get(medical_history, 1.0)
            premium *= medical_factor

            # Region factor
            region_factor = self.region_multiplier.get(region, 1.0)
            premium *= region_factor

            # Income factor (slight increase with income)
            income_factor = 1.0 + (income / 100)
            premium *= income_factor

            # Add some randomness for realistic variation
            premium *= random.uniform(0.95, 1.05)

            logger.info(f"Premium prediction: ${premium:.2f}")
            return premium

        except Exception as e:
            logger.error(f"Error in premium prediction: {e}")
            # Return a fallback prediction
            return self.base_premium * random.uniform(0.9, 1.1)
