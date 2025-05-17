"""
Model Deployment Component

This module contains the ModelDeployment class which is responsible for:
1. Loading the trained model and preprocessor
2. Creating prediction pipeline
3. Preparing the model for deployment
"""
from typing import Dict, Any
import numpy as np
import pandas as pd

from InsurancePremiumPrediction import logger
from InsurancePremiumPrediction.entity.config_entity import ModelDeploymentConfig
from InsurancePremiumPrediction.constants import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES
)
from InsurancePremiumPrediction.utils import load_model, read_yaml


class ModelDeployment:
    """
    Class for model deployment operations.

    This class handles:
    1. Loading the trained model and preprocessor
    2. Creating prediction pipeline
    3. Preparing the model for deployment
    """

    def __init__(self, config: ModelDeploymentConfig):
        """
        Initialize the ModelDeployment class.

        Args:
            config (ModelDeploymentConfig): Configuration for model deployment
        """
        self.config = config
        self.model = None
        self.preprocessor = None
        self.schema = None

    def load_artifacts(self):
        """
        Load the trained model, preprocessor, and schema.

        Returns:
            tuple: Model, preprocessor, and schema
        """
        logger.info("Loading artifacts for deployment")

        # Load model
        model = load_model(self.config.model_path)
        logger.info(f"Model loaded from {self.config.model_path}")

        # Load preprocessor
        preprocessor = load_model(self.config.preprocessor_path)
        logger.info(
            f"Preprocessor loaded from {self.config.preprocessor_path}")

        # Load schema
        schema = read_yaml(self.config.schema_path)
        logger.info(f"Schema loaded from {self.config.schema_path}")

        self.model = model
        self.preprocessor = preprocessor
        self.schema = schema

        return model, preprocessor, schema

    def engineer_features(self, input_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Engineer features for the input data.

        Args:
            input_data (Dict[str, Any]): Input data as a dictionary

        Returns:
            pd.DataFrame: DataFrame with engineered features
        """
        logger.info("Engineering features for prediction")

        # Convert input data to DataFrame
        df = pd.DataFrame([input_data])

        # Use consistent column names
        # Make sure we're using Number_Of_Dependants consistently
        if 'Number Of Dependants' in df.columns and 'Number_Of_Dependants' not in df.columns:
            df['Number_Of_Dependants'] = df['Number Of Dependants']

        # Make sure we're using Medical_History consistently
        if 'Medical History' in df.columns and 'Medical_History' not in df.columns:
            df['Medical_History'] = df['Medical History']

        # Create age group
        df['age_group'] = pd.cut(
            df['Age'],
            bins=[0, 30, 50, float('inf')],
            labels=['Young Adult', 'Adult', 'Senior']
        )

        # Create smoker interaction terms (map Smoker to 1, Non-Smoker to 0)
        # Handle missing values by filling with 'Non-Smoker'
        df['Smoking_Status'] = df['Smoking_Status'].fillna('Non-Smoker')

        # Map various smoking statuses to numeric values
        smoking_map = {
            'Smoker': 1, 'Regular': 1, 'Occasional': 0.5,
            'Non-Smoker': 0, 'Does Not Smoke': 0, 'No Smoking': 0,
            'Not Smoking': 0, 'Smoking=0': 0
        }
        df['smoker_numeric'] = df['Smoking_Status'].map(smoking_map).fillna(0)

        # Create interaction features
        df['smoker_age'] = df['smoker_numeric'] * df['Age']
        df['smoker_income'] = df['smoker_numeric'] * df['Income_Lakhs']

        # Create family status (married with dependents, etc.)
        df['family_status'] = 'Single'
        df.loc[(df['Marital_status'] == 'Married') & (
            df['Number_Of_Dependants'] > 0), 'family_status'] = 'Family'
        df.loc[(df['Marital_status'] == 'Married') & (
            df['Number_Of_Dependants'] == 0), 'family_status'] = 'Couple'
        df.loc[(df['Marital_status'] == 'Unmarried') & (
            df['Number_Of_Dependants'] > 0), 'family_status'] = 'Single Parent'

        # Create income to age ratio (higher ratio might indicate higher premium)
        df['income_age_ratio'] = df['Income_Lakhs'] / df['Age']

        # Create medical risk score (simple version)
        risk_mapping = {
            'None': 0, 'No Disease': 0,
            'Asthma': 1, 'Thyroid': 1,
            'Hypertension': 2, 'High blood pressure': 2,
            'Diabetes': 3, 'Heart Disease': 3, 'Heart disease': 3,
            'Diabetes & Thyroid': 4,
            'Diabetes & Heart disease': 5,
            'Diabetes & High blood pressure': 5,
            'High blood pressure & Heart disease': 5
        }
        df['medical_risk_score'] = df['Medical_History'].map(
            risk_mapping).fillna(0)

        # Create plan tier (numeric representation of insurance plan)
        plan_mapping = {
            'Bronze': 1,
            'Silver': 2,
            'Gold': 3,
            'Platinum': 4
        }
        df['plan_tier'] = df['Insurance_Plan'].map(plan_mapping)

        return df

    def predict(self, input_data: Dict[str, Any]) -> float:
        """
        Make a prediction for the input data.

        Args:
            input_data (Dict[str, Any]): Input data as a dictionary

        Returns:
            float: Predicted insurance premium
        """
        logger.info("Making prediction")

        # Load artifacts if not already loaded
        if self.model is None or self.preprocessor is None or self.schema is None:
            self.load_artifacts()

        try:
            # Engineer features
            df = self.engineer_features(input_data)

            # Select only the features used by the preprocessor
            # This includes only the original features, not the engineered ones
            try:
                # First, check if we need to rename any columns to match what the preprocessor expects
                # This is a temporary fix until we retrain the model
                if 'Number_Of_Dependants' in df.columns and 'Number Of Dependants' not in df.columns:
                    df['Number Of Dependants'] = df['Number_Of_Dependants']

                if 'Medical_History' in df.columns and 'Medical History' not in df.columns:
                    df['Medical History'] = df['Medical_History']

                # Try to get the features using the preprocessor column names
                from InsurancePremiumPrediction.constants import PREPROCESSOR_NUMERICAL_FEATURES, PREPROCESSOR_CATEGORICAL_FEATURES
                features_df = df[PREPROCESSOR_NUMERICAL_FEATURES +
                                 PREPROCESSOR_CATEGORICAL_FEATURES]
            except KeyError as e:
                logger.error(f"Missing required columns: {e}")
                # Try to map old column names to new column names if needed
                column_mapping = {
                    'age': 'Age',
                    'sex': 'Gender',
                    'bmi': 'BMI_Category',
                    'children': 'Number_Of_Dependants',
                    'smoker': 'Smoking_Status',
                    'region': 'Region',
                    'charges': 'Annual_Premium_Amount'
                }
                # Check if we need to do reverse mapping
                if set(NUMERICAL_FEATURES + CATEGORICAL_FEATURES).issubset(set(column_mapping.keys())):
                    # Create a new DataFrame with the expected column names
                    new_df = pd.DataFrame()
                    for old_col, new_col in column_mapping.items():
                        if new_col in df.columns:
                            new_df[old_col] = df[new_col]
                        else:
                            # If the new column name is not in the DataFrame, use a default value
                            if old_col == 'age':
                                new_df[old_col] = 30
                            elif old_col == 'sex':
                                new_df[old_col] = 'male'
                            elif old_col == 'bmi':
                                # Convert BMI_Category to numeric BMI
                                bmi_map = {'Underweight': 17.5, 'Normal': 22.0,
                                           'Overweight': 27.5, 'Obese': 32.5, 'Obesity': 35.0}
                                new_df[old_col] = df['BMI_Category'].map(
                                    bmi_map).fillna(25.0)
                            elif old_col == 'children':
                                new_df[old_col] = df['Number_Of_Dependants'] if 'Number_Of_Dependants' in df.columns else 0
                            elif old_col == 'smoker':
                                # Convert Smoking_Status to yes/no
                                smoker_map = {'Smoker': 'yes', 'Regular': 'yes', 'Occasional': 'yes',
                                              'Non-Smoker': 'no', 'Does Not Smoke': 'no', 'No Smoking': 'no',
                                              'Not Smoking': 'no', 'Smoking=0': 'no'}
                                new_df[old_col] = df['Smoking_Status'].map(
                                    smoker_map).fillna('no')
                            elif old_col == 'region':
                                # Convert Region to lowercase
                                region_map = {'Northeast': 'northeast', 'Northwest': 'northwest',
                                              'Southeast': 'southeast', 'Southwest': 'southwest'}
                                new_df[old_col] = df['Region'].map(
                                    region_map).fillna('northeast')
                    features_df = new_df
                else:
                    # If we can't map the columns, raise an error
                    missing_columns = set(
                        PREPROCESSOR_NUMERICAL_FEATURES + PREPROCESSOR_CATEGORICAL_FEATURES) - set(df.columns)
                    raise ValueError(
                        f"columns are missing: {missing_columns}")

            # Transform the features
            X = self.preprocessor.transform(features_df)

            # Make prediction
            prediction_log = self.model.predict(X)[0]

            # Convert from log scale back to original scale
            prediction = np.expm1(prediction_log)

            logger.info(f"Prediction made: ${prediction:.2f}")

            return prediction

        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise e

    def initiate_model_deployment(self):
        """
        Initiate the model deployment process.

        Returns:
            tuple: Model, preprocessor, and schema
        """
        logger.info("Initiating model deployment")

        try:
            # Load artifacts
            model, preprocessor, schema = self.load_artifacts()

            logger.info("Model deployment initiated successfully")

            return model, preprocessor, schema

        except Exception as e:
            logger.error(f"Error in model deployment: {e}")
            raise e
