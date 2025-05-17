"""
Data Transformation Component

This module contains the DataTransformation class which is responsible for:
1. Feature engineering
2. Data preprocessing
3. Handling categorical variables
4. Scaling numerical features
5. Creating and saving the preprocessor
"""
import os
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from InsurancePremiumPrediction import logger
from InsurancePremiumPrediction.entity.config_entity import DataTransformationConfig
from InsurancePremiumPrediction.constants import (
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    TARGET_COLUMN
)
from InsurancePremiumPrediction.utils import save_model


class DataTransformation:
    """
    Class for data transformation operations.

    This class handles:
    1. Feature engineering
    2. Data preprocessing
    3. Handling categorical variables
    4. Scaling numerical features
    5. Creating and saving the preprocessor
    """

    def __init__(self, config: DataTransformationConfig):
        """
        Initialize the DataTransformation class.

        Args:
            config (DataTransformationConfig): Configuration for data transformation
        """
        self.config = config

        # Create root directory
        os.makedirs(self.config.root_dir, exist_ok=True)

    def get_data_transformer(self):
        """
        Create a data transformer for preprocessing.

        Returns:
            ColumnTransformer: The data transformer
        """
        logger.info("Creating data transformer")

        # Numerical pipeline
        num_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ]
        )

        # Categorical pipeline
        cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder(drop='first',
                 sparse_output=False, handle_unknown='ignore'))
            ]
        )

        # Combine pipelines
        preprocessor = ColumnTransformer(
            [
                ("num_pipeline", num_pipeline, NUMERICAL_FEATURES),
                ("cat_pipeline", cat_pipeline, CATEGORICAL_FEATURES)
            ],
            remainder='drop'  # Drop columns not specified in the transformer
        )

        logger.info("Data transformer created")
        return preprocessor

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Engineer features for the dataset.

        Args:
            df (pd.DataFrame): The input dataframe

        Returns:
            pd.DataFrame: The dataframe with engineered features
        """
        logger.info("Engineering features")

        # Create a copy to avoid modifying the original dataframe
        df_copy = df.copy()

        # Create age group
        df_copy['age_group'] = pd.cut(
            df_copy['Age'],
            bins=[0, 30, 50, float('inf')],
            labels=['Young Adult', 'Adult', 'Senior']
        )

        # Create smoker interaction terms (map Smoker to 1, Non-Smoker to 0)
        # Handle missing values by filling with 'Non-Smoker'
        df_copy['Smoking_Status'] = df_copy['Smoking_Status'].fillna(
            'Non-Smoker')
        df_copy['smoker_numeric'] = df_copy['Smoking_Status'].map(
            {'Smoker': 1, 'Non-Smoker': 0})

        # Create interaction features
        df_copy['smoker_age'] = df_copy['smoker_numeric'] * df_copy['Age']
        df_copy['smoker_income'] = df_copy['smoker_numeric'] * \
            df_copy['Income_Lakhs']

        # Create family status (married with dependents, etc.)
        df_copy['family_status'] = 'Single'
        df_copy.loc[(df_copy['Marital_status'] == 'Married') & (
            df_copy['Number Of Dependants'] > 0), 'family_status'] = 'Family'
        df_copy.loc[(df_copy['Marital_status'] == 'Married') & (
            df_copy['Number Of Dependants'] == 0), 'family_status'] = 'Couple'
        df_copy.loc[(df_copy['Marital_status'] == 'Unmarried') & (
            df_copy['Number Of Dependants'] > 0), 'family_status'] = 'Single Parent'

        # Create income to age ratio (higher ratio might indicate higher premium)
        df_copy['income_age_ratio'] = df_copy['Income_Lakhs'] / df_copy['Age']

        # Create medical risk score (simple version)
        risk_mapping = {
            'None': 0,
            'Asthma': 1,
            'Hypertension': 2,
            'Diabetes': 3,
            'Heart Disease': 4
        }
        df_copy['medical_risk_score'] = df_copy['Medical History'].map(
            risk_mapping).fillna(0)

        # Create plan tier (numeric representation of insurance plan)
        plan_mapping = {
            'Bronze': 1,
            'Silver': 2,
            'Gold': 3,
            'Platinum': 4
        }
        df_copy['plan_tier'] = df_copy['Insurance_Plan'].map(plan_mapping)

        # Fill missing values in categorical columns
        for col in CATEGORICAL_FEATURES:
            if col in df_copy.columns and df_copy[col].isnull().sum() > 0:
                df_copy[col] = df_copy[col].fillna(df_copy[col].mode()[0])

        # Fill missing values in numerical columns
        for col in NUMERICAL_FEATURES:
            if col in df_copy.columns and df_copy[col].isnull().sum() > 0:
                df_copy[col] = df_copy[col].fillna(df_copy[col].median())

        logger.info("Feature engineering completed")
        return df_copy

    def transform_data(self, X_train, X_test, X_val):
        """
        Transform the data using the preprocessor.

        Args:
            X_train: Training features
            X_test: Test features
            X_val: Validation features

        Returns:
            tuple: Transformed training, test, and validation features
        """
        logger.info("Transforming data")

        # Get the preprocessor
        preprocessor = self.get_data_transformer()

        # Fit on training data
        X_train_transformed = preprocessor.fit_transform(X_train)

        # Transform test and validation data
        X_test_transformed = preprocessor.transform(X_test)
        X_val_transformed = preprocessor.transform(X_val)

        # Save the preprocessor
        save_model(self.config.preprocessor_path, preprocessor)
        logger.info(f"Preprocessor saved to {self.config.preprocessor_path}")

        logger.info("Data transformation completed")
        return X_train_transformed, X_test_transformed, X_val_transformed, preprocessor

    def initiate_data_transformation(self, train_path: Path, test_path: Path, validation_path: Path):
        """
        Initiate the data transformation process.

        Args:
            train_path (Path): Path to the training dataset
            test_path (Path): Path to the test dataset
            validation_path (Path): Path to the validation dataset

        Returns:
            tuple: Transformed data and the preprocessor
        """
        logger.info("Initiating data transformation")

        try:
            # Load datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            val_df = pd.read_csv(validation_path)

            logger.info(f"Loaded train dataset with shape: {train_df.shape}")
            logger.info(f"Loaded test dataset with shape: {test_df.shape}")
            logger.info(
                f"Loaded validation dataset with shape: {val_df.shape}")

            # Engineer features
            train_df = self.engineer_features(train_df)
            test_df = self.engineer_features(test_df)
            val_df = self.engineer_features(val_df)

            # Separate features and target
            target_column = TARGET_COLUMN

            # Apply log transformation to target for better model performance
            train_df[target_column] = np.log1p(train_df[target_column])
            test_df[target_column] = np.log1p(test_df[target_column])
            val_df[target_column] = np.log1p(val_df[target_column])

            # Split into features and target
            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            X_val = val_df.drop(columns=[target_column])
            y_val = val_df[target_column]

            # Transform the data
            X_train_transformed, X_test_transformed, X_val_transformed, preprocessor = self.transform_data(
                X_train, X_test, X_val
            )

            # Create directories if they don't exist
            os.makedirs(os.path.dirname(
                self.config.transformed_train_path), exist_ok=True)
            os.makedirs(os.path.dirname(
                self.config.transformed_test_path), exist_ok=True)
            os.makedirs(os.path.dirname(
                self.config.transformed_validation_path), exist_ok=True)

            # Convert transformed arrays to dataframes for saving
            # Get feature names from the preprocessor
            feature_names = []

            # Get numerical feature names (they remain the same)
            feature_names.extend(NUMERICAL_FEATURES)

            # Get one-hot encoded feature names for categorical features
            for cat_feature in CATEGORICAL_FEATURES:
                # Get unique values for the categorical feature
                unique_values = train_df[cat_feature].unique()
                # Skip the first value (as we're using drop='first' in OneHotEncoder)
                for value in sorted(unique_values)[1:]:
                    # Clean feature names to avoid XGBoost errors
                    # Replace special characters with underscores
                    clean_feature = cat_feature.replace(
                        ' ', '_').replace('-', '_')
                    clean_value = str(value).replace(' ', '_').replace(
                        '-', '_').replace('<', 'lt').replace('>', 'gt').replace('&', 'and')
                    feature_names.append(f"{clean_feature}_{clean_value}")

            # Create dataframes with transformed data
            train_transformed_df = pd.DataFrame(
                X_train_transformed, columns=feature_names)
            train_transformed_df[target_column] = y_train.values

            test_transformed_df = pd.DataFrame(
                X_test_transformed, columns=feature_names)
            test_transformed_df[target_column] = y_test.values

            val_transformed_df = pd.DataFrame(
                X_val_transformed, columns=feature_names)
            val_transformed_df[target_column] = y_val.values

            # Save transformed data
            train_transformed_df.to_csv(
                self.config.transformed_train_path, index=False)
            test_transformed_df.to_csv(
                self.config.transformed_test_path, index=False)
            val_transformed_df.to_csv(
                self.config.transformed_validation_path, index=False)

            logger.info(
                f"Transformed train data saved to {self.config.transformed_train_path}")
            logger.info(
                f"Transformed test data saved to {self.config.transformed_test_path}")
            logger.info(
                f"Transformed validation data saved to {self.config.transformed_validation_path}")

            # Return transformed data and preprocessor
            return (
                train_transformed_df,
                test_transformed_df,
                val_transformed_df,
                preprocessor
            )

        except Exception as e:
            logger.error(f"Error in data transformation: {e}")
            raise e
