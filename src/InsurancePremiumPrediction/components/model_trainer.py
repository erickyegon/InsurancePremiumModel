"""
Model Trainer Component

This module contains the ModelTrainer class which is responsible for:
1. Training different regression models
2. Hyperparameter tuning
3. Model evaluation
4. Saving the best model
"""
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import RandomizedSearchCV

from InsurancePremiumPrediction import logger
from InsurancePremiumPrediction.entity.config_entity import ModelTrainerConfig
from InsurancePremiumPrediction.constants import TARGET_COLUMN, PARAMS_FILE_PATH
from InsurancePremiumPrediction.utils import save_model, read_yaml


class ModelTrainer:
    """
    Class for model training operations.

    This class handles:
    1. Training different regression models
    2. Hyperparameter tuning
    3. Model evaluation
    4. Saving the best model
    """

    def __init__(self, config: ModelTrainerConfig, params_filepath=PARAMS_FILE_PATH):
        """
        Initialize the ModelTrainer class.

        Args:
            config (ModelTrainerConfig): Configuration for model trainer
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

    def train_and_evaluate_models(self, X_train, y_train, X_test, y_test):
        """
        Train and evaluate multiple regression models with hyperparameter tuning.

        Args:
            X_train: Training features
            y_train: Training target
            X_test: Test features
            y_test: Test target

        Returns:
            tuple: Best model and its metrics
        """
        logger.info("Training and evaluating models with hyperparameter tuning")

        # Get model parameters
        params = self.get_model_params()

        # Define hyperparameter grids for tuning
        param_grids = {
            "XGBoost": {
                'learning_rate': [0.01, 0.05, 0.1],
                'max_depth': [3, 5, 7],
                'n_estimators': [50, 100, 200],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            },
            "Random Forest": {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20, 30],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        }

        # Initialize base models with parameters
        base_models = {
            "Linear Regression": LinearRegression(**params.linear_regression),
            "Random Forest": RandomForestRegressor(**params.random_forest),
            "XGBoost": XGBRegressor(**params.xgboost),
            "SVR": SVR(**params.svr),
            "KNN": KNeighborsRegressor(**params.knn)
        }

        # Dictionary to store model metrics
        model_metrics = {}
        tuned_models = {}

        # Train and evaluate each model
        for model_name, model in base_models.items():
            logger.info(f"Training {model_name}")

            # Check if model should be tuned
            if model_name in param_grids:
                logger.info(
                    f"Performing hyperparameter tuning for {model_name}")

                # Use RandomizedSearchCV for faster tuning
                random_search = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=param_grids[model_name],
                    n_iter=5,  # Number of parameter settings sampled
                    cv=3,      # 3-fold cross-validation
                    scoring='r2',
                    n_jobs=-1,  # Use all available cores
                    random_state=42,
                    verbose=1
                )

                try:
                    # Fit the random search
                    random_search.fit(X_train, y_train)

                    # Get the best model
                    best_params = random_search.best_params_
                    logger.info(
                        f"Best parameters for {model_name}: {best_params}")

                    # Update the model with best parameters
                    if model_name == "XGBoost":
                        # Add safe parameters for XGBoost
                        best_params.update({
                            'tree_method': 'hist',  # Use histogram-based algorithm
                            'gpu_id': -1,  # Disable GPU
                            'objective': 'reg:squarederror'  # Ensure correct objective
                        })
                        model = XGBRegressor(**best_params)
                    elif model_name == "Random Forest":
                        model = RandomForestRegressor(**best_params)

                    # Train the tuned model
                    model.fit(X_train, y_train)
                    tuned_models[model_name] = model
                except Exception as e:
                    logger.error(f"Error tuning {model_name}: {e}")
                    logger.info(
                        f"Falling back to default parameters for {model_name}")

                    # Fall back to default parameters
                    if model_name == "XGBoost":
                        # Clean column names to avoid XGBoost errors
                        X_train_clean = X_train.copy()
                        X_train_clean.columns = [col.replace(' ', '_').replace(
                            '-', '_').replace('<', 'lt').replace('>', 'gt').replace('&', 'and') for col in X_train.columns]

                        model = XGBRegressor(
                            learning_rate=0.1,
                            max_depth=5,
                            n_estimators=100,
                            subsample=0.8,
                            colsample_bytree=0.8,
                            objective='reg:squarederror',
                            random_state=42,
                            tree_method='hist',
                            gpu_id=-1,
                            enable_categorical=False  # Disable categorical features to avoid errors
                        )
                    elif model_name == "Random Forest":
                        model = RandomForestRegressor(
                            n_estimators=100,
                            max_depth=10,
                            min_samples_split=2,
                            min_samples_leaf=1,
                            random_state=42
                        )

                    # Train with default parameters
                    if model_name == "XGBoost":
                        # Use the cleaned feature names for XGBoost
                        model.fit(X_train_clean, y_train)
                    else:
                        model.fit(X_train, y_train)
                    tuned_models[model_name] = model
            else:
                # Train the model without tuning
                model.fit(X_train, y_train)
                tuned_models[model_name] = model

            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Calculate metrics
            train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

            train_mae = mean_absolute_error(y_train, y_train_pred)
            test_mae = mean_absolute_error(y_test, y_test_pred)

            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2_score(y_test, y_test_pred)

            # Store metrics
            model_metrics[model_name] = {
                "train_rmse": train_rmse,
                "test_rmse": test_rmse,
                "train_mae": train_mae,
                "test_mae": test_mae,
                "train_r2": train_r2,
                "test_r2": test_r2
            }

            logger.info(
                f"{model_name} - Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}, Test R²: {test_r2:.4f}")

        # Find the best model based on test R²
        best_model_name = max(
            model_metrics, key=lambda x: model_metrics[x]["test_r2"])
        best_model = tuned_models[best_model_name]
        best_metrics = model_metrics[best_model_name]

        logger.info(
            f"Best model: {best_model_name} with Test R²: {best_metrics['test_r2']:.4f}")

        # Save all model metrics
        metrics_file = os.path.join(self.config.root_dir, "model_metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump(model_metrics, f, indent=4)

        logger.info(f"Model metrics saved to {metrics_file}")

        return best_model, best_metrics

    def save_metrics(self, metrics, file_path):
        """
        Save metrics to a JSON file.

        Args:
            metrics (dict): Metrics to save
            file_path (Path): Path to save the metrics
        """
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        # Save metrics to file
        with open(file_path, 'w') as f:
            json.dump(metrics, f, indent=4)

        logger.info(f"Metrics saved to {file_path}")

    def initiate_model_training(self, train_path: Path, test_path: Path):
        """
        Initiate the model training process.

        Args:
            train_path (Path): Path to the transformed training dataset
            test_path (Path): Path to the transformed test dataset

        Returns:
            tuple: Best model and its metrics
        """
        logger.info("Initiating model training")

        try:
            # Load datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logger.info(f"Loaded train dataset with shape: {train_df.shape}")
            logger.info(f"Loaded test dataset with shape: {test_df.shape}")

            # Separate features and target
            target_column = TARGET_COLUMN

            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            # Train and evaluate models
            best_model, metrics = self.train_and_evaluate_models(
                X_train, y_train, X_test, y_test)

            # Save the best model
            save_model(self.config.model_path, best_model)
            logger.info(f"Best model saved to {self.config.model_path}")

            # Save metrics
            self.save_metrics(metrics, self.config.train_metrics_path)
            self.save_metrics(metrics, self.config.test_metrics_path)

            # Return the best model and metrics
            return best_model, metrics

        except Exception as e:
            logger.error(f"Error in model training: {e}")
            raise e
