"""
Training Pipeline

This module orchestrates the training pipeline for the insurance premium prediction model.
"""
from InsurancePremiumPrediction import logger
from InsurancePremiumPrediction.config.configuration import ConfigurationManager
from InsurancePremiumPrediction.components.data_ingestion import DataIngestion
from InsurancePremiumPrediction.components.data_validation import DataValidation
from InsurancePremiumPrediction.components.data_transformation import DataTransformation
from InsurancePremiumPrediction.components.model_trainer import ModelTrainer
from InsurancePremiumPrediction.components.model_evaluation import ModelEvaluation


class TrainingPipeline:
    """
    Training pipeline for the insurance premium prediction model.

    This class orchestrates the entire training pipeline, from data ingestion
    to model evaluation.
    """

    def __init__(self):
        """Initialize the training pipeline."""
        self.config_manager = ConfigurationManager()

    def start_data_ingestion(self):
        """
        Start the data ingestion process.

        Returns:
            tuple: Paths to the train, test, and validation data files
        """
        logger.info("Starting data ingestion")
        data_ingestion_config = self.config_manager.get_data_ingestion_config()
        data_ingestion = DataIngestion(config=data_ingestion_config)
        train_data_path, test_data_path, validation_data_path = data_ingestion.initiate_data_ingestion()
        return train_data_path, test_data_path, validation_data_path

    def start_data_validation(self, train_data_path, test_data_path, validation_data_path):
        """
        Start the data validation process.

        Args:
            train_data_path (Path): Path to the training dataset
            test_data_path (Path): Path to the test dataset
            validation_data_path (Path): Path to the validation dataset

        Returns:
            bool: True if validation passes, False otherwise
        """
        logger.info("Starting data validation")
        data_validation_config = self.config_manager.get_data_validation_config()
        data_validation = DataValidation(config=data_validation_config)
        validation_status = data_validation.initiate_data_validation(
            train_data_path, test_data_path, validation_data_path
        )
        return validation_status

    def start_data_transformation(self, train_data_path, test_data_path, validation_data_path):
        """
        Start the data transformation process.

        Args:
            train_data_path (Path): Path to the training dataset
            test_data_path (Path): Path to the test dataset
            validation_data_path (Path): Path to the validation dataset

        Returns:
            tuple: Transformed data and the preprocessor
        """
        logger.info("Starting data transformation")
        data_transformation_config = self.config_manager.get_data_transformation_config()
        data_transformation = DataTransformation(
            config=data_transformation_config)
        train_df, test_df, val_df, preprocessor = data_transformation.initiate_data_transformation(
            train_data_path, test_data_path, validation_data_path
        )
        return train_df, test_df, val_df, preprocessor

    def start_model_training(self, train_path, test_path):
        """
        Start the model training process.

        Args:
            train_path (Path): Path to the transformed training dataset
            test_path (Path): Path to the transformed test dataset

        Returns:
            tuple: Best model and its metrics
        """
        logger.info("Starting model training")
        model_trainer_config = self.config_manager.get_model_trainer_config()
        model_trainer = ModelTrainer(config=model_trainer_config)
        model, metrics = model_trainer.initiate_model_training(
            train_path, test_path)
        return model, metrics

    def start_model_evaluation(self, validation_path, model_path):
        """
        Start the model evaluation process.

        Args:
            validation_path (Path): Path to the validation dataset
            model_path (Path): Path to the trained model

        Returns:
            dict: Evaluation metrics
        """
        logger.info("Starting model evaluation")
        model_evaluation_config = self.config_manager.get_model_evaluation_config()
        model_evaluation = ModelEvaluation(config=model_evaluation_config)
        metrics = model_evaluation.initiate_model_evaluation(
            validation_path, model_path)
        return metrics

    def run_pipeline(self):
        """
        Run the complete training pipeline.

        This method orchestrates the entire ML pipeline, from data ingestion
        to model evaluation.
        """
        logger.info("Starting the training pipeline")

        try:
            # Data Ingestion
            train_data_path, test_data_path, validation_data_path = self.start_data_ingestion()

            # Data Validation
            validation_status = self.start_data_validation(
                train_data_path, test_data_path, validation_data_path
            )

            if not validation_status:
                logger.warning(
                    "Data validation failed. Proceeding with caution.")

            # Data Transformation
            train_df, test_df, val_df, preprocessor = self.start_data_transformation(
                train_data_path, test_data_path, validation_data_path
            )

            # Model Training
            model, train_metrics = self.start_model_training(
                self.config_manager.get_data_transformation_config().transformed_train_path,
                self.config_manager.get_data_transformation_config().transformed_test_path
            )

            # Model Evaluation
            eval_metrics = self.start_model_evaluation(
                self.config_manager.get_data_transformation_config().transformed_validation_path,
                self.config_manager.get_model_trainer_config().model_path
            )

            logger.info(f"Training metrics: {train_metrics}")
            logger.info(f"Evaluation metrics: {eval_metrics}")

            logger.info("Training pipeline completed successfully")

            return train_metrics, eval_metrics

        except Exception as e:
            logger.error(f"Error in training pipeline: {e}")
            raise e
