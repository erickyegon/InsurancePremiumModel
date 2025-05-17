"""
Insurance Premium Prediction Model

This is the main entry point for the Insurance Premium Prediction application.
It orchestrates the entire ML pipeline from data ingestion to model evaluation.
"""
from InsurancePremiumPrediction import logger
from InsurancePremiumPrediction.pipeline.training_pipeline import TrainingPipeline


def main():
    """
    Main function that runs the complete ML pipeline.

    This function initializes and runs the training pipeline, which orchestrates
    all the components of the ML pipeline.

    Returns:
        None
    """
    logger.info("Starting Insurance Premium Prediction application")

    try:
        # Initialize and run the training pipeline
        pipeline = TrainingPipeline()
        pipeline.run_pipeline()

        logger.info("Application completed successfully")
    except Exception as e:
        logger.error(f"Error in main application: {e}")
        raise e


if __name__ == "__main__":
    main()
