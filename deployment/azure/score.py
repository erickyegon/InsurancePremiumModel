"""
Scoring Script for Azure ML

This script is used by Azure ML to load the model and make predictions.
"""
import os
import json
import numpy as np
import pandas as pd
import joblib
import logging
from azureml.core.model import Model
from azureml.monitoring import ModelDataCollector

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def init():
    """
    Initialize the model when the endpoint is created.
    """
    global model, input_collector, output_collector
    
    try:
        # Get model path
        model_path = Model.get_model_path(model_name='insurance-premium-model')
        logger.info(f"Model path: {model_path}")
        
        # Load model
        model = joblib.load(model_path)
        logger.info("Model loaded successfully")
        
        # Set up data collectors for monitoring
        input_collector = ModelDataCollector(
            model_name='insurance-premium-model',
            identifier='inputs',
            feature_names=None
        )
        
        output_collector = ModelDataCollector(
            model_name='insurance-premium-model',
            identifier='outputs',
            feature_names=None
        )
        
        logger.info("Initialization completed successfully")
    except Exception as e:
        logger.error(f"Error during initialization: {e}")
        raise

def run(raw_data):
    """
    Make predictions using the model.
    
    Args:
        raw_data: JSON string with input data
        
    Returns:
        JSON string with prediction results
    """
    try:
        # Parse input data
        data = json.loads(raw_data)
        logger.info(f"Received input data: {data}")
        
        # Convert to DataFrame
        input_df = pd.DataFrame([data])
        
        # Log input data for monitoring
        input_collector.collect(input_df)
        
        # Preprocess input data
        # In a real implementation, this would include feature engineering
        # For simplicity, we assume the input data is already preprocessed
        
        # Make prediction
        prediction = model.predict(input_df)
        prediction_value = float(prediction[0])
        
        # Create response
        response = {
            "prediction": prediction_value,
            "model_version": os.environ.get("AZUREML_MODEL_VERSION", "unknown")
        }
        
        # Log output data for monitoring
        output_collector.collect(prediction_value)
        
        logger.info(f"Prediction: {prediction_value}")
        return json.dumps(response)
    
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return json.dumps({
            "error": str(e)
        })
