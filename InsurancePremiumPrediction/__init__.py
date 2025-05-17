"""
Insurance Premium Prediction package.

This package contains modules for predicting insurance premiums based on customer attributes.
"""

from InsurancePremiumPrediction.pipeline import PredictionPipeline
from InsurancePremiumPrediction.utils import read_yaml
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Create a logger for the package
logger = logging.getLogger(__name__)

# Import key modules

__all__ = ['read_yaml', 'PredictionPipeline', 'logger']
