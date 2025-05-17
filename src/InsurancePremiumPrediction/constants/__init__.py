"""
Constants Module

This module contains constants used throughout the project.
"""
from pathlib import Path

# Configuration paths
CONFIG_FILE_PATH = Path("config/config.yaml")
SCHEMA_FILE_PATH = Path("schema.yaml")
PARAMS_FILE_PATH = Path("params.yaml")

# Data related constants
# Include both versions of column names to handle both old and new models
NUMERICAL_FEATURES = ["Age", "Number_Of_Dependants", "Income_Lakhs"]
CATEGORICAL_FEATURES = ["Gender", "BMI_Category", "Smoking_Status", "Region",
                        "Marital_status", "Employment_Status", "Income_Level",
                        "Medical_History", "Insurance_Plan"]

# These are the column names that the preprocessor expects (from the trained model)
PREPROCESSOR_NUMERICAL_FEATURES = [
    "Age", "Number Of Dependants", "Income_Lakhs"]
PREPROCESSOR_CATEGORICAL_FEATURES = ["Gender", "BMI_Category", "Smoking_Status", "Region",
                                     "Marital_status", "Employment_Status", "Income_Level",
                                     "Medical History", "Insurance_Plan"]

TARGET_COLUMN = "Annual_Premium_Amount"

# Model related constants
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1  # Percentage of original data

# Feature engineering constants
BMI_CATEGORIES = {
    "Underweight": 18.5,
    "Normal": 24.9,
    "Overweight": 29.9,
    "Obese": float('inf')
}

AGE_CATEGORIES = {
    "Young Adult": 30,
    "Adult": 50,
    "Senior": float('inf')
}
