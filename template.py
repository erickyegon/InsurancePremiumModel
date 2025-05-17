import os
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

# Project name (can be customized)
project_name = "InsurancePremiumPrediction"

# List of files and directories to create
list_of_files = [
    # Source root package
    f"src/{project_name}/__init__.py",

    # Components (pipeline stages)
    f"src/{project_name}/components/__init__.py",
    f"src/{project_name}/components/data_ingestion.py",
    f"src/{project_name}/components/data_validation.py",
    f"src/{project_name}/components/data_transformation.py",
    f"src/{project_name}/components/model_trainer.py",
    f"src/{project_name}/components/model_evaluation.py",
    f"src/{project_name}/components/model_deployment.py",

    # Utilities
    f"src/{project_name}/utils/__init__.py",
    f"src/{project_name}/utils/common.py",

    # Configuration
    f"src/{project_name}/config/__init__.py",
    f"src/{project_name}/config/configuration.py",

    # Pipeline execution scripts
    f"src/{project_name}/pipeline/__init__.py",
    f"src/{project_name}/pipeline/training_pipeline.py",
    f"src/{project_name}/pipeline/prediction_pipeline.py",

    # Data entities and schema definitions
    f"src/{project_name}/entity/__init__.py",
    f"src/{project_name}/entity/config_entity.py",

    # Constants module
    f"src/{project_name}/constants/__init__.py",

    # Configuration and schema files
    "config/config.yaml",
    "params.yaml",
    "schema.yaml",

    # Main application entry point
    "main.py",

    # Web API endpoint (if deployed as service)
    "app.py",

    # Requirements & setup
    "requirements.txt",
    "setup.py",

    # Jupyter notebooks for EDA and modeling
    "research/__init__.py",
    "research/exploratory_data_analysis.ipynb",              # Initial data exploration
    # Data cleaning + model building
    "research/data_preprocessing_and_modeling.ipynb",
    # Advanced feature creation
    "research/feature_engineering_experiments.ipynb",
    "research/model_comparison_and_selection.ipynb",           # Model benchmarking
    "research/final_model_analysis.ipynb",                    # Post-training insights

    # HTML templates (for web apps or reports)
    "templates/index.html",

    # Unit tests
    "test.py",
    "tests/__init__.py",
    "tests/test_data_preprocessing.py",
    "tests/test_model.py",

    # Docker support
    "Dockerfile",
    ".dockerignore",

    # CI/CD and deployment configs (optional)
    ".github/workflows/main.yaml",
    "deployment/app_service.yaml",
]

# Create files and directories
for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"✅ Created directory: {filedir} (for file: {filename})")

    if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
        with open(filepath, "w") as f:
            pass
        logging.info(f"📄 Created empty file: {filepath}")
    else:
        logging.info(f"⚠️ File already exists: {filename}")
