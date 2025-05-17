"""
Project Structure Generator for Insurance Premium Model

This script creates a standardized project structure for the Insurance Premium Model project.
It generates directories and files according to a predefined structure, and adds template
content to key files like README.md and requirements.txt.

Usage:
    python template.py

The script will create all necessary directories and files for the project.
Existing files will not be overwritten.
"""

import os
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s]: %(message)s:')

# Project name (can be customized)
PROJECT_NAME = "InsurancePremiumModel"

# List of files and directories to create
list_of_files = [
    # Source root package
    f"{PROJECT_NAME}/__init__.py",

    # Components (pipeline stages)
    f"{PROJECT_NAME}/components/__init__.py",
    f"{PROJECT_NAME}/components/data_ingestion.py",
    f"{PROJECT_NAME}/components/data_validation.py",
    f"{PROJECT_NAME}/components/data_transformation.py",
    f"{PROJECT_NAME}/components/model_trainer.py",
    f"{PROJECT_NAME}/components/model_evaluation.py",
    f"{PROJECT_NAME}/components/model_deployment.py",
    f"{PROJECT_NAME}/components/model_monitoring.py",
    f"{PROJECT_NAME}/components/model_retraining.py",

    # Utilities
    f"{PROJECT_NAME}/utils/__init__.py",
    f"{PROJECT_NAME}/utils/common.py",
    f"{PROJECT_NAME}/utils/data_utils.py",
    f"{PROJECT_NAME}/utils/visualization.py",
    f"{PROJECT_NAME}/utils/monitoring_utils.py",

    # Configuration
    f"{PROJECT_NAME}/config/__init__.py",
    f"{PROJECT_NAME}/config/configuration.py",

    # Pipeline execution scripts
    f"{PROJECT_NAME}/pipeline/__init__.py",
    f"{PROJECT_NAME}/pipeline/training_pipeline.py",
    f"{PROJECT_NAME}/pipeline/prediction_pipeline.py",
    f"{PROJECT_NAME}/pipeline/monitoring_pipeline.py",
    f"{PROJECT_NAME}/pipeline/retraining_pipeline.py",

    # Data entities and schema definitions
    f"{PROJECT_NAME}/entity/__init__.py",
    f"{PROJECT_NAME}/entity/config_entity.py",
    f"{PROJECT_NAME}/entity/monitoring_entity.py",
    f"{PROJECT_NAME}/entity/retraining_entity.py",

    # Constants module
    f"{PROJECT_NAME}/constants/__init__.py",

    # Configuration and schema files
    "config/config.yaml",
    "config/monitoring_config.yaml",
    "config/retraining_config.yaml",
    "params.yaml",
    "schema.yaml",

    # Main application entry point
    "main.py",

    # Web API endpoint (FastAPI)
    "app.py",

    # Streamlit application
    "streamlit_app.py",

    # Requirements & setup
    "requirements.txt",
    "setup.py",

    # Jupyter notebooks for EDA and modeling
    "research/__init__.py",
    "research/exploratory_data_analysis.ipynb",              # Initial data exploration
    # Data cleaning + model building
    "research/data_preprocessing_and_modeling.ipynb",
    "research/feature_engineering_experiments.ipynb",        # Advanced feature creation
    "research/model_comparison_and_selection.ipynb",         # Model benchmarking
    "research/final_model_analysis.ipynb",                   # Post-training insights
    # Drift detection and monitoring
    "research/model_monitoring_analysis.ipynb",
    # Retraining criteria and process
    "research/retraining_strategy.ipynb",

    # HTML templates (for FastAPI web app)
    "templates/index.html",
    "templates/result.html",
    "templates/monitoring.html",
    "templates/instructions.html",
    "templates/static/css/style.css",
    "templates/static/js/main.js",
    "templates/static/images/logo.png",

    # Data directory
    "data/raw/premiums.xls",
    "data/processed/.gitkeep",
    "data/interim/.gitkeep",
    "data/external/.gitkeep",

    # Model directory
    "models/.gitkeep",
    "models/model_registry.json",

    # Monitoring data directory
    "monitoring_data/.gitkeep",
    "monitoring_data/drift_reports/.gitkeep",
    "monitoring_data/performance_metrics/.gitkeep",
    "monitoring_data/retraining_history/.gitkeep",

    # Unit tests
    "tests/__init__.py",
    "tests/test_data_preprocessing.py",
    "tests/test_model.py",
    "tests/test_monitoring.py",
    "tests/test_retraining.py",
    "tests/test_api.py",
    "tests/test_streamlit.py",

    # Docker support
    "Dockerfile",
    ".dockerignore",
    "docker-compose.yml",

    # CI/CD and deployment configs
    ".github/workflows/ci.yaml",
    ".github/workflows/cd.yaml",
    "deployment/app_service.yaml",
    "deployment/monitoring_service.yaml",
    "deployment/model_registry_service.yaml",

    # Azure deployment
    "azure/deploy.sh",
    "azure/app_service_template.json",
    "azure/container_registry_template.json",
    "azure/monitoring_service_template.json",

    # Documentation
    "docs/index.md",
    "docs/user_guide.md",
    "docs/api_reference.md",
    "docs/monitoring_guide.md",
    "docs/retraining_guide.md",
    "docs/deployment_guide.md",

    # README and other project files
    "README.md",
    ".gitignore",
    "LICENSE",
    "CONTRIBUTING.md"
]

# Function to create template content for key files


def create_template_content(filepath):
    """Create template content for key files in the project structure."""

    # Define template content for important files
    templates = {
        f"{PROJECT_NAME}/__init__.py": f"""
import os
import sys
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(lineno)d %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/app.log"),
        logging.StreamHandler(sys.stdout)
    ]
)

# Create logger
logger = logging.getLogger("{PROJECT_NAME}")
""",
        "README.md": f"""
# {PROJECT_NAME}

## Overview
This project implements a machine learning model to predict insurance premiums based on customer information. It includes a complete MLOps pipeline with model monitoring, drift detection, and retraining capabilities.

## Features
- Premium prediction based on customer attributes
- Interactive web interface with FastAPI and Streamlit
- Comprehensive model monitoring dashboard
- Data drift detection and visualization
- Flexible model retraining with multiple data source options
- Automated deployment pipeline for Azure

## Project Structure
- `{PROJECT_NAME}/`: Core ML package
- `app.py`: FastAPI web application
- `streamlit_app.py`: Streamlit dashboard
- `research/`: Jupyter notebooks for analysis
- `models/`: Model storage and registry
- `monitoring_data/`: Monitoring metrics and drift reports
- `data/`: Training and validation datasets
- `templates/`: HTML templates for web interface
- `tests/`: Unit and integration tests
- `docs/`: Project documentation

## Getting Started
1. Install dependencies: `pip install -r requirements.txt`
2. Run the FastAPI app: `uvicorn app:app --reload`
3. Run the Streamlit dashboard: `streamlit run streamlit_app.py`

## MLOps Features
- Model monitoring with drift detection
- Flexible retraining with custom data sources
- Model versioning and registry
- Automated deployment pipeline

## Acknowledgments
Developed by Erick K. Yegon, PhD | keyegon@gmail.com
""",
        "requirements.txt": """
# Core ML libraries
numpy>=1.20.0
pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
plotly>=5.3.0

# Web frameworks
fastapi>=0.68.0
streamlit>=1.10.0
uvicorn>=0.15.0
jinja2>=3.0.0
python-multipart>=0.0.5

# Data handling
openpyxl>=3.0.0
xlrd>=2.0.0
pyyaml>=6.0.0

# MLOps
mlflow>=1.20.0
evidently>=0.1.50
scipy>=1.7.0

# Testing
pytest>=6.2.5
pytest-cov>=2.12.0

# Deployment
python-dotenv>=0.19.0
azure-storage-blob>=12.9.0
azure-identity>=1.7.0
"""
    }

    # Return template content if available, otherwise empty string
    return templates.get(str(filepath), "")

# Create project structure


def create_project_structure():
    """Create the project structure with directories and files."""

    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    logging.info(f"✅ Created directory: logs")

    # Process each file in the list
    for filepath in list_of_files:
        filepath = Path(filepath)
        filedir, _ = os.path.split(filepath)

        # Create directory if needed
        if filedir != "":
            os.makedirs(filedir, exist_ok=True)
            logging.info(f"✅ Created directory: {filedir}")

        # Create file if it doesn't exist or is empty
        if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
            # Get template content for this file
            content = create_template_content(filepath)

            # Write content to file
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(content)

            file_status = "📄 Created file" if content else "📄 Created empty file"
            logging.info(f"{file_status}: {filepath}")
        else:
            logging.info(f"⚠️ File already exists: {filepath}")

    # Create .gitkeep files for empty directories
    for dirpath in [
        "data/processed",
        "data/interim",
        "data/external",
        "models",
        "monitoring_data",
        "monitoring_data/drift_reports",
        "monitoring_data/performance_metrics",
        "monitoring_data/retraining_history"
    ]:
        gitkeep_file = os.path.join(dirpath, ".gitkeep")
        if not os.path.exists(gitkeep_file):
            os.makedirs(dirpath, exist_ok=True)
            with open(gitkeep_file, "w", encoding="utf-8") as f:
                pass
            logging.info(f"📄 Created .gitkeep file in {dirpath}")


# Execute the project structure creation
if __name__ == "__main__":
    logging.info(f"Creating project structure for {PROJECT_NAME}...")
    create_project_structure()
    logging.info("✅ Project structure created successfully!")
