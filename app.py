"""
FastAPI Application

This module contains the FastAPI application for the Insurance Premium Prediction model.
It provides endpoints for prediction, model monitoring, and retraining.
"""
from InsurancePremiumPrediction.pipeline.prediction_pipeline import PredictionPipeline
from InsurancePremiumPrediction import logger
import uvicorn
import sys
import os
import json
import random
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from fastapi import FastAPI, Request, Form, HTTPException, BackgroundTasks, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import Literal

# Add the current directory to the path so we can import the package
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Constants for model monitoring
MONITORING_DATA_PATH = "monitoring_data"
os.makedirs(MONITORING_DATA_PATH, exist_ok=True)

# Simulated model performance data


def generate_performance_data():
    """Generate simulated model performance data for monitoring."""
    # Create dates for the last 6 months
    end_date = datetime.now()
    start_date = end_date - timedelta(days=180)
    dates = pd.date_range(start=start_date, end=end_date, freq='W')

    # Generate R² values with a slight downward trend
    r2_start = 0.92
    r2_values = [max(0.85, r2_start - 0.001 * i - random.uniform(0, 0.005))
                 for i in range(len(dates))]

    # Generate MAE values with a slight upward trend
    mae_start = 150
    mae_values = [mae_start + 0.5 * i +
                  random.uniform(0, 10) for i in range(len(dates))]

    # Create dataframe
    performance_df = pd.DataFrame({
        'date': dates.strftime('%Y-%m-%d'),
        'r2_score': r2_values,
        'mae': mae_values
    })

    return performance_df

# Simulated drift detection data


def generate_drift_data():
    """Generate simulated drift detection data."""
    features = ['Age', 'BMI_Category', 'Smoking_Status',
                'Income_Lakhs', 'Medical_History']

    # Generate p-values with some features showing drift
    drift_data = {
        'feature': features,
        'p_value': [0.32, 0.04, 0.67, 0.02, 0.45],
        'test_statistic': [0.87, 2.34, 0.56, 2.78, 0.92],
        'drift_detected': [False, True, False, True, False]
    }

    return pd.DataFrame(drift_data)

# Simulated retraining history


def get_retraining_history():
    """Get the retraining history."""
    history = [
        {
            "date": "2024-10-15",
            "version": "1.0.0",
            "trigger": "Initial Deployment",
            "performance_improvement": "-",
            "notes": "Initial model deployment"
        },
        {
            "date": "2025-01-05",
            "version": "2.0.0",
            "trigger": "Significant Data Drift",
            "performance_improvement": "+5.2% R²",
            "notes": "Retrained with 3 months of additional data to address drift in age and income distributions"
        }
    ]

    # Check if there's a pending retraining
    retraining_status_path = os.path.join(
        MONITORING_DATA_PATH, "retraining_status.json")
    if os.path.exists(retraining_status_path):
        with open(retraining_status_path, 'r') as f:
            status = json.load(f)
            if status.get("status") == "pending":
                history.append({
                    "date": status.get("timestamp"),
                    "version": "3.0.0 (Pending)",
                    "trigger": "Manual Trigger",
                    "performance_improvement": "Pending",
                    "notes": "Retraining in progress - triggered manually by user"
                })

    return history


# Simulate model retraining process
# Data ingestion for retraining
def ingest_data_for_retraining(data_source_type: str, file_path: str = None, file_upload=None, data_url: str = None):
    """
    Ingest data for model retraining from various sources.

    Args:
        data_source_type: Type of data source ('file_path', 'file_upload', 'url', or 'default')
        file_path: Path to the data file (if data_source_type is 'file_path')
        file_upload: Uploaded file object (if data_source_type is 'file_upload')
        data_url: URL to the data file (if data_source_type is 'url')

    Returns:
        DataFrame: The ingested data
        str: Path to the saved data file
    """
    logger.info(
        f"Ingesting data for retraining from source type: {data_source_type}")

    # Create directory for retraining data if it doesn't exist
    retraining_data_dir = os.path.join(MONITORING_DATA_PATH, "retraining_data")
    os.makedirs(retraining_data_dir, exist_ok=True)

    # Generate a unique filename for the data
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    data_filename = f"retraining_data_{timestamp}.csv"
    data_file_path = os.path.join(retraining_data_dir, data_filename)

    try:
        # Load data based on source type
        if data_source_type == "default":
            # Use default training data
            logger.info("Using default training data")
            # In a real implementation, this would load the original training data
            # For this demo, we'll use a sample dataset
            df = pd.DataFrame({
                "Age": np.random.randint(18, 80, 1000),
                "Gender": np.random.choice(["Male", "Female"], 1000),
                "BMI_Category": np.random.choice(["Underweight", "Normal", "Overweight", "Obese"], 1000),
                "Number_Of_Dependants": np.random.randint(0, 5, 1000),
                "Smoking_Status": np.random.choice(["Smoker", "Non-Smoker"], 1000),
                "Region": np.random.choice(["Northeast", "Northwest", "Southeast", "Southwest"], 1000),
                "Annual_Premium_Amount": np.random.uniform(5000, 50000, 1000)
            })

        elif data_source_type == "file_path":
            # Load data from file path
            logger.info(f"Loading data from file path: {file_path}")
            if file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif file_path.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")

        elif data_source_type == "file_upload":
            # Load data from uploaded file
            logger.info("Loading data from uploaded file")
            # Save the uploaded file
            temp_file_path = os.path.join(
                retraining_data_dir, f"temp_{data_filename}")
            with open(temp_file_path, "wb") as temp_file:
                temp_file.write(file_upload.file.read())

            # Read the file based on its extension
            file_extension = os.path.splitext(file_upload.filename)[1].lower()
            if file_extension == '.csv':
                df = pd.read_csv(temp_file_path)
            elif file_extension in ('.xls', '.xlsx'):
                df = pd.read_excel(temp_file_path)
            else:
                raise ValueError(
                    f"Unsupported file format: {file_upload.filename}")

            # Remove the temporary file
            os.remove(temp_file_path)

        elif data_source_type == "url":
            # Load data from URL
            logger.info(f"Loading data from URL: {data_url}")
            # Determine file type from URL
            if data_url.endswith('.csv'):
                df = pd.read_csv(data_url)
            elif data_url.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(data_url)
            else:
                raise ValueError(f"Unsupported file format in URL: {data_url}")
        else:
            raise ValueError(
                f"Unsupported data source type: {data_source_type}")

        # Save the ingested data to a standard format
        df.to_csv(data_file_path, index=False)
        logger.info(
            f"Data ingestion successful. Data saved to {data_file_path}")

        return df, data_file_path

    except Exception as e:
        logger.error(f"Error during data ingestion: {str(e)}")
        raise

# Data validation for retraining


def validate_data_for_retraining(df):
    """
    Validate the ingested data for model retraining.

    Args:
        df: DataFrame containing the ingested data

    Returns:
        bool: True if validation passed, False otherwise
        dict: Validation results with details
    """
    logger.info("Validating data for retraining")

    validation_results = {
        "passed": True,
        "errors": [],
        "warnings": []
    }

    try:
        # Check if required columns are present
        required_columns = ["Age", "Gender", "BMI_Category", "Number_Of_Dependants",
                            "Smoking_Status", "Region", "Annual_Premium_Amount"]

        missing_columns = [
            col for col in required_columns if col not in df.columns]
        if missing_columns:
            validation_results["passed"] = False
            validation_results["errors"].append(
                f"Missing required columns: {', '.join(missing_columns)}")

        # Check data types
        if "Age" in df.columns and not pd.api.types.is_numeric_dtype(df["Age"]):
            validation_results["errors"].append("Age column must be numeric")
            validation_results["passed"] = False

        if "Annual_Premium_Amount" in df.columns and not pd.api.types.is_numeric_dtype(df["Annual_Premium_Amount"]):
            validation_results["errors"].append(
                "Annual_Premium_Amount column must be numeric")
            validation_results["passed"] = False

        # Check for missing values
        missing_values = df.isnull().sum()
        if missing_values.sum() > 0:
            validation_results["warnings"].append(
                f"Dataset contains missing values: {missing_values[missing_values > 0].to_dict()}")

        # Check data ranges
        if "Age" in df.columns:
            if df["Age"].min() < 18 or df["Age"].max() > 100:
                validation_results["warnings"].append(
                    f"Age values outside expected range (18-100): min={df['Age'].min()}, max={df['Age'].max()}")

        # Check categorical values
        if "Gender" in df.columns:
            valid_genders = ["Male", "Female"]
            invalid_genders = df[~df["Gender"].isin(
                valid_genders)]["Gender"].unique()
            if len(invalid_genders) > 0:
                validation_results["warnings"].append(
                    f"Invalid Gender values: {', '.join(map(str, invalid_genders))}")

        # Check dataset size
        if len(df) < 100:
            validation_results["warnings"].append(
                f"Dataset is small ({len(df)} rows). Model performance may be affected.")

        logger.info(
            f"Data validation completed. Passed: {validation_results['passed']}")
        if validation_results["errors"]:
            logger.error(f"Validation errors: {validation_results['errors']}")
        if validation_results["warnings"]:
            logger.warning(
                f"Validation warnings: {validation_results['warnings']}")

        return validation_results["passed"], validation_results

    except Exception as e:
        logger.error(f"Error during data validation: {str(e)}")
        validation_results["passed"] = False
        validation_results["errors"].append(f"Validation error: {str(e)}")
        return False, validation_results


def retrain_model_task(reason: str, requested_by: str, data_source_type: str = "default",
                       file_path: str = None, file_upload=None, data_url: str = None,
                       validate_data: bool = True):
    """
    Background task to simulate model retraining with data ingestion.

    Args:
        reason: Reason for retraining
        requested_by: Name of the person requesting retraining
        data_source_type: Type of data source ('file_path', 'file_upload', 'url', or 'default')
        file_path: Path to the data file (if data_source_type is 'file_path')
        file_upload: Uploaded file object (if data_source_type is 'file_upload')
        data_url: URL to the data file (if data_source_type is 'url')
        validate_data: Whether to validate the data before retraining
    """
    logger.info(
        f"Starting model retraining requested by {requested_by}. Reason: {reason}")

    # Record initial retraining status
    status = {
        "status": "pending",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "reason": reason,
        "requested_by": requested_by,
        "data_source_type": data_source_type,
        "progress": "Initializing retraining process",
        "validation_results": None,
        "completion_time": None
    }

    # Save initial status to file
    status_file_path = os.path.join(
        MONITORING_DATA_PATH, "retraining_status.json")
    with open(status_file_path, 'w') as f:
        json.dump(status, f)

    try:
        # Step 1: Data Ingestion
        status["progress"] = "Ingesting data"
        with open(status_file_path, 'w') as f:
            json.dump(status, f)

        df, data_file_path = ingest_data_for_retraining(
            data_source_type=data_source_type,
            file_path=file_path,
            file_upload=file_upload,
            data_url=data_url
        )

        # Step 2: Data Validation (if enabled)
        if validate_data:
            status["progress"] = "Validating data"
            with open(status_file_path, 'w') as f:
                json.dump(status, f)

            validation_passed, validation_results = validate_data_for_retraining(
                df)
            status["validation_results"] = validation_results

            if not validation_passed:
                status["status"] = "failed"
                status["progress"] = "Data validation failed"
                with open(status_file_path, 'w') as f:
                    json.dump(status, f)
                logger.error("Model retraining failed: Data validation failed")
                return

        # Step 3: Data Preprocessing
        status["progress"] = "Preprocessing data"
        with open(status_file_path, 'w') as f:
            json.dump(status, f)

        # In a real implementation, this would preprocess the data
        # For this demo, we'll just simulate preprocessing
        logger.info("Preprocessing data for retraining")
        time.sleep(2)  # Simulate preprocessing time

        # Step 4: Model Training
        status["progress"] = "Training model"
        with open(status_file_path, 'w') as f:
            json.dump(status, f)

        # In a real implementation, this would train the model
        # For this demo, we'll just simulate training
        logger.info("Training model with new data")
        time.sleep(3)  # Simulate training time

        # Step 5: Model Evaluation
        status["progress"] = "Evaluating model"
        with open(status_file_path, 'w') as f:
            json.dump(status, f)

        # In a real implementation, this would evaluate the model
        # For this demo, we'll just simulate evaluation
        logger.info("Evaluating retrained model")
        time.sleep(2)  # Simulate evaluation time

        # Step 6: Model Registration
        status["progress"] = "Registering model"
        with open(status_file_path, 'w') as f:
            json.dump(status, f)

        # In a real implementation, this would register the model
        # For this demo, we'll just simulate registration
        logger.info("Registering retrained model")
        time.sleep(1)  # Simulate registration time

        # Update status to completed
        status["status"] = "completed"
        status["progress"] = "Retraining completed successfully"
        status["completion_time"] = datetime.now().strftime(
            "%Y-%m-%d %H:%M:%S")

        with open(status_file_path, 'w') as f:
            json.dump(status, f)

        logger.info("Model retraining completed successfully")

    except Exception as e:
        logger.error(f"Error during model retraining: {str(e)}")
        status["status"] = "failed"
        status["progress"] = f"Retraining failed: {str(e)}"

        with open(status_file_path, 'w') as f:
            json.dump(status, f)

        # In a real implementation, this would send a notification about the failure
        # For this demo, we'll just log the error
        logger.error("Model retraining task failed")


# Create FastAPI app
app = FastAPI(
    title="Insurance Premium Prediction",
    description="API for predicting insurance premiums",
    version="2.0.0",
)

# Mount static files directory
app.mount("/static", StaticFiles(directory="templates/static"), name="static")

# Set up templates
templates = Jinja2Templates(directory="templates")

# Create prediction pipeline
prediction_pipeline = PredictionPipeline()


# Input validation model
class InsuranceInput(BaseModel):
    """
    Input model for insurance premium prediction.
    """
    Age: int = Field(..., ge=18, le=120,
                     description="Age of the insured person")
    Gender: Literal["Male",
                    "Female"] = Field(..., description="Gender of the insured person")
    BMI_Category: Literal["Underweight", "Normal", "Overweight", "Obese", "Obesity"] = Field(
        ..., description="Body mass index category of the insured person")
    Number_Of_Dependants: int = Field(..., ge=-5, le=10,
                                      description="Number of dependents covered by the insurance")
    Smoking_Status: Literal["Smoker", "Non-Smoker", "Regular", "Occasional",
                            "Does Not Smoke", "No Smoking", "Not Smoking", "Smoking=0"] = Field(
        ..., description="Smoking status of the insured person")
    Region: Literal["Northeast", "Northwest", "Southeast", "Southwest"] = Field(
        ..., description="Region where the insured person lives")
    Marital_status: Literal["Married", "Unmarried"] = Field(
        ..., description="Marital status of the insured person")
    Employment_Status: Literal["Employed", "Unemployed", "Self-employed", "Self-Employed",
                               "Salaried", "Freelancer"] = Field(
        ..., description="Employment status of the insured person")
    Income_Level: Literal["Low", "Medium", "High", "<10L", "10L - 25L", "25L - 40L", "> 40L"] = Field(
        ..., description="Income level of the insured person")
    Income_Lakhs: float = Field(..., ge=0.0,
                                description="Income in lakhs")
    Medical_History: Literal["None", "No Disease", "Diabetes", "Heart Disease", "Heart disease",
                             "Hypertension", "High blood pressure", "Asthma", "Thyroid",
                             "Diabetes & Thyroid", "Diabetes & Heart disease",
                             "Diabetes & High blood pressure", "High blood pressure & Heart disease"] = Field(
        ..., description="Medical history of the insured person")
    Insurance_Plan: Literal["Bronze", "Silver", "Gold", "Platinum"] = Field(
        ..., description="Type of insurance plan")


# Output model
class InsuranceOutput(BaseModel):
    """
    Output model for insurance premium prediction.
    """
    predicted_premium: float = Field(...,
                                     description="Predicted insurance premium")


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """
    Home page.
    """
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/monitoring", response_class=HTMLResponse)
async def monitoring_dashboard(request: Request):
    """
    Model monitoring dashboard.
    """
    # Get monitoring data
    performance_data = generate_performance_data()
    drift_data = generate_drift_data()
    retraining_history = get_retraining_history()

    # Calculate monitoring metrics
    latest_r2 = performance_data['r2_score'].iloc[-1]
    latest_mae = performance_data['mae'].iloc[-1]
    drift_detected = any(drift_data['drift_detected'])
    features_with_drift = drift_data[drift_data['drift_detected']]['feature'].tolist(
    )
    retraining_recommended = drift_detected or latest_r2 < 0.9

    return templates.TemplateResponse(
        "monitoring.html",
        {
            "request": request,
            "performance_data": performance_data.to_dict(orient='records'),
            "drift_data": drift_data.to_dict(orient='records'),
            "retraining_history": retraining_history,
            "latest_r2": latest_r2,
            "latest_mae": latest_mae,
            "drift_detected": drift_detected,
            "features_with_drift": features_with_drift,
            "retraining_recommended": retraining_recommended
        }
    )


@app.get("/instructions", response_class=HTMLResponse)
async def instructions(request: Request):
    """
    Instructions page.
    """
    return templates.TemplateResponse("instructions.html", {"request": request})


@app.post("/predict", response_model=InsuranceOutput)
async def predict(insurance_input: InsuranceInput):
    """
    Predict insurance premium.
    """
    try:
        # Convert input model to dictionary
        input_data = insurance_input.model_dump()

        # Make prediction
        prediction = prediction_pipeline.predict(input_data)

        # Return prediction
        return InsuranceOutput(predicted_premium=prediction)

    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict_form", response_class=HTMLResponse)
async def predict_form(
    request: Request,
    Age: int = Form(...),
    Gender: str = Form(...),
    BMI_Category: str = Form(...),
    Number_Of_Dependants: int = Form(...),
    Smoking_Status: str = Form(...),
    Region: str = Form(...),
    Marital_status: str = Form(...),
    Employment_Status: str = Form(...),
    Income_Level: str = Form(...),
    Income_Lakhs: float = Form(...),
    Medical_History: str = Form(...),
    Insurance_Plan: str = Form(...)
):
    """
    Predict insurance premium from form data.
    """
    try:
        # Validate input data
        insurance_input = InsuranceInput(
            Age=Age,
            Gender=Gender,
            BMI_Category=BMI_Category,
            Number_Of_Dependants=Number_Of_Dependants,
            Smoking_Status=Smoking_Status,
            Region=Region,
            Marital_status=Marital_status,
            Employment_Status=Employment_Status,
            Income_Level=Income_Level,
            Income_Lakhs=Income_Lakhs,
            Medical_History=Medical_History,
            Insurance_Plan=Insurance_Plan
        )

        # Convert input model to dictionary
        input_data = insurance_input.model_dump()

        # Make prediction
        prediction = prediction_pipeline.predict(input_data)

        # Calculate confidence interval (simulated)
        lower_bound = prediction * 0.92
        upper_bound = prediction * 1.08

        # Generate feature importance (simulated)
        feature_importance = {
            "smoking_status": 0.35,
            "age": 0.25,
            "bmi_category": 0.15,
            "medical_history": 0.12,
            "region": 0.08,
            "income_lakhs": 0.05
        }

        # Return prediction
        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "prediction": f"${prediction:,.2f}",
                "input_data": input_data,
                "lower_bound": f"${lower_bound:,.2f}",
                "upper_bound": f"${upper_bound:,.2f}",
                "feature_importance": feature_importance
            }
        )

    except Exception as e:
        logger.error(f"Error making prediction: {e}")
        return templates.TemplateResponse(
            "error.html",
            {
                "request": request,
                "error": str(e)
            }
        )


@app.post("/trigger-retraining", response_class=HTMLResponse)
async def trigger_retraining_form(
    request: Request,
    background_tasks: BackgroundTasks,
    reason: str = Form(...),
    requested_by: str = Form(...),
    email: str = Form(...),
    data_source_type: str = Form(...),
    file_path: str = Form(None),
    file_upload: UploadFile = File(None),
    data_url: str = Form(None),
    validate_data: bool = Form(True)
):
    """
    Process the retraining form submission with data source selection.
    """
    try:
        logger.info(
            f"Received retraining request from {requested_by} with data source type: {data_source_type}")

        # Add retraining task to background tasks with data source information
        background_tasks.add_task(
            retrain_model_task,
            reason=reason,
            requested_by=requested_by,
            data_source_type=data_source_type,
            file_path=file_path,
            file_upload=file_upload,
            data_url=data_url,
            validate_data=validate_data
        )

        # Get monitoring data for the response
        performance_data = generate_performance_data()
        drift_data = generate_drift_data()
        retraining_history = get_retraining_history()

        # Calculate monitoring metrics
        latest_r2 = performance_data['r2_score'].iloc[-1]
        latest_mae = performance_data['mae'].iloc[-1]
        drift_detected = any(drift_data['drift_detected'])
        features_with_drift = drift_data[drift_data['drift_detected']]['feature'].tolist(
        )
        retraining_recommended = drift_detected or latest_r2 < 0.9

        # Prepare success message based on data source
        data_source_info = ""
        if data_source_type == "file_upload" and file_upload:
            data_source_info = f" using uploaded file '{file_upload.filename}'"
        elif data_source_type == "file_path" and file_path:
            data_source_info = f" using file path '{file_path}'"
        elif data_source_type == "url" and data_url:
            data_source_info = f" using data from URL '{data_url}'"
        elif data_source_type == "default":
            data_source_info = " using default training data"

        # Return to monitoring page with success message
        return templates.TemplateResponse(
            "monitoring.html",
            {
                "request": request,
                "performance_data": performance_data.to_dict(orient='records'),
                "drift_data": drift_data.to_dict(orient='records'),
                "retraining_history": retraining_history,
                "latest_r2": latest_r2,
                "latest_mae": latest_mae,
                "drift_detected": drift_detected,
                "features_with_drift": features_with_drift,
                "retraining_recommended": retraining_recommended,
                "retraining_success": True,
                "retraining_message": f"Model retraining has been initiated{data_source_info}. You will be notified at {email} when the process is complete."
            }
        )
    except Exception as e:
        logger.error(f"Error during retraining request: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/monitoring/metrics")
async def get_monitoring_metrics():
    """
    API endpoint for getting model monitoring metrics.
    """
    try:
        # Get monitoring data
        performance_data = generate_performance_data()
        drift_data = generate_drift_data()

        # Calculate monitoring metrics
        latest_r2 = performance_data['r2_score'].iloc[-1]
        latest_mae = performance_data['mae'].iloc[-1]
        drift_detected = any(drift_data['drift_detected'])
        features_with_drift = drift_data[drift_data['drift_detected']]['feature'].tolist(
        )
        retraining_recommended = drift_detected or latest_r2 < 0.9

        # Return metrics as JSON
        return {
            "r2_score": round(latest_r2, 4),
            "mae": round(latest_mae, 2),
            "drift_detected": drift_detected,
            "features_with_drift": features_with_drift,
            "retraining_recommended": retraining_recommended,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
    except Exception as e:
        logger.error(f"Error getting monitoring metrics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
