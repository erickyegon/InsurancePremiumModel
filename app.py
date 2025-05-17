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
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from fastapi import FastAPI, Request, Form, HTTPException, BackgroundTasks
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
from typing import Literal, Dict, List, Optional, Union

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
def retrain_model_task(reason: str, requested_by: str):
    """Background task to simulate model retraining."""
    logger.info(
        f"Starting model retraining requested by {requested_by}. Reason: {reason}")

    # Record retraining status
    status = {
        "status": "pending",
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "reason": reason,
        "requested_by": requested_by
    }

    # Save status to file
    with open(os.path.join(MONITORING_DATA_PATH, "retraining_status.json"), 'w') as f:
        json.dump(status, f)

    # In a real implementation, this would trigger an ML pipeline job
    # For this demo, we'll just log the request
    logger.info("Model retraining task completed")


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
    email: str = Form(...)
):
    """
    Process the retraining form submission.
    """
    try:
        logger.info(f"Received retraining request from {requested_by}")

        # Add retraining task to background tasks
        background_tasks.add_task(
            retrain_model_task,
            reason,
            requested_by
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
                "retraining_message": f"Model retraining has been initiated. You will be notified at {email} when the process is complete."
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
