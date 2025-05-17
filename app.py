"""
FastAPI Application

This module contains the FastAPI application for the Insurance Premium Prediction model.
"""
from InsurancePremiumPrediction.pipeline.prediction_pipeline import PredictionPipeline
from InsurancePremiumPrediction import logger
import uvicorn
import sys
import os
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field
from typing import Literal

# Add the current directory to the path so we can import the package
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# Create FastAPI app
app = FastAPI(
    title="Insurance Premium Prediction",
    description="API for predicting insurance premiums",
    version="0.1.0",
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

        # Return prediction
        return templates.TemplateResponse(
            "result.html",
            {
                "request": request,
                "prediction": f"${prediction:,.2f}",
                "input_data": input_data
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


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
