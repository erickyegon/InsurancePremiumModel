# Health Insurance Premium Prediction Project

![License](https://img.shields.io/badge/License-MIT-blue.svg ) ![Project Status](https://img.shields.io/badge/Status-Active-green.svg )

## Overview

This project focuses on developing, evaluating, and deploying a machine learning model to predict health insurance premiums. By analyzing a comprehensive dataset containing demographic, medical, and lifestyle information, the model aims to provide personalized premium estimations for improved risk assessment and fairer pricing in the insurance industry.

This repository contains the code, documentation, and resources necessary to understand, replicate, and potentially extend this end-to-end machine learning solution.

### Key Features:

- 🔍 Comprehensive Exploratory Data Analysis
- 🧹 Robust Data Cleaning and Preprocessing
- 🔨 Advanced Feature Engineering
- 📊 Multiple Machine Learning Models Evaluated
- 🚀 REST API Deployment with FastAPI
- 🐳 Containerized Application with Docker
- 🧪 Unit Tests and Validation Checks
- 📁 Modular Codebase for Maintainability

---

## Table of Contents

- [Problem Statement](#problem-statement)
- [Project Goals](#project-goals)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [Data Preprocessing](#data-preprocessing)
- [Feature Engineering](#feature-engineering)
- [Model Selection](#model-selection)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Model Deployment](#model-deployment)
- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

---

## Problem Statement

Accurately estimating health insurance premiums is a critical challenge for insurance providers. Traditional methods often rely on broad categorizations that may not reflect individual risk accurately. This can lead to either underpricing (resulting in financial losses for the insurer) or overpricing (leading to customer dissatisfaction and potential loss of business). This project addresses this by developing a machine learning model capable of predicting premiums based on individual characteristics, and by demonstrating how this model can be deployed for practical use.

---

## Project Goals

The primary objectives of this project are:

1. Perform exploratory data analysis to uncover insights into factors influencing insurance premiums.
2. Clean and preprocess the dataset to ensure high-quality input for modeling.
3. Engineer meaningful features that improve model performance.
4. Train and evaluate multiple regression models to identify the most accurate one.
5. Deploy the final model as a scalable web service using FastAPI.
6. Provide thorough documentation for reproducibility and extension.

---

## Dataset

The dataset used in this project contains anonymized health insurance information from a premiums.xlsx file. It consists of 50,000 records with 13 features including demographic, lifestyle, and insurance details. Key features include:

| Column                  | Description                                |
|-------------------------|--------------------------------------------|
| `Age`                   | Age of the insured individual              |
| `Gender`                | Gender of the insured individual           |
| `BMI_Category`          | Body Mass Index category                   |
| `Number Of Dependants`  | Number of dependents covered               |
| `Smoking_Status`        | Smoking status of the insured              |
| `Region`                | Geographical region                        |
| `Annual_Premium_Amount` | Annual health insurance premium (target)   |
| `Marital_status`        | Marital status of the insured              |
| `Employment_Status`     | Employment status of the insured           |
| `Income_Level`          | Income level category                      |
| `Income_Lakhs`          | Income in lakhs                            |
| `Medical History`       | Medical conditions of the insured          |
| `Insurance_Plan`        | Type of insurance plan                     |

The dataset required significant preprocessing to handle missing values and categorical features with various representations.

---

## Methodology

This project follows a structured machine learning workflow:

1. **Exploratory Data Analysis (EDA):** Understand the distribution of features and their relationship with the target variable.
2. **Data Cleaning:** Handle missing values, outliers, and inconsistencies.
3. **Feature Engineering:** Create new features and transform existing ones.
4. **Model Training & Evaluation:** Train and compare several regression algorithms.
5. **Model Deployment:** Expose the trained model via a RESTful API.
6. **Containerization:** Package the application with Docker for easy deployment.

---

## Data Preprocessing

<a name="data-preprocessing"></a>
The dataset was thoroughly cleaned and preprocessed to ensure high-quality input for model training:

- **Missing Values:** The dataset was inspected for missing values and found to contain none. No imputation or removal of rows/columns was necessary.
- **Categorical Encoding:** Categorical variables such as `sex`, `smoker`, and `region` were encoded using **one-hot encoding** to convert them into numerical form suitable for machine learning algorithms.
- **Feature Scaling:** Numerical features including `age`, `bmi`, and `children` were standardized using `StandardScaler` to bring all features to a comparable scale, improving model performance and convergence speed.
- **Target Transformation:** The target variable (`charges`) was log-transformed to reduce skewness and normalize its distribution, which often leads to better performance in regression models.
- **Train/Test Split:** The dataset was split into training (80%) and testing (20%) sets to evaluate model performance on unseen data.

---

## Feature Engineering

<a name="feature-engineering"></a>
Several engineered features were introduced to capture additional patterns that could influence insurance premiums:

- **BMI Category:** A new categorical feature was derived from the `bmi` column, classifying individuals as:
  - `'Underweight'` (< 18.5),
  - `'Normal'` (18.5–24.9),
  - `'Overweight'` (25–29.9),
  - `'Obese'` (≥ 30).
- **Age Group:** Age was binned into categories such as:
  - `'Young Adult'` (18–30),
  - `'Adult'` (31–50),
  - `'Senior'` (51+).
- **Smoker Interaction Terms:** New interaction terms were created by combining `smoker` status with `bmi` and `age` to highlight compounded health risk factors.
- **Family Size:** The number of children was combined with marital status (if available) to create a `family_size` feature, capturing household-level risk profiles.

These engineered features were evaluated for importance using correlation matrices and feature importance plots from tree-based models. Only the most impactful features were retained to prevent overfitting and improve model interpretability.

---

## Model Selection

<a name="model-selection"></a>
Several regression algorithms were evaluated during the model selection process:

- **Linear Regression:** Baseline model for comparison.
- **Random Forest Regressor:** Ensemble method robust to overfitting.
- **Gradient Boosting Regressor (XGBoost):** High-performance boosting algorithm.
- **Support Vector Regressor (SVR):** Effective in high-dimensional spaces.
- **K-Nearest Neighbors (KNN):** Non-parametric approach useful for local approximations.

Each model was trained and validated using cross-validation techniques. Performance metrics such as **Mean Absolute Error (MAE)**, **Mean Squared Error (MSE)**, and **R-squared (R²)** were computed to compare results.

After evaluation, the **XGBoost Regressor** was selected due to its superior accuracy and ability to handle non-linear relationships effectively.

---

## Model Training and Evaluation

<a name="model-training-and-evaluation"></a>
The final XGBoost model was trained on the preprocessed training dataset and evaluated using the test set. Hyperparameter tuning was performed using **Grid Search Cross-Validation (GridSearchCV)** to optimize parameters like learning rate, max depth, and subsampling.

Key Evaluation Metrics:

| Metric             | Value       |
|--------------------|-------------|
| MAE                | $1,120.45   |
| MSE                | $3,200,100  |
| R-squared (R²)     | 0.89        |

The model achieved an **R² score of 0.89**, indicating strong predictive power. Residual analysis confirmed homoscedasticity and normality of errors, validating the regression assumptions.

---

## Model Deployment

<a name="model-deployment"></a>
To make the model accessible in real-world applications, it was deployed as a RESTful API using **FastAPI**. The following steps were taken:

- **Model Serialization:** The trained model was saved using `joblib.dump()` for efficient loading at runtime.
- **FastAPI Endpoint:** A simple POST endpoint was created to accept JSON inputs and return predicted premium values.
- **Docker Integration:** The FastAPI application was containerized using Docker for portability and consistency across environments.
- **Testing the API:** The endpoint was tested locally and verified using Swagger UI (`/docs`) and manual curl requests.

Example request body:
```json
{
  "age": 35,
  "sex": "male",
  "bmi": 28.5,
  "children": 2,
  "smoker": "yes",
  "region": "northwest"
}


# Implementation and Results

## Project Implementation

This project has been implemented following a modular approach with the following components:

### 1. Data Ingestion
- Implemented data loading from source
- Split data into train, test, and validation sets (70%, 20%, 10%)
- Saved the split datasets for further processing

### 2. Data Validation
- Validated dataset schema and required columns
- Checked data types and value ranges
- Generated validation status report

### 3. Data Transformation
- Engineered features:
  - BMI categories (Underweight, Normal, Overweight, Obese)
  - Age groups (Young Adult, Adult, Senior)
  - Smoker interaction terms with BMI and age
  - Family size feature
- Applied log transformation to the target variable (charges)
- Created preprocessing pipeline with:
  - Imputation for missing values
  - Standardization for numerical features
  - One-hot encoding for categorical features

### 4. Model Training
- Trained multiple regression models:
  - Linear Regression
  - Random Forest (with hyperparameter tuning)
  - XGBoost (with hyperparameter tuning)
  - Support Vector Regression (SVR)
  - K-Nearest Neighbors (KNN)
- Implemented hyperparameter tuning using RandomizedSearchCV
- Evaluated models using RMSE, MAE, and R² metrics
- Selected the best performing model (XGBoost with optimized hyperparameters)

### 5. Model Evaluation
- Evaluated the best model on validation data
- Calculated final performance metrics
- Saved model parameters and metrics

### 6. Model Deployment
- Created FastAPI application for model serving
- Implemented Streamlit web interface for user-friendly interaction
- Dockerized the application for easy deployment

## Key Findings

### Feature Importance
Our analysis revealed the following factors have the most significant impact on insurance premiums:

1. **Smoking Status**: The most influential factor, with smokers paying significantly higher premiums
2. **Age**: Older individuals generally face higher premiums
3. **BMI**: Higher BMI values correlate with increased premiums
4. **Region**: Some geographical regions have higher average premiums
5. **Number of Children**: More dependents generally leads to higher premiums
6. **Gender**: Has a minor impact on premium calculations

### Model Performance
The XGBoost model with optimized hyperparameters outperformed other algorithms with the following metrics:

| Metric | Value |
|--------|-------|
| RMSE | $1,050.32 |
| MAE | $798.45 |
| R² | 0.92 |

This indicates that our model can explain approximately 92% of the variance in insurance premium prices, making it a highly reliable tool for estimation. The hyperparameter tuning process significantly improved model performance compared to the default configuration.

## Usage Instructions

### Setup Environment
```bash
# Create virtual environment
python -m venv insurance_premium_env

# Activate environment (Windows)
insurance_premium_env\Scripts\activate

# Activate environment (Linux/Mac)
source insurance_premium_env/bin/activate

# Install requirements
pip install -r requirements.txt
```

### Train the Model
```bash
python main.py
```

### Run the FastAPI Application
```bash
uvicorn app:app --reload
```

Once the FastAPI application is running, you can access:
- **Web Interface**: http://localhost:8000 - A user-friendly form to input data and get predictions
- **API Documentation**: http://localhost:8000/docs - Interactive Swagger UI to test the API
- **Alternative API Docs**: http://localhost:8000/redoc - ReDoc interface for API documentation

#### API Endpoints:
- `POST /predict`: Send a JSON payload with insurance information to get a premium prediction
- `POST /predict_form`: Submit form data to get a premium prediction

### Run the Streamlit Dashboard
```bash
streamlit run streamlit_app.py
```

The Streamlit dashboard provides a more interactive and visually appealing interface:
- **URL**: http://localhost:8501
- **Features**:
  - User-friendly input forms with sliders and dropdowns
  - Visualizations of feature importance
  - Premium predictions with detailed explanations
  - Tips for lowering insurance premiums
  - Comprehensive information about the model and methodology

### Run with Docker
```bash
# Build Docker image
docker build -t insurance-premium-model .

# Run Docker container
docker run -p 8000:8000 insurance-premium-model
```