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

The dataset used in this project contains anonymized health insurance information, including features such as age, gender, BMI, number of children, smoker status, region, and the corresponding insurance premiums. The dataset was obtained from a publicly available source and consists of 1,338 rows and 7 columns. Key features include:

| Column     | Description                             |
|------------|-----------------------------------------|
| `age`      | Age of the insured individual           |
| `sex`      | Gender of the insured individual        |
| `bmi`      | Body Mass Index                         |
| `children` | Number of dependents covered            |
| `smoker`   | Whether the insured is a smoker         |
| `region`   | Geographical region                     |
| `charges`  | Annual health insurance premium (target)|

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
