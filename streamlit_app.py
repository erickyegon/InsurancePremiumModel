"""
Streamlit Application

This module contains the Streamlit application for the Insurance Premium Prediction model.
"""
from InsurancePremiumPrediction.utils import read_yaml
from InsurancePremiumPrediction.pipeline.prediction_pipeline import PredictionPipeline
from InsurancePremiumPrediction import logger
import sys
import os
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Add the current directory to the path so we can import the package
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# Set page configuration
st.set_page_config(
    page_title="Insurance Premium Prediction",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Create prediction pipeline
prediction_pipeline = PredictionPipeline()

# Load schema for validation
schema = read_yaml("schema.yaml")

# Custom CSS for a professional, modern look
st.markdown("""
<style>
    /* Main layout and colors */
    .main {
        background-color: #f8f9fa;
    }

    /* Headers */
    .main-header {
        font-size: 2.5rem;
        color: #0a58ca;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        color: #495057;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 400;
    }

    /* Prediction result styling */
    .prediction-box {
        background: linear-gradient(135deg, #d1e7dd 0%, #a3cfbb 100%);
        padding: 1.8rem;
        border-radius: 0.8rem;
        text-align: center;
        margin: 1.5rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border: 1px solid #a3cfbb;
    }
    .prediction-label {
        font-size: 1.2rem;
        color: #0f5132;
        margin-bottom: 0.5rem;
    }
    .prediction-value {
        font-size: 2.5rem;
        font-weight: 700;
        color: #0f5132;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }

    /* Feature importance section */
    .feature-importance {
        margin-top: 2rem;
        padding: 1rem;
        border-radius: 0.5rem;
        background-color: #f8f9fa;
    }

    /* Tips section */
    .tips-section {
        background-color: #e9ecef;
        padding: 1.2rem;
        border-radius: 0.5rem;
        margin-top: 1.5rem;
        border-left: 4px solid #0a58ca;
    }

    /* Footer styling */
    .footer {
        text-align: center;
        padding: 1rem;
        font-size: 0.9rem;
        color: #6c757d;
        margin-top: 2rem;
        border-top: 1px solid #dee2e6;
    }

    /* Author info */
    .author-info {
        background-color: #e7f5ff;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-top: 1rem;
        text-align: center;
        border: 1px solid #a8d7ff;
    }

    /* Improve sidebar appearance */
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
    }

    /* Make tables more professional */
    table {
        width: 100%;
        border-collapse: collapse;
    }
    th {
        background-color: #0a58ca;
        color: white;
        padding: 0.75rem;
        text-align: left;
    }
    td {
        padding: 0.75rem;
        border-bottom: 1px solid #dee2e6;
    }
    tr:nth-child(even) {
        background-color: #f2f2f2;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.markdown("<h1 class='main-header'>Insurance Premium Prediction</h1>",
            unsafe_allow_html=True)
st.markdown("<p class='sub-header'>Predict your health insurance premium based on personal information</p>",
            unsafe_allow_html=True)

# Create sidebar for inputs
st.sidebar.header("Enter Your Information")

# Age input
age = st.sidebar.slider(
    "Age",
    min_value=int(schema.columns.Age.constraints.min),
    max_value=int(schema.columns.Age.constraints.max),
    value=30,
    help="Age of the insured person"
)

# Gender input
gender = st.sidebar.selectbox(
    "Gender",
    options=schema.columns.Gender.categories,
    help="Gender of the insured person"
)

# BMI Category input
bmi_category = st.sidebar.selectbox(
    "BMI Category",
    options=schema.columns.BMI_Category.categories,
    help="Body Mass Index category"
)

# Number of Dependants input
# Use getattr to access attributes with underscores
number_of_dependants_schema = getattr(schema.columns, "Number_Of_Dependants")
dependants = st.sidebar.slider(
    "Number of Dependants",
    min_value=int(number_of_dependants_schema.constraints.min),
    max_value=int(number_of_dependants_schema.constraints.max),
    value=0,
    help="Number of dependents covered by the insurance"
)

# Smoking Status input
smoking_status = st.sidebar.selectbox(
    "Smoking Status",
    options=schema.columns.Smoking_Status.categories,
    help="Smoking status of the insured person"
)

# Region input
region = st.sidebar.selectbox(
    "Region",
    options=schema.columns.Region.categories,
    help="Region where the insured person lives"
)

# Marital Status input
marital_status = st.sidebar.selectbox(
    "Marital Status",
    options=schema.columns.Marital_status.categories,
    help="Marital status of the insured person"
)

# Employment Status input
employment_status = st.sidebar.selectbox(
    "Employment Status",
    options=schema.columns.Employment_Status.categories,
    help="Employment status of the insured person"
)

# Income Level input
income_level = st.sidebar.selectbox(
    "Income Level",
    options=schema.columns.Income_Level.categories,
    help="Income level of the insured person"
)

# Income in Lakhs input
income_lakhs = st.sidebar.slider(
    "Income (Lakhs)",
    min_value=float(schema.columns.Income_Lakhs.constraints.min),
    max_value=50.0,  # Setting a reasonable max value
    value=10.0,
    step=0.5,
    help="Income in lakhs"
)

# Medical History input
# Use getattr to access attributes with underscores
medical_history_schema = getattr(schema.columns, "Medical_History")
medical_history = st.sidebar.selectbox(
    "Medical History",
    options=medical_history_schema.categories,
    help="Medical history of the insured person"
)

# Insurance Plan input
insurance_plan = st.sidebar.selectbox(
    "Insurance Plan",
    options=schema.columns.Insurance_Plan.categories,
    help="Type of insurance plan"
)

# Create two columns for the main content
col1, col2 = st.columns([2, 1])

# Display user information in the first column
with col1:
    st.subheader("Your Information")

    # Create a DataFrame to display the user's information
    # Convert all values to strings to avoid Arrow conversion issues
    user_data = pd.DataFrame({
        "Feature": ["Age", "Gender", "BMI Category", "Number of Dependants", "Smoking Status",
                    "Region", "Marital Status", "Employment Status", "Income Level",
                    "Income (Lakhs)", "Medical History", "Insurance Plan"],
        "Value": [str(age), str(gender), str(bmi_category), str(dependants), str(smoking_status),
                  str(region.title()), str(marital_status), str(
                      employment_status), str(income_level),
                  str(income_lakhs), str(medical_history), str(insurance_plan)]
    })

    st.table(user_data)

    # Add a predict button
    predict_button = st.button(
        "Predict Premium", type="primary", use_container_width=True)

    # Make prediction when the button is clicked
    if predict_button:
        try:
            # Create input data dictionary
            input_data = {
                "Age": age,
                "Gender": gender,
                "BMI_Category": bmi_category,
                "Number_Of_Dependants": dependants,
                "Smoking_Status": smoking_status,
                "Region": region,
                "Marital_status": marital_status,
                "Employment_Status": employment_status,
                "Income_Level": income_level,
                "Income_Lakhs": income_lakhs,
                "Medical_History": medical_history,
                "Insurance_Plan": insurance_plan
            }

            # Make prediction
            prediction = prediction_pipeline.predict(input_data)

            # Display prediction with improved styling
            st.markdown("<div class='prediction-box'>", unsafe_allow_html=True)
            st.markdown(f"<p class='prediction-label'>Estimated Annual Premium:</p>",
                        unsafe_allow_html=True)
            st.markdown(
                f"<p class='prediction-value'>${prediction:,.2f}</p>", unsafe_allow_html=True)
            st.markdown("<p>Based on your provided information</p>",
                        unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            # Log the prediction
            logger.info(f"Streamlit prediction made: ${prediction:.2f}")

        except Exception as e:
            st.error(f"Error making prediction: {e}")
            logger.error(f"Error in Streamlit prediction: {e}")

# Display information about the model in the second column
with col2:
    st.subheader("Key Factors Affecting Premiums")

    # Create a simple bar chart showing the importance of different factors
    factors = ["Smoking Status", "Age", "Medical History",
               "Income", "BMI Category", "Region", "Insurance Plan"]
    # Approximate importance values
    importance = [0.35, 0.15, 0.15, 0.12, 0.10, 0.08, 0.05]

    fig, ax = plt.subplots(figsize=(8, 6))
    bars = ax.barh(factors, importance, color='skyblue')
    ax.set_xlabel('Relative Importance')
    ax.set_title('Feature Importance')

    # Add value labels to the bars
    for bar in bars:
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.0%}',
                ha='left', va='center')

    st.pyplot(fig)

    st.markdown("<div class='tips-section'>", unsafe_allow_html=True)
    st.markdown("""
    ### Tips for Lower Premiums:

    1. **Maintain a healthy lifestyle** - Non-smokers pay significantly lower premiums
    2. **Keep your BMI in the normal range** - A normal BMI category can reduce premiums
    3. **Preventive care** - Regular check-ups can help avoid health issues that may increase premiums
    4. **Choose the right insurance plan** - Different plans have different premium structures
    5. **Consider your income and employment** - These factors can affect your premium rates
    """)
    st.markdown("</div>", unsafe_allow_html=True)

# Add information about the model at the bottom
st.markdown("---")
st.markdown("""
### About the Model

This prediction model was trained on health insurance data and uses machine learning to estimate premiums based on personal factors. The model considers:

- Age of the insured person
- Gender
- BMI Category
- Number of dependents
- Smoking status
- Region of residence
- Marital status
- Employment status
- Income level
- Medical history
- Insurance plan type

The model was trained using multiple algorithms including Linear Regression, Random Forest, and XGBoost with hyperparameter tuning, with the best performing model selected for predictions.
""")

# Add footer with acknowledgment
st.markdown("---")
st.markdown("<div class='author-info'>", unsafe_allow_html=True)
st.markdown("""
### Developed By
**Erick K. Yegon, PhD**
Email: keyegon@gmail.com
Data Scientist & Machine Learning Engineer
""")
st.markdown("</div>", unsafe_allow_html=True)

st.markdown("<div class='footer'>", unsafe_allow_html=True)
st.markdown(
    "© 2025 Insurance Premium Prediction Model | Developed with Streamlit and FastAPI")
st.markdown("</div>", unsafe_allow_html=True)

# Run the app with: streamlit run streamlit_app.py
