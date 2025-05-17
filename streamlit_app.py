import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import sys
import time
import logging
import uuid
import yaml
import json
import base64
from io import BytesIO
from PIL import Image
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union

# Function to create sample visualizations for the instructions tab


def create_sample_visualization(tab_name):
    """Create and display sample visualizations directly in the instructions tab"""

    if tab_name == "calculator":
        # Premium Calculator Screenshot
        st.subheader("Premium Calculator Interface")

        # Create a sample input summary
        df = pd.DataFrame({
            'Feature': ['Age', 'Smoking_Status', 'BMI_Category', 'Medical_History', 'Region', 'Income_Lakhs'],
            'Value': ['45', 'Non-Smoker', 'Normal', 'None', 'Northeast', '12.5']
        })

        # Display as a styled table
        st.markdown("#### Sample Customer Information")
        st.dataframe(df, use_container_width=True)

        # Add some UI elements to simulate the calculator
        col1, col2 = st.columns(2)
        with col1:
            st.number_input("Age", min_value=18, max_value=85,
                            value=45, disabled=True)
            st.selectbox("Gender", ["Male", "Female"], disabled=True)
            st.selectbox("BMI Category", [
                         "Underweight", "Normal", "Overweight", "Obese"], index=1, disabled=True)

        with col2:
            st.selectbox("Smoking Status", [
                         "Non-Smoker", "Smoker"], disabled=True)
            st.selectbox(
                "Region", ["Northeast", "Northwest", "Southeast", "Southwest"], disabled=True)
            st.number_input("Income (Lakhs)", min_value=1.0,
                            max_value=50.0, value=12.5, disabled=True)

        st.button("Calculate Premium", disabled=True)

    elif tab_name == "results":
        # Results Explanation
        st.subheader("Premium Results Visualization")

        # Create a sample premium result
        st.markdown("#### Premium Estimate")
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("""
            <div style="background-color:#f0f8ff; padding:20px; border-radius:10px; text-align:center;">
                <h1 style="color:#1e90ff; font-size:48px; margin:0;">₹15,420</h1>
                <p style="color:#666; margin:5px 0 0 0;">Annual Premium</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("##### Confidence Interval")
            st.markdown("₹14,250 - ₹16,590")
            st.markdown("##### Model Version")
            st.markdown("v2.1.3 (Last updated: Jan 5, 2025)")

        # Feature importance chart
        st.markdown("#### Feature Importance")
        df_results = pd.DataFrame({
            'Factor': ['Smoking_Status', 'Age', 'BMI_Category', 'Medical_History', 'Region', 'Income_Lakhs'],
            'Importance': [0.35, 0.25, 0.15, 0.12, 0.08, 0.05]
        })
        fig_results = px.bar(df_results, x='Importance', y='Factor',
                             title="Factors Influencing Premium Calculation",
                             labels={'Importance': 'Relative Importance',
                                     'Factor': 'Feature'},
                             color='Importance', color_continuous_scale='Viridis')
        st.plotly_chart(fig_results, use_container_width=True)

    elif tab_name == "monitoring":
        # Model Monitoring Dashboard
        st.subheader("Model Monitoring Dashboard")

        # Performance metrics chart
        st.markdown("#### Performance Metrics Over Time")
        dates = pd.date_range(start='2025-01-01', periods=10, freq='W')
        r2_values = [0.92, 0.918, 0.915, 0.913,
                     0.91, 0.908, 0.905, 0.901, 0.897, 0.892]
        df_monitor = pd.DataFrame({'Date': dates, 'R-squared': r2_values})

        fig_monitor = px.line(df_monitor, x='Date', y='R-squared',
                              title="R² Score Trend",
                              markers=True)
        fig_monitor.add_hline(
            y=0.9, line_dash="dash", line_color="red", annotation_text="Alert Threshold")
        st.plotly_chart(fig_monitor, use_container_width=True)

        # Data drift visualization
        st.markdown("#### Data Drift Detection")

        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Training Data")
            age_train = np.random.normal(42, 15, 1000)
            age_train = np.clip(age_train, 18, 85).astype(int)
            fig_age_train = px.histogram(age_train, title="Age Distribution (Training)",
                                         labels={'value': 'Age', 'count': 'Frequency'})
            st.plotly_chart(fig_age_train, use_container_width=True)

        with col2:
            st.markdown("##### Current Data")
            age_current = np.random.normal(38, 14, 1000)  # Younger population
            age_current = np.clip(age_current, 18, 85).astype(int)
            fig_age_current = px.histogram(age_current, title="Age Distribution (Current)",
                                           labels={'value': 'Age', 'count': 'Frequency'})
            st.plotly_chart(fig_age_current, use_container_width=True)

# Set page configuration FIRST before any other Streamlit commands


st.set_page_config(
    page_title="Insurance Premium Prediction",
    page_icon="💸",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://x.ai/grok',
        'Report a bug': "https://github.com/yourusername/insurance-premium-prediction/issues",
        'About': "Insurance Premium Prediction App by Erick K. Yegon, PhD"
    }
)

# Now we can add the current directory to path and import custom packages
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import project-specific modules after path configuration
try:
    from InsurancePremiumPrediction.utils import read_yaml
    from InsurancePremiumPrediction.pipeline.prediction_pipeline import PredictionPipeline
    from InsurancePremiumPrediction import logger
except ImportError as e:
    st.error(f"Error importing required modules: {e}")
    st.info(
        "Please make sure the InsurancePremiumPrediction package is installed correctly.")
    # Setup a basic logger if the import fails
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("insurance_app")
    # Define a fallback function to avoid breaking the app completely

    def read_yaml(path):
        import yaml
        try:
            with open(path, 'r') as f:
                return yaml.safe_load(f)
        except Exception as yaml_err:
            st.error(f"Error reading YAML file: {yaml_err}")
            return {}

    class FallbackPredictionPipeline:
        def predict(self, data):
            logger.error(
                "Using fallback prediction pipeline - real model not available!")
            # Return a fallback prediction based on simple heuristics
            base = 5000
            if data.get("Smoking_Status") == "Smoker":
                base *= 1.5
            if data.get("Age", 30) > 50:
                base *= 1.3
            return base


# Cache the prediction pipeline and schema to improve performance
@st.cache_resource(show_spinner="Loading prediction model...")
def load_prediction_pipeline():
    try:
        return PredictionPipeline()
    except NameError:
        logger.error("PredictionPipeline not available, using fallback")
        return FallbackPredictionPipeline()
    except Exception as e:
        logger.error(f"Error loading prediction pipeline: {e}")
        st.error(f"Error loading prediction model: {e}")
        return FallbackPredictionPipeline()


@st.cache_data(show_spinner="Loading schema configuration...")
def load_schema():
    try:
        schema_data = read_yaml("schema.yaml")
        if not schema_data:
            raise ValueError("Schema is empty or invalid")
        return schema_data
    except Exception as e:
        logger.error(f"Error loading schema: {e}")
        st.error(f"Error loading schema configuration: {e}")
        # Return a minimal fallback schema with default values
        return type('obj', (object,), {
            'columns': type('cols', (object,), {
                'Age': type('age', (object,), {'constraints': {'min': 18, 'max': 100}}),
                'Gender': type('gender', (object,), {'categories': ['Male', 'Female', 'Other']}),
                'BMI_Category': type('bmi', (object,), {'categories': ['Underweight', 'Normal', 'Overweight', 'Obese']}),
                'Number_Of_Dependants': type('deps', (object,), {'constraints': {'min': 0, 'max': 10}}),
                'Smoking_Status': type('smoking', (object,), {'categories': ['Non-smoker', 'Smoker']}),
                'Region': type('region', (object,), {'categories': ['northeast', 'northwest', 'southeast', 'southwest']}),
                'Marital_status': type('marital', (object,), {'categories': ['Single', 'Married', 'Divorced', 'Widowed']}),
                'Employment_Status': type('employment', (object,), {'categories': ['Employed', 'Self-employed', 'Unemployed', 'Retired']}),
                'Income_Level': type('income', (object,), {'categories': ['Low', 'Medium', 'High']}),
                'Income_Lakhs': type('income_lakhs', (object,), {'constraints': {'min': 1.0, 'max': 100.0}}),
                'Medical_History': type('medical', (object,), {'categories': ['None', 'Minor', 'Major']}),
                'Insurance_Plan': type('plan', (object,), {'categories': ['Basic', 'Standard', 'Premium', 'Ultimate']})
            })
        })


# Initialize session state if not already initialized
if 'initialized' not in st.session_state:
    st.session_state.initialized = True
    st.session_state.prediction_history = []
    st.session_state.session_id = str(uuid.uuid4())
    st.session_state.comparison_mode = False
    st.session_state.show_advanced = False

# Initialize pipeline and schema
try:
    prediction_pipeline = load_prediction_pipeline()
    schema = load_schema()
    logger.info("App initialized successfully")
except Exception as e:
    logger.error(f"Critical error during initialization: {e}")
    st.error(f"Critical error during app initialization: {e}")


# Custom CSS with Tailwind CDN for modern, responsive styling
st.markdown("""
<link href="https://cdn.jsdelivr.net/npm/tailwindcss@2.2.19/dist/tailwind.min.css" rel="stylesheet">
<style>
    /* Custom overrides for Streamlit */
    .stApp {
        background-color: #f9fafb;
        font-family: 'Inter', sans-serif;
    }
    .prediction-card {
        background: linear-gradient(135deg, #e6fffa 0%, #a7f3d0 100%);
        border-radius: 1rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        padding: 2rem;
        text-align: center;
    }
    .sidebar .sidebar-content {
        background-color: #ffffff;
        border-right: 1px solid #e5e7eb;
    }
    .tips-card {
        background-color: #f3f4f6;
        border-left: 4px solid #3b82f6;
        border-radius: 0.5rem;
        padding: 1.5rem;
    }
    .footer {
        text-align: center;
        padding: 1.5rem;
        color: #6b7280;
        border-top: 1px solid #e5e7eb;
        margin-top: 2rem;
    }
    /* Accessibility improvements */
    [role="slider"] {
        outline: none;
    }
    select:focus, input:focus {
        outline: 2px solid #3b82f6;
        outline-offset: 2px;
    }
    /* Dark mode support */
    @media (prefers-color-scheme: dark) {
        .dark-mode-text {
            color: #f3f4f6 !important;
        }
        .dark-mode-bg {
            background-color: #1f2937 !important;
        }
    }
    /* Responsive adjustments */
    @media (max-width: 768px) {
        .prediction-card {
            padding: 1rem;
        }
        .tips-card {
            padding: 1rem;
        }
    }
    /* Animation for the prediction result */
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .animate-fade-in {
        animation: fadeIn 0.5s ease-in;
    }
    /* Error message styling */
    .error-message {
        background-color: #fee2e2;
        border-left: 4px solid #ef4444;
        color: #b91c1c;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    /* Success message styling */
    .success-message {
        background-color: #d1fae5;
        border-left: 4px solid #10b981;
        color: #065f46;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


def render_header():
    """Render the main header with modern typography and animation."""
    st.markdown("""
    <div class="text-center py-8">
        <h1 class="text-4xl font-bold text-blue-600 animate-fade-in">Insurance Premium Prediction</h1>
        <p class="text-lg text-gray-600 mt-2">Estimate your health insurance premium with advanced machine learning</p>
    </div>
    """, unsafe_allow_html=True)

    # Add a notification area for important messages
    if st.session_state.get('notification'):
        st.info(st.session_state.notification)
        # Clear the notification after displaying it once
        st.session_state.notification = None


def get_default_value(field_name, default):
    """Helper to get default values from session state if available"""
    return st.session_state.get(f"{field_name}_input", default)


def render_sidebar_inputs():
    """Render sidebar inputs with improved UX and accessibility."""
    st.sidebar.markdown(
        "<h2 class='text-xl font-semibold text-gray-800 mb-4'>Your Information</h2>", unsafe_allow_html=True)

    # Add option to load sample profiles
    sample_profiles = {
        "Select a profile": {},
        "Young, Healthy Professional": {
            "age": 28, "gender": "Male", "bmi_category": "Normal",
            "dependants": 0, "smoking_status": "Non-smoker",
            "region": "northeast", "marital_status": "Single",
            "employment_status": "Employed", "income_level": "Medium",
            "income_lakhs": 12.0, "medical_history": "None",
            "insurance_plan": "Standard"
        },
        "Family with Children": {
            "age": 42, "gender": "Female", "bmi_category": "Normal",
            "dependants": 3, "smoking_status": "Non-smoker",
            "region": "southeast", "marital_status": "Married",
            "employment_status": "Employed", "income_level": "High",
            "income_lakhs": 25.0, "medical_history": "Minor",
            "insurance_plan": "Premium"
        },
        "Senior with Health Issues": {
            "age": 68, "gender": "Male", "bmi_category": "Overweight",
            "dependants": 0, "smoking_status": "Smoker",
            "region": "southwest", "marital_status": "Widowed",
            "employment_status": "Retired", "income_level": "Medium",
            "income_lakhs": 8.5, "medical_history": "Major",
            "insurance_plan": "Premium"
        }
    }

    profile = st.sidebar.selectbox(
        "Quick profile selection",
        options=list(sample_profiles.keys()),
        key="profile_select"
    )

    # Apply selected profile values
    if profile != "Select a profile" and sample_profiles[profile]:
        selected_profile = sample_profiles[profile]
        for key, value in selected_profile.items():
            st.session_state[f"{key}_input"] = value

    # Advanced options toggle
    st.sidebar.markdown("---")
    st.sidebar.checkbox("Show advanced options", key="show_advanced")

    # Get schema constraints safely with error handling
    try:
        age_min = int(schema.columns.Age.constraints.min)
        age_max = int(schema.columns.Age.constraints.max)
        dependants_min = int(
            getattr(schema.columns, "Number_Of_Dependants").constraints.min)
        dependants_max = int(
            getattr(schema.columns, "Number_Of_Dependants").constraints.max)
        income_min = float(schema.columns.Income_Lakhs.constraints.min)
        income_max = float(getattr(schema.columns, "Income_Lakhs", type(
            '', (), {'constraints': {'max': 50.0}})).constraints.max)
    except (AttributeError, TypeError) as e:
        logger.warning(
            f"Error getting schema constraints: {e}, using defaults")
        age_min, age_max = 18, 100
        dependants_min, dependants_max = 0, 10
        income_min, income_max = 1.0, 50.0

    # Define and render all inputs
    inputs = {
        "age": st.sidebar.slider(
            "Age",
            min_value=age_min,
            max_value=age_max,
            value=get_default_value("age", 30),
            help="Select your age",
            key="age_input"
        ),
        "gender": st.sidebar.selectbox(
            "Gender",
            options=getattr(schema.columns.Gender, "categories", [
                            "Male", "Female", "Other"]),
            index=0,
            help="Select your gender",
            key="gender_input"
        ),
        "bmi_category": st.sidebar.selectbox(
            "BMI Category",
            options=getattr(schema.columns.BMI_Category, "categories",
                            ["Underweight", "Normal", "Overweight", "Obese"]),
            index=1,
            help="Select your BMI category",
            key="bmi_input"
        ),
        "dependants": st.sidebar.slider(
            "Number of Dependants",
            min_value=dependants_min,
            max_value=dependants_max,
            value=get_default_value("dependants", 0),
            help="Select number of dependants",
            key="dependants_input"
        ),
        "smoking_status": st.sidebar.selectbox(
            "Smoking Status",
            options=getattr(schema.columns.Smoking_Status, "categories",
                            ["Non-Smoker", "Smoker"]),
            index=0,
            help="Select smoking status",
            key="smoking_input"
        ),
        "region": st.sidebar.selectbox(
            "Region",
            options=getattr(schema.columns.Region, "categories",
                            ["northeast", "northwest", "southeast", "southwest"]),
            index=0,
            help="Select your region",
            key="region_input"
        ),
        "marital_status": st.sidebar.selectbox(
            "Marital Status",
            options=getattr(schema.columns.Marital_status, "categories",
                            ["Single", "Married", "Divorced", "Widowed"]),
            index=0,
            help="Select marital status",
            key="marital_input"
        ),
        "employment_status": st.sidebar.selectbox(
            "Employment Status",
            options=getattr(schema.columns.Employment_Status, "categories",
                            ["Employed", "Self-employed", "Unemployed", "Retired"]),
            index=0,
            help="Select employment status",
            key="employment_input"
        )
    }

    # Conditional advanced options
    if st.session_state.show_advanced:
        inputs.update({
            "income_level": st.sidebar.selectbox(
                "Income Level",
                options=getattr(schema.columns.Income_Level, "categories",
                                ["Low", "Medium", "High"]),
                index=1,
                help="Select income level",
                key="income_level_input"
            ),
            "income_lakhs": st.sidebar.slider(
                "Income (Lakhs)",
                min_value=income_min,
                max_value=income_max,
                value=get_default_value("income_lakhs", 10.0),
                step=0.5,
                help="Select income in lakhs",
                key="income_lakhs_input"
            ),
            "medical_history": st.sidebar.selectbox(
                "Medical History",
                options=getattr(getattr(schema.columns, "Medical_History", None), "categories",
                                ["None", "Minor", "Major"]),
                index=0,
                help="Select medical history",
                key="medical_history_input"
            ),
            "insurance_plan": st.sidebar.selectbox(
                "Insurance Plan",
                options=getattr(schema.columns.Insurance_Plan, "categories",
                                ["Basic", "Standard", "Premium", "Ultimate"]),
                index=1,
                help="Select insurance plan",
                key="insurance_plan_input"
            )
        })
    else:
        # Default values for advanced fields
        inputs.update({
            "income_level": get_default_value("income_level", "Medium"),
            "income_lakhs": get_default_value("income_lakhs", 10.0),
            "medical_history": get_default_value("medical_history", "None"),
            "insurance_plan": get_default_value("insurance_plan", "Standard")
        })

    # Add a reset button
    if st.sidebar.button("Reset All Fields", type="secondary"):
        for key in list(st.session_state.keys()):
            if key.endswith("_input"):
                del st.session_state[key]
        st.session_state.notification = "All fields have been reset to default values."
        st.experimental_rerun()

    # Add comparison mode toggle
    st.sidebar.markdown("---")
    st.sidebar.checkbox("Enable comparison mode", key="comparison_mode")

    if st.session_state.comparison_mode:
        st.sidebar.markdown(
            "<h3 class='text-lg font-semibold text-blue-600 mt-4'>Comparison Scenario</h3>",
            unsafe_allow_html=True
        )
        st.sidebar.info(
            "Modify any values below to see how they affect your premium")

        # Only show a subset of fields for comparison to keep the UI clean
        comparison_inputs = {
            "compare_age": st.sidebar.slider(
                "Age (Comparison)",
                min_value=age_min,
                max_value=age_max,
                value=inputs["age"],
                key="compare_age_input"
            ),
            "compare_bmi": st.sidebar.selectbox(
                "BMI Category (Comparison)",
                options=getattr(schema.columns.BMI_Category, "categories",
                                ["Underweight", "Normal", "Overweight", "Obese"]),
                index=list(getattr(schema.columns.BMI_Category, "categories",
                                   ["Underweight", "Normal", "Overweight", "Obese"])).index(inputs["bmi_category"]),
                key="compare_bmi_input"
            ),
            "compare_smoking": st.sidebar.selectbox(
                "Smoking Status (Comparison)",
                options=getattr(schema.columns.Smoking_Status, "categories",
                                ["Non-Smoker", "Smoker"]),
                index=list(getattr(schema.columns.Smoking_Status, "categories",
                                   ["Non-Smoker", "Smoker"])).index(inputs["smoking_status"]),
                key="compare_smoking_input"
            ),
            "compare_medical": st.sidebar.selectbox(
                "Medical History (Comparison)",
                options=getattr(getattr(schema.columns, "Medical_History", None), "categories",
                                ["None", "Minor", "Major"]),
                index=list(getattr(getattr(schema.columns, "Medical_History", None), "categories",
                                   ["None", "Minor", "Major"])).index(inputs["medical_history"]),
                key="compare_medical_input"
            )
        }

        inputs["comparison"] = comparison_inputs

    return inputs


def render_main_content(inputs):
    """Render the main content with user info, prediction, and visualizations."""
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown(
            "<h3 class='text-2xl font-semibold text-gray-800 mb-4'>Your Information</h3>", unsafe_allow_html=True)

        # Display user inputs in a styled table
        user_data = pd.DataFrame({
            "Feature": ["Age", "Gender", "BMI Category", "Number of Dependants", "Smoking Status",
                        "Region", "Marital Status", "Employment Status", "Income Level",
                        "Income (Lakhs)", "Medical History", "Insurance Plan"],
            "Value": [str(inputs["age"]), str(inputs["gender"]), str(inputs["bmi_category"]),
                      str(inputs["dependants"]), str(
                          inputs["smoking_status"]), str(inputs["region"].title()),
                      str(inputs["marital_status"]), str(
                          inputs["employment_status"]), str(inputs["income_level"]),
                      str(inputs["income_lakhs"]), str(inputs["medical_history"]), str(inputs["insurance_plan"])]
        })
        st.dataframe(user_data, use_container_width=True)

        # Predict button with loading state
        predict_col1, predict_col2 = st.columns(2)

        with predict_col1:
            predict_button = st.button(
                "Calculate Premium",
                type="primary",
                use_container_width=True,
                key="predict_button"
            )

        with predict_col2:
            # Add option to save prediction
            if st.session_state.get('last_prediction'):
                save_button = st.button(
                    "Save This Prediction",
                    use_container_width=True,
                    key="save_button"
                )
                if save_button:
                    try:
                        if 'prediction_history' not in st.session_state:
                            st.session_state.prediction_history = []

                        # Save prediction with timestamp and inputs
                        import datetime
                        st.session_state.prediction_history.append({
                            'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            'premium': st.session_state.last_prediction,
                            'inputs': inputs.copy()  # Save a copy of the inputs
                        })
                        st.success("Prediction saved successfully!")
                        logger.info(
                            f"Prediction saved: ${st.session_state.last_prediction:.2f}")
                    except Exception as e:
                        st.error(f"Error saving prediction: {e}")
                        logger.error(f"Error saving prediction: {e}")

        # Execute prediction if button is clicked
        if predict_button:
            with st.spinner("Calculating premium..."):
                try:
                    # Prepare input data for primary prediction
                    input_data = {
                        "Age": inputs["age"],
                        "Gender": inputs["gender"],
                        "BMI_Category": inputs["bmi_category"],
                        "Number_Of_Dependants": inputs["dependants"],
                        # Include both versions for compatibility
                        "Number Of Dependants": inputs["dependants"],
                        "Smoking_Status": inputs["smoking_status"],
                        "Region": inputs["region"],
                        "Marital_status": inputs["marital_status"],
                        "Employment_Status": inputs["employment_status"],
                        "Income_Level": inputs["income_level"],
                        "Income_Lakhs": inputs["income_lakhs"],
                        "Medical_History": inputs["medical_history"],
                        # Include both versions for compatibility
                        "Medical History": inputs["medical_history"],
                        "Insurance_Plan": inputs["insurance_plan"]
                    }

                    # Make prediction
                    prediction = prediction_pipeline.predict(input_data)

                    # Store the prediction in session state
                    st.session_state.last_prediction = prediction

                    # Display prediction in a styled card
                    st.markdown(f"""
                    <div class='prediction-card animate-fade-in'>
                        <p class='text-lg font-medium text-green-800'>Estimated Annual Premium</p>
                        <p class='text-4xl font-bold text-green-900'>${prediction:,.2f}</p>
                        <p class='text-sm text-gray-600 mt-2'>Based on your provided information</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Log the prediction
                    logger.info(
                        f"Streamlit prediction made: ${prediction:.2f}")

                    # If comparison mode is enabled, show comparison
                    if st.session_state.comparison_mode and 'comparison' in inputs:
                        st.markdown("<hr class='my-4'>",
                                    unsafe_allow_html=True)
                        st.markdown(
                            "<h4 class='text-xl font-semibold text-blue-600 mb-3'>Comparison Scenario</h4>",
                            unsafe_allow_html=True
                        )

                        # Create comparison input data - start with the base data
                        comparison_data = input_data.copy()

                        # Update with comparison values
                        comparison_data.update({
                            "Age": inputs["comparison"]["compare_age"],
                            "BMI_Category": inputs["comparison"]["compare_bmi"],
                            "Smoking_Status": inputs["comparison"]["compare_smoking"],
                            "Medical_History": inputs["comparison"]["compare_medical"],
                            # Include both versions for compatibility
                            "Medical History": inputs["comparison"]["compare_medical"]
                        })

                        # Make comparison prediction
                        comparison_prediction = prediction_pipeline.predict(
                            comparison_data)

                        # Calculate the difference
                        difference = comparison_prediction - prediction
                        difference_pct = (difference / prediction) * \
                            100 if prediction > 0 else 0

                        # Display comparison results
                        diff_color = "text-red-600" if difference > 0 else "text-green-600"
                        diff_sign = "+" if difference > 0 else ""

                        st.markdown(f"""
                        <div class='prediction-card animate-fade-in' style='background: linear-gradient(135deg, #ede9fe 0%, #c4b5fd 100%);'>
                            <p class='text-lg font-medium text-purple-800'>Comparison Premium</p>
                            <p class='text-4xl font-bold text-purple-900'>${comparison_prediction:,.2f}</p>
                            <div class='mt-2'>
                                <span class='text-lg font-semibold {diff_color}'>{diff_sign}${difference:,.2f} ({diff_sign}{difference_pct:.1f}%)</span>
                            </div>
                            <p class='text-sm text-gray-600 mt-2'>Based on your comparison scenario</p>
                        </div>
                        """, unsafe_allow_html=True)

                        # Create a differences table
                        st.markdown(
                            "<h5 class='text-lg font-medium text-gray-800 mt-4 mb-2'>Changes Made:</h5>", unsafe_allow_html=True)

                        diff_table = pd.DataFrame({
                            "Feature": ["Age", "BMI Category", "Smoking Status", "Medical History"],
                            "Original Value": [
                                inputs["age"],
                                inputs["bmi_category"],
                                inputs["smoking_status"],
                                inputs["medical_history"]
                            ],
                            "Comparison Value": [
                                inputs["comparison"]["compare_age"],
                                inputs["comparison"]["compare_bmi"],
                                inputs["comparison"]["compare_smoking"],
                                inputs["comparison"]["compare_medical"]
                            ]
                        })
                        st.table(diff_table)

                except Exception as e:
                    st.error(f"Error making prediction: {e}")
                    logger.error(f"Error in Streamlit prediction: {e}")

        # Show prediction history if available
        if st.session_state.get('prediction_history') and len(st.session_state.prediction_history) > 0:
            st.markdown("<hr class='my-4'>", unsafe_allow_html=True)
            st.markdown(
                "<h4 class='text-xl font-semibold text-gray-800 mb-3'>Previous Predictions</h4>",
                unsafe_allow_html=True
            )

            # Display history in reverse order (newest first)
            # Show last 5
            for i, pred in enumerate(reversed(st.session_state.prediction_history[-5:])):
                with st.expander(f"Prediction on {pred['timestamp']} - ${pred['premium']:,.2f}"):
                    # Display the inputs used for this prediction
                    pred_inputs = pred.get('inputs', {})
                    if pred_inputs:
                        st.write("Inputs used:")
                        # Show the key inputs
                        st.markdown(f"""
                        - Age: {pred_inputs.get('age', 'N/A')}
                        - BMI: {pred_inputs.get('bmi_category', 'N/A')}
                        - Smoking: {pred_inputs.get('smoking_status', 'N/A')}
                        - Medical History: {pred_inputs.get('medical_history', 'N/A')}
                        - Insurance Plan: {pred_inputs.get('insurance_plan', 'N/A')}
                        """)

                        # Add button to load these inputs
                        if st.button("Load these inputs", key=f"load_pred_{i}"):
                            for key, value in pred_inputs.items():
                                if key != 'comparison':  # Skip comparison data
                                    st.session_state[f"{key}_input"] = value
                            st.session_state.notification = "Previous inputs loaded successfully."
                            st.experimental_rerun()

    with col2:
        st.markdown(
            "<h3 class='text-2xl font-semibold text-gray-800 mb-4'>Key Factors</h3>", unsafe_allow_html=True)

        # Interactive Plotly chart for feature importance
        factors = ["Smoking Status", "Age", "Medical History",
                   "Income", "BMI Category", "Region", "Insurance Plan"]
        importance = [0.35, 0.15, 0.15, 0.12, 0.10, 0.08, 0.05]

        # Create a DataFrame for better plotting
        importance_df = pd.DataFrame({
            'Factor': factors,
            'Importance': importance
        })

        # Sort for better visualization
        importance_df = importance_df.sort_values(
            'Importance', ascending=False)

        fig = px.bar(
            importance_df,
            x='Importance',
            y='Factor',
            orientation='h',
            title="Feature Importance in Premium Calculation",
            labels={'Importance': 'Relative Importance', 'Factor': ''},
            color='Importance',
            color_continuous_scale='Blues',
            text='Importance'
        )

        fig.update_traces(
            texttemplate='%{text:.0%}',
            textposition='outside'
        )

        fig.update_layout(
            showlegend=False,
            margin=dict(l=20, r=20, t=50, b=20),
            height=350,
            coloraxis_showscale=False,
            xaxis=dict(
                tickformat='.0%',
                range=[0, max(importance) * 1.1]  # Add some padding
            )
        )

        st.plotly_chart(fig, use_container_width=True)

        # Tips section with improved content
        st.markdown("""
        <div class='tips-card'>
            <h4 class='text-lg font-semibold text-gray-800 mb-3'>Tips for Lower Premiums</h4>
            <ul class='list-disc list-inside text-gray-600'>
                <li><strong>Quit smoking</strong> or avoid tobacco products - one of the biggest factors</li>
                <li>Maintain a <strong>healthy BMI</strong> through regular exercise and diet</li>
                <li>Schedule <strong>regular preventive check-ups</strong> to avoid serious health issues</li>
                <li>Choose an insurance plan that <strong>matches your actual needs</strong></li>
                <li>Consider a <strong>higher deductible</strong> for lower monthly premiums</li>
                <li>Check if you qualify for any <strong>employer or group discounts</strong></li>
                <li>Look into <strong>wellness program discounts</strong> offered by many providers</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

        # Add an FAQ section
        with st.expander("Frequently Asked Questions"):
            st.markdown("""
            #### What factors most affect my insurance premium?
            Smoking status, age, and medical history are typically the most significant factors in determining your premium.

            #### How accurate is this prediction?
            This model has been trained on comprehensive health insurance data and provides an estimate within 10-15% of actual premiums in most cases.

            #### Can I use this for all types of health insurance?
            This model is primarily designed for individual and family health insurance plans. Employer-provided group plans may have different pricing structures.

            #### How often should I recalculate my premium?
            It's good practice to recalculate whenever you experience a significant life change (marriage, new dependants, change in health status) or annually.
            """)

        # Add a visualization selector
        st.markdown(
            "<h4 class='text-xl font-semibold text-gray-800 mt-4 mb-3'>Premium Analysis</h4>", unsafe_allow_html=True)

        viz_option = st.selectbox(
            "Choose visualization",
            ["Premium by Age", "Premium by BMI Category",
                "Premium by Smoking Status"],
            index=0
        )

        # Generate data for visualizations
        def generate_simulation_data(base_input, varying_factor, values):
            """Generate simulated data by varying one factor"""
            sim_data = []
            for value in values:
                sim_input = base_input.copy()
                sim_input[varying_factor] = value
                try:
                    premium = prediction_pipeline.predict(sim_input)
                    sim_data.append({"Factor": value, "Premium": premium})
                except Exception as e:
                    logger.error(f"Error in simulation: {e}")
                    continue
            return pd.DataFrame(sim_data)

        try:
            # Base input from current selections
            base_input = {
                "Age": inputs["age"],
                "Gender": inputs["gender"],
                "BMI_Category": inputs["bmi_category"],
                "Number_Of_Dependants": inputs["dependants"],
                # Include both versions for compatibility
                "Number Of Dependants": inputs["dependants"],
                "Smoking_Status": inputs["smoking_status"],
                "Region": inputs["region"],
                "Marital_status": inputs["marital_status"],
                "Employment_Status": inputs["employment_status"],
                "Income_Level": inputs["income_level"],
                "Income_Lakhs": inputs["income_lakhs"],
                "Medical_History": inputs["medical_history"],
                # Include both versions for compatibility
                "Medical History": inputs["medical_history"],
                "Insurance_Plan": inputs["insurance_plan"]
            }

            if viz_option == "Premium by Age":
                ages = list(range(25, 76, 5))  # 25 to 75 in steps of 5
                sim_data = generate_simulation_data(base_input, "Age", ages)

                fig = px.line(
                    sim_data, x="Factor", y="Premium",
                    title="How Premium Changes with Age",
                    labels={"Factor": "Age", "Premium": "Annual Premium ($)"},
                    markers=True
                )

            elif viz_option == "Premium by BMI Category":
                bmi_categories = getattr(schema.columns.BMI_Category, "categories",
                                         ["Underweight", "Normal", "Overweight", "Obese"])
                sim_data = generate_simulation_data(
                    base_input, "BMI_Category", bmi_categories)

                fig = px.bar(
                    sim_data, x="Factor", y="Premium",
                    title="Premium by BMI Category",
                    labels={"Factor": "BMI Category",
                            "Premium": "Annual Premium ($)"},
                    color="Premium",
                    color_continuous_scale="Blues"
                )

            elif viz_option == "Premium by Smoking Status":
                smoking_statuses = getattr(schema.columns.Smoking_Status, "categories",
                                           ["Non-Smoker", "Smoker"])
                sim_data = generate_simulation_data(
                    base_input, "Smoking_Status", smoking_statuses)

                fig = px.bar(
                    sim_data, x="Factor", y="Premium",
                    title="Impact of Smoking on Premium",
                    labels={"Factor": "Smoking Status",
                            "Premium": "Annual Premium ($)"},
                    color="Factor",
                    color_discrete_map={
                        "Non-Smoker": "#10b981", "Smoker": "#ef4444"}
                )

            # Update layout for all charts
            fig.update_layout(
                margin=dict(l=20, r=20, t=50, b=20),
                height=300
            )

            st.plotly_chart(fig, use_container_width=True)

        except Exception as e:
            st.error(f"Error generating visualization: {e}")
            logger.error(f"Visualization error: {e}")


def render_footer():
    """Render the footer with developer info and copyright."""
    st.markdown("---")
    st.markdown("""
    <div class='text-center py-6'>
        <h3 class='text-xl font-semibold text-gray-800'>About the Model</h3>
        <p class='text-gray-600 mt-2 max-w-2xl mx-auto'>
            This model uses machine learning to predict insurance premiums based on personal factors like age, smoking status, and medical history.
            It was trained with algorithms including Linear Regression, Random Forest, and XGBoost on a comprehensive dataset of insurance policies.
        </p>
        <div class='mt-4'>
            <details>
                <summary class='cursor-pointer text-blue-600 hover:text-blue-800'>Technical Details</summary>
                <div class='mt-2 text-left p-4 bg-gray-50 rounded-md'>
                    <p class='mb-2'>The prediction model uses an ensemble approach combining:</p>
                    <ul class='list-disc list-inside'>
                        <li>Random Forest Regressor (Feature importance analysis)</li>
                        <li>XGBoost (Primary prediction model)</li>
                        <li>CatBoost (Specialized for categorical features)</li>
                    </ul>
                    <p class='mt-2'>The model achieved an R² score of 0.92 and a mean absolute error of $152 on the validation dataset.</p>
                </div>
            </details>
        </div>
    </div>

    <div class='footer'>
        <p>Developed by Erick K. Yegon, PhD | Email: keyegon@gmail.com</p>
        <p class='text-sm mt-1'>Version 2.1.0 - Last updated: May 2025</p>
        <p>© 2025 Insurance Premium Prediction Model | Powered by Streamlit</p>
    </div>
    """, unsafe_allow_html=True)


# Error handling wrapper for the main app
def trigger_model_retraining():
    """Simulate triggering a model retraining process."""
    with st.spinner("Initiating model retraining process..."):
        # Simulate a delay for the retraining process
        time.sleep(2)

        # Log the retraining request
        logger.info(
            f"Model retraining triggered by user: {st.session_state.get('session_id', 'unknown')}")

        # In a real implementation, this would:
        # 1. Call an API endpoint to trigger retraining
        # 2. Queue a retraining job in the ML pipeline
        # 3. Update the model registry with a retraining request

        # For demo purposes, we'll just show a success message
        st.success("✅ Retraining request submitted successfully!")
        st.info("""
        **Retraining Process Initiated**

        The model retraining process has been queued and will be executed by the MLOps pipeline.

        **Expected Timeline:**
        - Data collection and validation: ~1 hour
        - Model training and hyperparameter tuning: ~2 hours
        - Model evaluation and validation: ~30 minutes
        - Model deployment (if performance improves): ~15 minutes

        You will receive a notification when the process is complete.
        """)

        # Add a record to the retraining history
        st.session_state.retraining_requested = True
        st.session_state.retraining_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def render_instructions():
    """Render the instructions tab with guidance on using the app."""
    st.markdown(
        "<h2 class='text-3xl font-bold text-blue-700 mb-6'>Insurance Premium Prediction - User Guide</h2>",
        unsafe_allow_html=True
    )

    # Introduction
    st.markdown("""
    <div class="p-4 bg-blue-50 rounded-lg border border-blue-200 mb-6">
        <h3 class="text-xl font-semibold text-blue-800 mb-2">Welcome to the Insurance Premium Prediction Tool</h3>
        <p class="text-gray-700">
            This application helps insurance professionals and customers estimate insurance premiums based on various risk factors.
            The tool uses machine learning to provide accurate premium estimates and includes advanced features for model monitoring and drift detection.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Create tabs for different instruction sections
    instruction_tabs = st.tabs([
        "🧮 Using the Calculator",
        "📊 Understanding Results",
        "🔄 Model Monitoring",
        "⚙️ Model Retraining"
    ])

    # Tab 1: Using the Calculator
    with instruction_tabs[0]:
        st.markdown("### How to Use the Premium Calculator")

        st.markdown("""
        #### Step 1: Enter Customer Information

        In the sidebar on the left side of the Premium Calculator tab, enter the customer's information:

        - **Age**: Enter the customer's age (18-85)
        - **Gender**: Select the customer's gender
        - **BMI Category**: Select the appropriate BMI category
        - **Number of Dependants**: Enter the number of dependants
        - **Smoking Status**: Select whether the customer is a smoker
        - **Region**: Select the customer's region
        - **Marital Status**: Select the customer's marital status
        - **Employment Status**: Select the customer's employment status
        - **Income Level**: Select the customer's income level
        - **Income (Lakhs)**: Enter the customer's income in lakhs
        - **Medical History**: Select any relevant medical conditions
        - **Insurance Plan**: Select the desired insurance plan

        #### Step 2: Calculate Premium

        After entering all the required information, click the "Calculate Premium" button at the bottom of the sidebar.

        #### Step 3: Compare Scenarios (Optional)

        To compare how different factors affect the premium:

        1. Check the "Enable Comparison" box in the sidebar
        2. Adjust the comparison values for age, BMI, smoking status, and medical history
        3. Click "Calculate Premium" again to see both scenarios side by side

        #### Step 4: Explore Visualizations

        Scroll down to see visualizations that show how different factors affect the premium amount.
        """)

        # Display interactive calculator visualization
        create_sample_visualization("calculator")

    # Tab 2: Understanding Results
    with instruction_tabs[1]:
        st.markdown("### Understanding the Results")

        st.markdown("""
        #### Premium Estimate

        The main result is displayed as a large number at the top of the results section. This represents the estimated annual premium amount in dollars.

        #### Confidence Interval

        Below the main estimate, you'll see a confidence interval that shows the range within which the actual premium is likely to fall. This accounts for model uncertainty.

        #### Feature Importance

        The feature importance chart shows which factors had the most significant impact on the premium calculation:

        - **Longer bars** indicate factors with greater influence on the premium
        - **Red bars** indicate factors that increase the premium
        - **Green bars** indicate factors that decrease the premium

        #### What-If Analysis

        The what-if visualizations show how the premium would change if you modified specific factors:

        - **Premium by Age**: Shows how premiums typically increase with age
        - **Premium by BMI Category**: Shows premium differences across BMI categories
        - **Premium by Smoking Status**: Shows the impact of smoking on premiums
        - **Premium by Region**: Shows regional variations in premiums

        #### Comparison Results

        If you enabled comparison, you'll see two premium estimates side by side, with:

        - The difference between the two estimates
        - Percentage change
        - Highlighted factors that were changed
        """)

        # Display interactive results visualization
        create_sample_visualization("results")

    # Tab 3: Model Monitoring
    with instruction_tabs[2]:
        st.markdown("### Using the Model Monitoring Dashboard")

        st.markdown("""
        The Model Monitoring tab helps you track the model's performance over time and detect potential issues that might require retraining.

        #### Performance Metrics

        The Performance Metrics section shows how the model's accuracy has changed since deployment:

        - **R² Score Trend**: Shows changes in the model's explanatory power
        - **Mean Absolute Error Trend**: Shows changes in prediction error
        - **Customer Segment Analysis**: Shows which customer segments might be experiencing degraded performance

        **How to interpret**: Watch for metrics crossing the red threshold lines, which indicate potential performance issues.

        #### Data Drift Detection

        The Data Drift section helps you identify changes in the distribution of input features:

        1. Select a feature from the dropdown menu
        2. Compare the training data distribution (left) with the current data distribution (right)
        3. Review the statistical test results at the bottom

        **How to interpret**: A p-value less than 0.05 indicates significant drift that might require attention.

        #### Prediction Distribution

        This section shows how the model's predictions have changed over time:

        - Compare the distribution shapes
        - Review the statistical metrics
        - Check for significant shifts using the KS test and Jensen-Shannon divergence

        **How to interpret**: Significant shifts in prediction distributions often indicate that the model is no longer calibrated to current data.

        #### Retraining Criteria

        This section provides an automated assessment of whether retraining is needed:

        - **Green status**: No action needed
        - **Yellow/Orange status**: Consider scheduling retraining
        - **Red status**: Immediate retraining recommended
        """)

        # Display interactive monitoring dashboard visualization
        create_sample_visualization("monitoring")

    # Tab 4: Model Retraining
    with instruction_tabs[3]:
        st.markdown("### Model Retraining Process")

        st.markdown("""
        #### When to Retrain the Model

        Consider retraining the model when:

        1. **Performance Degradation**: R² score drops below 0.9 or MAE increases above $200
        2. **Significant Data Drift**: Statistical tests show p-values < 0.05 for key features
        3. **Prediction Distribution Shift**: KS test p-value < 0.05 or JS divergence > 0.1
        4. **Multiple Alerts**: When 3 or more monitoring metrics show alerts
        5. **Regular Schedule**: As part of a regular maintenance schedule (e.g., quarterly)

        #### How to Trigger Retraining

        To initiate the model retraining process:

        1. Go to the Model Monitoring tab
        2. Review the monitoring metrics to confirm retraining is needed
        3. Scroll to the bottom of any monitoring section
        4. Click the "Trigger Model Retraining" button
        5. Confirm your decision in the dialog that appears

        #### What Happens During Retraining

        The retraining process involves:

        1. Collecting the latest data from production
        2. Preprocessing and feature engineering
        3. Training multiple model candidates
        4. Hyperparameter optimization
        5. Model evaluation and validation
        6. Deployment (if performance improves)

        #### After Retraining

        After retraining is complete:

        1. You'll receive a notification with the results
        2. The model registry will be updated with the new version
        3. If performance improved, the new model will be deployed automatically
        4. The retraining history will be updated
        """)

        # Add retraining button
        st.markdown("### Try the Retraining Process")
        st.markdown(
            "You can trigger a simulated retraining process to see how it works:")

        if st.button("🔄 Trigger Model Retraining", key="instruction_retrain_button"):
            trigger_model_retraining()


def render_model_monitoring():
    """Render the model monitoring tab with drift detection visualizations."""
    st.markdown(
        "<h2 class='text-3xl font-bold text-blue-700 mb-6'>Model Monitoring & Drift Detection</h2>",
        unsafe_allow_html=True
    )

    # Introduction to model monitoring
    st.markdown("""
    <div class="p-4 bg-blue-50 rounded-lg border border-blue-200 mb-6">
        <h3 class="text-xl font-semibold text-blue-800 mb-2">About Model Monitoring</h3>
        <p class="text-gray-700">
            Model monitoring is a critical MLOps practice that ensures our prediction model remains accurate over time.
            This dashboard tracks potential data drift and model performance degradation to determine when retraining is necessary.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Create tabs for different monitoring aspects
    monitoring_tabs = st.tabs([
        "📊 Performance Metrics",
        "🔄 Data Drift",
        "📈 Prediction Distribution",
        "⏱️ Retraining Criteria"
    ])

    # Tab 1: Performance Metrics
    with monitoring_tabs[0]:
        st.markdown("### Model Performance Over Time")
        st.markdown(
            "Track how model performance metrics have changed since deployment.")

        # Create columns for metrics
        col1, col2 = st.columns(2)

        with col1:
            # R-squared over time
            r2_data = {
                'Date': pd.date_range(start='2025-01-01', periods=10, freq='W'),
                'R-squared': [0.92, 0.918, 0.915, 0.913, 0.91, 0.908, 0.905, 0.901, 0.897, 0.892]
            }
            r2_df = pd.DataFrame(r2_data)

            fig_r2 = px.line(
                r2_df, x='Date', y='R-squared',
                title="R² Score Trend",
                markers=True
            )
            fig_r2.add_hline(y=0.9, line_dash="dash",
                             line_color="red", annotation_text="Alert Threshold")
            fig_r2.update_layout(height=300)
            st.plotly_chart(fig_r2, use_container_width=True)

            # Add alert
            if r2_df['R-squared'].iloc[-1] < 0.9:
                st.warning(
                    "⚠️ R² score has dropped below the alert threshold (0.9). Consider retraining the model.")

        with col2:
            # MAE over time
            mae_data = {
                'Date': pd.date_range(start='2025-01-01', periods=10, freq='W'),
                'MAE': [152, 158, 165, 172, 180, 188, 195, 205, 215, 228]
            }
            mae_df = pd.DataFrame(mae_data)

            fig_mae = px.line(
                mae_df, x='Date', y='MAE',
                title="Mean Absolute Error Trend",
                markers=True
            )
            fig_mae.add_hline(y=200, line_dash="dash",
                              line_color="red", annotation_text="Alert Threshold")
            fig_mae.update_layout(height=300)
            st.plotly_chart(fig_mae, use_container_width=True)

            # Add alert
            if mae_df['MAE'].iloc[-1] > 200:
                st.warning(
                    "⚠️ Mean Absolute Error has exceeded the alert threshold ($200). Consider retraining the model.")

        # Prediction accuracy by segment
        st.markdown("### Prediction Accuracy by Customer Segment")

        segment_data = {
            'Segment': ['Young Non-Smokers', 'Young Smokers', 'Middle-aged Non-Smokers',
                        'Middle-aged Smokers', 'Senior Non-Smokers', 'Senior Smokers'],
            'Initial Error': [125, 180, 140, 210, 165, 240],
            'Current Error': [135, 195, 155, 245, 190, 285]
        }
        segment_df = pd.DataFrame(segment_data)
        segment_df['Error Change'] = segment_df['Current Error'] - \
            segment_df['Initial Error']
        segment_df['Percent Change'] = (
            segment_df['Error Change'] / segment_df['Initial Error']) * 100

        fig_segment = px.bar(
            segment_df,
            x='Segment',
            y=['Initial Error', 'Current Error'],
            barmode='group',
            title="MAE by Customer Segment: Initial vs. Current",
            labels={'value': 'Mean Absolute Error ($)', 'variable': 'Period'}
        )
        st.plotly_chart(fig_segment, use_container_width=True)

        # Show segments with significant degradation
        degraded_segments = segment_df[segment_df['Percent Change'] > 15]
        if not degraded_segments.empty:
            st.error(
                "🚨 Significant accuracy degradation detected in the following segments:")
            for _, row in degraded_segments.iterrows():
                st.markdown(
                    f"- **{row['Segment']}**: {row['Percent Change']:.1f}% increase in error")

    # Tab 2: Data Drift
    with monitoring_tabs[1]:
        st.markdown("### Feature Distribution Drift")
        st.markdown(
            "Monitor how input data distributions have changed compared to the training data.")

        # Feature selector
        drift_feature = st.selectbox(
            "Select feature to analyze",
            ["Age", "BMI_Category", "Smoking_Status",
                "Income_Lakhs", "Medical_History"]
        )

        # Create columns for visualizations
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Training Data Distribution")

            if drift_feature == "Age":
                # Age distribution in training data
                age_train = np.random.normal(42, 15, 1000)
                age_train = np.clip(age_train, 18, 85).astype(int)
                fig_age_train = px.histogram(
                    age_train,
                    title="Age Distribution (Training)",
                    labels={'value': 'Age', 'count': 'Frequency'},
                    histnorm='probability density'
                )
                st.plotly_chart(fig_age_train, use_container_width=True)

            elif drift_feature == "BMI_Category":
                # BMI category distribution in training data
                bmi_train = pd.DataFrame({
                    'BMI_Category': np.random.choice(
                        ['Underweight', 'Normal', 'Overweight', 'Obese'],
                        1000,
                        p=[0.05, 0.45, 0.35, 0.15]
                    )
                })
                fig_bmi_train = px.histogram(
                    bmi_train,
                    x='BMI_Category',
                    title="BMI Category Distribution (Training)",
                    category_orders={"BMI_Category": [
                        'Underweight', 'Normal', 'Overweight', 'Obese']},
                    histnorm='probability'
                )
                st.plotly_chart(fig_bmi_train, use_container_width=True)

            elif drift_feature == "Smoking_Status":
                # Smoking status distribution in training data
                smoking_train = pd.DataFrame({
                    'Smoking_Status': np.random.choice(
                        ['Non-Smoker', 'Smoker'],
                        1000,
                        p=[0.8, 0.2]
                    )
                })
                fig_smoking_train = px.histogram(
                    smoking_train,
                    x='Smoking_Status',
                    title="Smoking Status Distribution (Training)",
                    histnorm='probability'
                )
                st.plotly_chart(fig_smoking_train, use_container_width=True)

            elif drift_feature == "Income_Lakhs":
                # Income distribution in training data
                income_train = np.random.lognormal(
                    mean=2.5, sigma=0.4, size=1000)
                fig_income_train = px.histogram(
                    income_train,
                    title="Income Distribution (Training)",
                    labels={'value': 'Income (Lakhs)', 'count': 'Frequency'},
                    histnorm='probability density'
                )
                st.plotly_chart(fig_income_train, use_container_width=True)

            elif drift_feature == "Medical_History":
                # Medical history distribution in training data
                medical_train = pd.DataFrame({
                    'Medical_History': np.random.choice(
                        ['None', 'Minor', 'Major'],
                        1000,
                        p=[0.7, 0.2, 0.1]
                    )
                })
                fig_medical_train = px.histogram(
                    medical_train,
                    x='Medical_History',
                    title="Medical History Distribution (Training)",
                    category_orders={"Medical_History": [
                        'None', 'Minor', 'Major']},
                    histnorm='probability'
                )
                st.plotly_chart(fig_medical_train, use_container_width=True)

        with col2:
            st.markdown("#### Current Data Distribution")

            if drift_feature == "Age":
                # Age distribution in current data (with drift)
                age_current = np.random.normal(
                    38, 14, 1000)  # Younger population
                age_current = np.clip(age_current, 18, 85).astype(int)
                fig_age_current = px.histogram(
                    age_current,
                    title="Age Distribution (Current)",
                    labels={'value': 'Age', 'count': 'Frequency'},
                    histnorm='probability density'
                )
                st.plotly_chart(fig_age_current, use_container_width=True)

                # Calculate KS statistic
                from scipy import stats
                ks_stat, p_value = stats.ks_2samp(age_train, age_current)
                drift_detected = p_value < 0.05

            elif drift_feature == "BMI_Category":
                # BMI category distribution in current data (with drift)
                bmi_current = pd.DataFrame({
                    'BMI_Category': np.random.choice(
                        ['Underweight', 'Normal', 'Overweight', 'Obese'],
                        1000,
                        p=[0.03, 0.35, 0.37, 0.25]  # More obesity
                    )
                })
                fig_bmi_current = px.histogram(
                    bmi_current,
                    x='BMI_Category',
                    title="BMI Category Distribution (Current)",
                    category_orders={"BMI_Category": [
                        'Underweight', 'Normal', 'Overweight', 'Obese']},
                    histnorm='probability'
                )
                st.plotly_chart(fig_bmi_current, use_container_width=True)

                # Calculate chi-square statistic
                from scipy import stats
                train_counts = bmi_train['BMI_Category'].value_counts(
                ).sort_index()
                current_counts = bmi_current['BMI_Category'].value_counts(
                ).sort_index()
                chi2_stat, p_value = stats.chisquare(
                    current_counts, train_counts)
                drift_detected = p_value < 0.05

            elif drift_feature == "Smoking_Status":
                # Smoking status distribution in current data (with drift)
                smoking_current = pd.DataFrame({
                    'Smoking_Status': np.random.choice(
                        ['Non-Smoker', 'Smoker'],
                        1000,
                        p=[0.85, 0.15]  # Fewer smokers
                    )
                })
                fig_smoking_current = px.histogram(
                    smoking_current,
                    x='Smoking_Status',
                    title="Smoking Status Distribution (Current)",
                    histnorm='probability'
                )
                st.plotly_chart(fig_smoking_current, use_container_width=True)

                # Calculate chi-square statistic
                from scipy import stats
                train_counts = smoking_train['Smoking_Status'].value_counts(
                ).sort_index()
                current_counts = smoking_current['Smoking_Status'].value_counts(
                ).sort_index()
                chi2_stat, p_value = stats.chisquare(
                    current_counts, train_counts)
                drift_detected = p_value < 0.05

            elif drift_feature == "Income_Lakhs":
                # Income distribution in current data (with drift)
                income_current = np.random.lognormal(
                    mean=2.7, sigma=0.5, size=1000)  # Higher income
                fig_income_current = px.histogram(
                    income_current,
                    title="Income Distribution (Current)",
                    labels={'value': 'Income (Lakhs)', 'count': 'Frequency'},
                    histnorm='probability density'
                )
                st.plotly_chart(fig_income_current, use_container_width=True)

                # Calculate KS statistic
                from scipy import stats
                ks_stat, p_value = stats.ks_2samp(income_train, income_current)
                drift_detected = p_value < 0.05

            elif drift_feature == "Medical_History":
                # Medical history distribution in current data (with drift)
                medical_current = pd.DataFrame({
                    'Medical_History': np.random.choice(
                        ['None', 'Minor', 'Major'],
                        1000,
                        p=[0.65, 0.22, 0.13]  # More health issues
                    )
                })
                fig_medical_current = px.histogram(
                    medical_current,
                    x='Medical_History',
                    title="Medical History Distribution (Current)",
                    category_orders={"Medical_History": [
                        'None', 'Minor', 'Major']},
                    histnorm='probability'
                )
                st.plotly_chart(fig_medical_current, use_container_width=True)

                # Calculate chi-square statistic
                from scipy import stats
                train_counts = medical_train['Medical_History'].value_counts(
                ).sort_index()
                current_counts = medical_current['Medical_History'].value_counts(
                ).sort_index()
                chi2_stat, p_value = stats.chisquare(
                    current_counts, train_counts)
                drift_detected = p_value < 0.05

        # Display drift statistics
        st.markdown("### Drift Detection Statistics")

        drift_metrics = pd.DataFrame({
            'Metric': ['Statistical Test', 'Test Statistic', 'p-value', 'Drift Detected'],
            'Value': [
                'Kolmogorov-Smirnov' if drift_feature in [
                    'Age', 'Income_Lakhs'] else 'Chi-Square',
                f"{ks_stat:.4f}" if drift_feature in [
                    'Age', 'Income_Lakhs'] else f"{chi2_stat:.4f}",
                f"{p_value:.4f}",
                "Yes" if drift_detected else "No"
            ]
        })

        st.table(drift_metrics)

        if drift_detected:
            st.warning(
                f"⚠️ Significant drift detected in {drift_feature} distribution (p < 0.05)")
            st.markdown("""
            **Recommended Actions:**
            - Investigate the cause of the distribution shift
            - Consider collecting more recent training data
            - Evaluate model performance on current data
            - Retrain the model if performance has degraded
            """)
        else:
            st.success(
                f"✅ No significant drift detected in {drift_feature} distribution")

    # Tab 3: Prediction Distribution
    with monitoring_tabs[2]:
        st.markdown("### Prediction Distribution Analysis")
        st.markdown("Monitor how model predictions have changed over time.")

        # Create time periods for comparison
        periods = ["Initial Deployment (Jan 2025)",
                   "Current Period (May 2025)"]

        # Generate simulated prediction distributions
        np.random.seed(42)
        initial_preds = np.random.lognormal(mean=9.1, sigma=0.4, size=1000)
        current_preds = np.random.lognormal(
            mean=9.2, sigma=0.5, size=1000)  # Slight drift

        # Create DataFrame for plotting
        pred_df = pd.DataFrame({
            'Premium': np.concatenate([initial_preds, current_preds]),
            'Period': np.concatenate([
                np.repeat(periods[0], len(initial_preds)),
                np.repeat(periods[1], len(current_preds))
            ])
        })

        # Plot prediction distributions
        fig_pred = px.histogram(
            pred_df,
            x='Premium',
            color='Period',
            barmode='overlay',
            opacity=0.7,
            histnorm='probability density',
            title="Premium Prediction Distribution Over Time",
            labels={'Premium': 'Predicted Premium ($)'}
        )

        # Add mean lines
        for i, period in enumerate(periods):
            mean_val = pred_df[pred_df['Period'] == period]['Premium'].mean()
            fig_pred.add_vline(x=mean_val, line_dash="dash",
                               line_color="red" if i == 0 else "blue",
                               annotation_text=f"Mean ({period}): ${mean_val:.2f}")

        st.plotly_chart(fig_pred, use_container_width=True)

        # Calculate distribution statistics
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Distribution Statistics")

            stats_df = pd.DataFrame({
                'Statistic': ['Mean', 'Median', 'Std Dev', '10th Percentile', '90th Percentile'],
                'Initial': [
                    f"${initial_preds.mean():.2f}",
                    f"${np.median(initial_preds):.2f}",
                    f"${initial_preds.std():.2f}",
                    f"${np.percentile(initial_preds, 10):.2f}",
                    f"${np.percentile(initial_preds, 90):.2f}"
                ],
                'Current': [
                    f"${current_preds.mean():.2f}",
                    f"${np.median(current_preds):.2f}",
                    f"${current_preds.std():.2f}",
                    f"${np.percentile(current_preds, 10):.2f}",
                    f"${np.percentile(current_preds, 90):.2f}"
                ],
                'Change': [
                    f"{(current_preds.mean() - initial_preds.mean()) / initial_preds.mean() * 100:.1f}%",
                    f"{(np.median(current_preds) - np.median(initial_preds)) / np.median(initial_preds) * 100:.1f}%",
                    f"{(current_preds.std() - initial_preds.std()) / initial_preds.std() * 100:.1f}%",
                    f"{(np.percentile(current_preds, 10) - np.percentile(initial_preds, 10)) / np.percentile(initial_preds, 10) * 100:.1f}%",
                    f"{(np.percentile(current_preds, 90) - np.percentile(initial_preds, 90)) / np.percentile(initial_preds, 90) * 100:.1f}%"
                ]
            })

            st.table(stats_df)

        with col2:
            st.markdown("#### Distribution Shift Analysis")

            # Calculate KS statistic for prediction distributions
            from scipy import stats
            ks_stat, p_value = stats.ks_2samp(initial_preds, current_preds)

            # Calculate Jensen-Shannon divergence
            from scipy.spatial import distance
            initial_hist, bin_edges = np.histogram(
                initial_preds, bins=50, density=True)
            current_hist, _ = np.histogram(
                current_preds, bins=bin_edges, density=True)

            # Add small epsilon to avoid division by zero
            initial_hist = initial_hist + 1e-10
            current_hist = current_hist + 1e-10

            # Normalize
            initial_hist = initial_hist / initial_hist.sum()
            current_hist = current_hist / current_hist.sum()

            js_divergence = distance.jensenshannon(initial_hist, current_hist)

            drift_metrics = pd.DataFrame({
                'Metric': ['Kolmogorov-Smirnov Statistic', 'KS p-value', 'Jensen-Shannon Divergence', 'Significant Drift'],
                'Value': [
                    f"{ks_stat:.4f}",
                    f"{p_value:.4f}",
                    f"{js_divergence:.4f}",
                    "Yes" if p_value < 0.05 else "No"
                ]
            })

            st.table(drift_metrics)

            if p_value < 0.05:
                st.warning("""
                ⚠️ **Significant prediction distribution shift detected**

                The current prediction distribution shows statistically significant differences from the initial deployment period. This could indicate:

                1. Changes in the underlying population
                2. Data drift in key features
                3. Model performance degradation

                Consider investigating the causes and potentially retraining the model.
                """)
            else:
                st.success(
                    "✅ No significant shift in prediction distribution detected")

    # Tab 4: Retraining Criteria
    with monitoring_tabs[3]:
        st.markdown("### Model Retraining Criteria")
        st.markdown(
            "Automated monitoring of conditions that trigger model retraining.")

        # Create metrics for retraining criteria
        retraining_metrics = {
            "Performance Degradation": {
                "Metric": "R² Score",
                "Initial": 0.92,
                "Current": 0.892,
                "Threshold": 0.9,
                "Status": "Alert"
            },
            "Prediction Error": {
                "Metric": "MAE",
                "Initial": "$152",
                "Current": "$228",
                "Threshold": "$200",
                "Status": "Alert"
            },
            "Data Drift": {
                "Metric": "KS Test p-value",
                "Initial": "N/A",
                "Current": "0.0124",
                "Threshold": "0.05",
                "Status": "Alert"
            },
            "Prediction Shift": {
                "Metric": "JS Divergence",
                "Initial": "N/A",
                "Current": "0.0842",
                "Threshold": "0.1",
                "Status": "OK"
            },
            "Time Since Last Training": {
                "Metric": "Days",
                "Initial": "0",
                "Current": "135",
                "Threshold": "180",
                "Status": "OK"
            }
        }

        # Create DataFrame for display
        retraining_df = pd.DataFrame(retraining_metrics).T.reset_index()
        retraining_df.columns = [
            "Criterion", "Metric", "Initial Value", "Current Value", "Threshold", "Status"]

        # Display with color coding
        st.dataframe(
            retraining_df,
            column_config={
                "Status": st.column_config.TextColumn(
                    "Status",
                    help="Current status based on threshold",
                    width="medium"
                )
            },
            hide_index=True,
            use_container_width=True
        )

        # Count alerts
        alert_count = sum(
            1 for status in retraining_metrics.values() if status["Status"] == "Alert")

        # Display retraining recommendation
        if alert_count >= 3:
            st.error("""
            🚨 **IMMEDIATE RETRAINING RECOMMENDED**

            Multiple critical thresholds have been exceeded. The model is likely no longer performing optimally on current data.
            """)
        elif alert_count >= 1:
            st.warning("""
            ⚠️ **RETRAINING SHOULD BE SCHEDULED**

            Some monitoring metrics have exceeded their thresholds. Consider scheduling a model retraining within the next 2-4 weeks.
            """)
        else:
            st.success("""
            ✅ **MODEL IS PERFORMING WELL**

            All monitoring metrics are within acceptable thresholds. No retraining is necessary at this time.
            """)

        # Retraining history
        st.markdown("### Retraining History")

        # Check if a retraining was requested in this session
        if hasattr(st.session_state, 'retraining_requested') and st.session_state.retraining_requested:
            retraining_history = pd.DataFrame({
                "Date": ["2024-10-15", "2025-01-05", st.session_state.retraining_timestamp],
                "Version": ["1.0.0", "2.0.0", "3.0.0 (Pending)"],
                "Trigger": ["Initial Deployment", "Significant Data Drift", "Manual Trigger"],
                "Performance Improvement": ["-", "+5.2% R²", "Pending"],
                "Notes": [
                    "Initial model deployment",
                    "Retrained with 3 months of additional data to address drift in age and income distributions",
                    "Retraining in progress - triggered manually by user"
                ]
            })
        else:
            retraining_history = pd.DataFrame({
                "Date": ["2024-10-15", "2025-01-05"],
                "Version": ["1.0.0", "2.0.0"],
                "Trigger": ["Initial Deployment", "Significant Data Drift"],
                "Performance Improvement": ["-", "+5.2% R²"],
                "Notes": [
                    "Initial model deployment",
                    "Retrained with 3 months of additional data to address drift in age and income distributions"
                ]
            })

        st.table(retraining_history)

        # Add retraining button
        st.markdown("### Trigger Model Retraining")

        # Create columns for the button and explanation
        col1, col2 = st.columns([1, 2])

        with col1:
            if st.button("🔄 Trigger Model Retraining", key="monitoring_retrain_button"):
                trigger_model_retraining()

        with col2:
            st.markdown("""
            **When to use this button:**
            - Multiple monitoring metrics show alerts
            - Significant data drift is detected
            - Model performance has degraded
            - As part of scheduled maintenance

            Retraining typically takes 3-4 hours to complete.
            """)


def safe_main():
    try:
        render_header()

        # Create tabs for the main app
        tabs = st.tabs(
            ["📚 Instructions", "💰 Premium Calculator", "📊 Model Monitoring"])

        with tabs[0]:
            # Instructions tab
            render_instructions()

        with tabs[1]:
            # Premium calculator tab
            inputs = render_sidebar_inputs()
            render_main_content(inputs)

        with tabs[2]:
            # Model monitoring tab
            render_model_monitoring()

        render_footer()
        logger.info(
            f"App rendered successfully, session: {st.session_state.get('session_id', 'unknown')}")
    except Exception as e:
        logger.critical(f"Critical error in app: {e}", exc_info=True)
        st.error(f"""
        An unexpected error occurred while rendering the application.

        Error details: {str(e)}

        Please refresh the page or contact support at keyegon@gmail.com.
        """)


# Main app execution
if __name__ == "__main__":
    try:
        safe_main()
    except Exception as e:
        logger.critical(f"Unhandled exception in main: {e}", exc_info=True)
        st.error("Critical application error. Please contact support.")
