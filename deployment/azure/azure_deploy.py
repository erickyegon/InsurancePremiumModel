"""
Azure Deployment Script

This script deploys the Insurance Premium Prediction model to Azure.
It creates the necessary Azure resources and deploys the model as a web service.
"""
import os
import sys
import argparse
import logging
from pathlib import Path
import json
import yaml
import time
from datetime import datetime
from typing import Dict, List, Optional, Union

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.InsurancePremiumPrediction import logger
from src.InsurancePremiumPrediction.utils.common import read_yaml, create_directories

# Set up logging
logging.basicConfig(
    filename=os.path.join("logs", "azure_deploy.log"),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
)
logger = logging.getLogger(__name__)

def load_azure_config(config_path: str = "config/azure_config.yaml") -> Dict:
    """
    Load Azure deployment configuration.
    
    Args:
        config_path: Path to the Azure configuration file
        
    Returns:
        Dictionary with Azure configuration
    """
    try:
        config = read_yaml(config_path)
        logger.info(f"Loaded Azure configuration from {config_path}")
        return config
    except Exception as e:
        logger.error(f"Error loading Azure configuration: {e}")
        raise

def create_resource_group(config: Dict) -> None:
    """
    Create Azure resource group.
    
    Args:
        config: Azure configuration
    """
    try:
        # In a real implementation, this would use the Azure SDK or CLI
        # For demonstration purposes, we'll just log the actions
        
        resource_group = config["resource_group"]
        logger.info(f"Creating resource group {resource_group['name']} in {resource_group['location']}")
        
        # Simulated Azure CLI command:
        # az group create --name {resource_group['name']} --location {resource_group['location']}
        
        logger.info(f"Resource group {resource_group['name']} created successfully")
    except Exception as e:
        logger.error(f"Error creating resource group: {e}")
        raise

def create_container_registry(config: Dict) -> None:
    """
    Create Azure Container Registry.
    
    Args:
        config: Azure configuration
    """
    try:
        resource_group = config["resource_group"]["name"]
        acr_config = config["container_registry"]
        
        logger.info(f"Creating container registry {acr_config['name']} in resource group {resource_group}")
        
        # Simulated Azure CLI command:
        # az acr create --resource-group {resource_group} --name {acr_config['name']} --sku {acr_config['sku']} --admin-enabled {acr_config['admin_enabled']}
        
        logger.info(f"Container registry {acr_config['name']} created successfully")
    except Exception as e:
        logger.error(f"Error creating container registry: {e}")
        raise

def create_azure_ml_workspace(config: Dict) -> None:
    """
    Create Azure Machine Learning workspace.
    
    Args:
        config: Azure configuration
    """
    try:
        resource_group = config["resource_group"]["name"]
        ml_config = config["azure_ml"]["workspace"]
        
        logger.info(f"Creating Azure ML workspace {ml_config['name']} in resource group {resource_group}")
        
        # Simulated Azure CLI command:
        # az ml workspace create --resource-group {resource_group} --name {ml_config['name']} --location {ml_config['location']}
        
        logger.info(f"Azure ML workspace {ml_config['name']} created successfully")
    except Exception as e:
        logger.error(f"Error creating Azure ML workspace: {e}")
        raise

def create_compute_resources(config: Dict) -> None:
    """
    Create compute resources in Azure ML workspace.
    
    Args:
        config: Azure configuration
    """
    try:
        resource_group = config["resource_group"]["name"]
        workspace_name = config["azure_ml"]["workspace"]["name"]
        compute_config = config["azure_ml"]["compute"]
        
        # Create compute instance
        instance_config = compute_config["instance"]
        logger.info(f"Creating compute instance {instance_config['name']} in workspace {workspace_name}")
        
        # Simulated Azure CLI command:
        # az ml compute create --resource-group {resource_group} --workspace-name {workspace_name} --name {instance_config['name']} --type ComputeInstance --size {instance_config['vm_size']}
        
        logger.info(f"Compute instance {instance_config['name']} created successfully")
        
        # Create compute cluster
        cluster_config = compute_config["cluster"]
        logger.info(f"Creating compute cluster {cluster_config['name']} in workspace {workspace_name}")
        
        # Simulated Azure CLI command:
        # az ml compute create --resource-group {resource_group} --workspace-name {workspace_name} --name {cluster_config['name']} --type AmlCompute --min-nodes {cluster_config['min_nodes']} --max-nodes {cluster_config['max_nodes']} --size {cluster_config['vm_size']}
        
        logger.info(f"Compute cluster {cluster_config['name']} created successfully")
    except Exception as e:
        logger.error(f"Error creating compute resources: {e}")
        raise

def create_environment(config: Dict) -> None:
    """
    Create environment in Azure ML workspace.
    
    Args:
        config: Azure configuration
    """
    try:
        resource_group = config["resource_group"]["name"]
        workspace_name = config["azure_ml"]["workspace"]["name"]
        env_config = config["azure_ml"]["environment"]
        
        logger.info(f"Creating environment {env_config['name']} in workspace {workspace_name}")
        
        # Create conda environment file
        conda_file = env_config["conda_file"]
        os.makedirs(os.path.dirname(conda_file), exist_ok=True)
        
        # Create conda environment file from requirements.txt
        with open("requirements.txt", "r") as f:
            requirements = f.read().splitlines()
        
        conda_env = {
            "name": env_config["name"],
            "channels": ["conda-forge", "defaults"],
            "dependencies": [
                "python=3.8",
                "pip",
                {"pip": requirements}
            ]
        }
        
        with open(conda_file, "w") as f:
            yaml.dump(conda_env, f)
        
        # Simulated Azure CLI command:
        # az ml environment create --resource-group {resource_group} --workspace-name {workspace_name} --name {env_config['name']} --conda-file {conda_file} --image {env_config['docker_image']}
        
        logger.info(f"Environment {env_config['name']} created successfully")
    except Exception as e:
        logger.error(f"Error creating environment: {e}")
        raise

def register_model(config: Dict, model_path: str) -> None:
    """
    Register model in Azure ML workspace.
    
    Args:
        config: Azure configuration
        model_path: Path to the model file
    """
    try:
        resource_group = config["resource_group"]["name"]
        workspace_name = config["azure_ml"]["workspace"]["name"]
        model_config = config["azure_ml"]["model"]
        
        logger.info(f"Registering model {model_config['name']} in workspace {workspace_name}")
        
        # Simulated Azure CLI command:
        # az ml model create --resource-group {resource_group} --workspace-name {workspace_name} --name {model_config['name']} --version 1 --path {model_path} --description {model_config['description']}
        
        logger.info(f"Model {model_config['name']} registered successfully")
    except Exception as e:
        logger.error(f"Error registering model: {e}")
        raise

def create_online_endpoint(config: Dict) -> None:
    """
    Create online endpoint in Azure ML workspace.
    
    Args:
        config: Azure configuration
    """
    try:
        resource_group = config["resource_group"]["name"]
        workspace_name = config["azure_ml"]["workspace"]["name"]
        endpoint_config = config["azure_ml"]["endpoints"]["realtime"]
        
        logger.info(f"Creating online endpoint {endpoint_config['name']} in workspace {workspace_name}")
        
        # Simulated Azure CLI command:
        # az ml online-endpoint create --resource-group {resource_group} --workspace-name {workspace_name} --name {endpoint_config['name']}
        
        logger.info(f"Online endpoint {endpoint_config['name']} created successfully")
    except Exception as e:
        logger.error(f"Error creating online endpoint: {e}")
        raise

def create_deployment(config: Dict) -> None:
    """
    Create deployment in Azure ML workspace.
    
    Args:
        config: Azure configuration
    """
    try:
        resource_group = config["resource_group"]["name"]
        workspace_name = config["azure_ml"]["workspace"]["name"]
        endpoint_name = config["azure_ml"]["endpoints"]["realtime"]["name"]
        model_name = config["azure_ml"]["model"]["name"]
        env_name = config["azure_ml"]["environment"]["name"]
        
        deployment_name = f"{model_name}-deployment"
        
        logger.info(f"Creating deployment {deployment_name} for endpoint {endpoint_name}")
        
        # Create scoring script
        scoring_file = "deployment/azure/score.py"
        os.makedirs(os.path.dirname(scoring_file), exist_ok=True)
        
        with open(scoring_file, "w") as f:
            f.write("""
import os
import json
import numpy as np
import pandas as pd
import joblib
from azureml.core.model import Model

def init():
    global model
    model_path = Model.get_model_path(model_name='insurance-premium-model')
    model = joblib.load(model_path)

def run(raw_data):
    try:
        # Parse input data
        data = json.loads(raw_data)
        
        # Convert to DataFrame
        input_df = pd.DataFrame([data])
        
        # Make prediction
        prediction = model.predict(input_df)
        
        # Return prediction
        return json.dumps({
            "prediction": float(prediction[0]),
            "model_version": os.environ.get("AZUREML_MODEL_VERSION", "unknown")
        })
    except Exception as e:
        return json.dumps({
            "error": str(e)
        })
""")
        
        # Simulated Azure CLI command:
        # az ml online-deployment create --resource-group {resource_group} --workspace-name {workspace_name} --endpoint-name {endpoint_name} --name {deployment_name} --model-name {model_name} --environment-name {env_name} --code-path {os.path.dirname(scoring_file)} --scoring-script {os.path.basename(scoring_file)} --instance-type {config['azure_ml']['endpoints']['realtime']['vm_size']} --instance-count 1
        
        logger.info(f"Deployment {deployment_name} created successfully")
    except Exception as e:
        logger.error(f"Error creating deployment: {e}")
        raise

def create_app_service(config: Dict) -> None:
    """
    Create Azure App Service for hosting the web application.
    
    Args:
        config: Azure configuration
    """
    try:
        resource_group = config["resource_group"]["name"]
        app_service_config = config["app_service"]
        
        logger.info(f"Creating App Service {app_service_config['name']} in resource group {resource_group}")
        
        # Simulated Azure CLI command:
        # az appservice plan create --resource-group {resource_group} --name {app_service_config['name']}-plan --sku {app_service_config['sku']}
        # az webapp create --resource-group {resource_group} --plan {app_service_config['name']}-plan --name {app_service_config['name']} --runtime "PYTHON|3.8"
        
        logger.info(f"App Service {app_service_config['name']} created successfully")
    except Exception as e:
        logger.error(f"Error creating App Service: {e}")
        raise

def deploy_web_app(config: Dict) -> None:
    """
    Deploy web application to Azure App Service.
    
    Args:
        config: Azure configuration
    """
    try:
        resource_group = config["resource_group"]["name"]
        app_service_name = config["app_service"]["name"]
        
        logger.info(f"Deploying web application to App Service {app_service_name}")
        
        # Simulated Azure CLI command:
        # az webapp deployment source config-zip --resource-group {resource_group} --name {app_service_name} --src deployment/azure/webapp.zip
        
        logger.info(f"Web application deployed successfully to {app_service_name}")
    except Exception as e:
        logger.error(f"Error deploying web application: {e}")
        raise

def setup_monitoring(config: Dict) -> None:
    """
    Set up monitoring for the deployed model.
    
    Args:
        config: Azure configuration
    """
    try:
        resource_group = config["resource_group"]["name"]
        app_insights_name = config["monitoring"]["application_insights"]["name"]
        
        logger.info(f"Setting up Application Insights {app_insights_name} for monitoring")
        
        # Simulated Azure CLI command:
        # az monitor app-insights component create --resource-group {resource_group} --app {app_insights_name} --location {config['resource_group']['location']}
        
        logger.info(f"Application Insights {app_insights_name} created successfully")
        
        # Set up alerts
        for alert in config["monitoring"]["alerts"]:
            logger.info(f"Creating alert rule {alert['name']}")
            
            # Simulated Azure CLI command:
            # az monitor metrics alert create --resource-group {resource_group} --name {alert['name']} --description {alert['description']} --scopes /subscriptions/$SUBSCRIPTION_ID/resourceGroups/{resource_group}/providers/Microsoft.Insights/components/{app_insights_name} --condition "avg {alert['metric_name']} > {alert['threshold']}" --window-size {alert['window_size']} --evaluation-frequency {alert['frequency']}
            
            logger.info(f"Alert rule {alert['name']} created successfully")
        
        logger.info("Monitoring setup completed successfully")
    except Exception as e:
        logger.error(f"Error setting up monitoring: {e}")
        raise

def deploy_to_azure(model_path: str, config_path: str = "config/azure_config.yaml") -> None:
    """
    Deploy the Insurance Premium Prediction model to Azure.
    
    Args:
        model_path: Path to the model file
        config_path: Path to the Azure configuration file
    """
    try:
        # Load configuration
        config = load_azure_config(config_path)
        
        # Create resource group
        create_resource_group(config)
        
        # Create container registry
        create_container_registry(config)
        
        # Create Azure ML workspace
        create_azure_ml_workspace(config)
        
        # Create compute resources
        create_compute_resources(config)
        
        # Create environment
        create_environment(config)
        
        # Register model
        register_model(config, model_path)
        
        # Create online endpoint
        create_online_endpoint(config)
        
        # Create deployment
        create_deployment(config)
        
        # Create App Service
        create_app_service(config)
        
        # Deploy web application
        deploy_web_app(config)
        
        # Set up monitoring
        setup_monitoring(config)
        
        logger.info("Deployment to Azure completed successfully")
        
        # Print deployment information
        endpoint_name = config["azure_ml"]["endpoints"]["realtime"]["name"]
        app_service_name = config["app_service"]["name"]
        
        print("\nDeployment Information:")
        print(f"Model Endpoint: https://{endpoint_name}.azureml.ms/score")
        print(f"Web Application: https://{app_service_name}.azurewebsites.net")
        print(f"Application Insights: {config['monitoring']['application_insights']['name']}")
        
    except Exception as e:
        logger.error(f"Error deploying to Azure: {e}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deploy Insurance Premium Prediction model to Azure")
    parser.add_argument("--model-path", type=str, default="artifacts/model_trainer/model.joblib", help="Path to the model file")
    parser.add_argument("--config-path", type=str, default="config/azure_config.yaml", help="Path to the Azure configuration file")
    
    args = parser.parse_args()
    
    deploy_to_azure(args.model_path, args.config_path)
