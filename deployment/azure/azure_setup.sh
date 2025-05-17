#!/bin/bash
# Azure Setup Script for Insurance Premium Prediction Model

# Exit on error
set -e

# Load configuration
echo "Loading configuration..."
RESOURCE_GROUP=$(yq '.resource_group.name' config/azure_config.yaml)
LOCATION=$(yq '.resource_group.location' config/azure_config.yaml)
ACR_NAME=$(yq '.container_registry.name' config/azure_config.yaml)
ACR_SKU=$(yq '.container_registry.sku' config/azure_config.yaml)
ML_WORKSPACE=$(yq '.azure_ml.workspace.name' config/azure_config.yaml)
COMPUTE_CLUSTER=$(yq '.azure_ml.compute.cluster.name' config/azure_config.yaml)
COMPUTE_VM_SIZE=$(yq '.azure_ml.compute.cluster.vm_size' config/azure_config.yaml)
COMPUTE_MIN_NODES=$(yq '.azure_ml.compute.cluster.min_nodes' config/azure_config.yaml)
COMPUTE_MAX_NODES=$(yq '.azure_ml.compute.cluster.max_nodes' config/azure_config.yaml)
ENDPOINT_NAME=$(yq '.azure_ml.endpoints.realtime.name' config/azure_config.yaml)
APP_SERVICE_NAME=$(yq '.app_service.name' config/azure_config.yaml)
APP_SERVICE_SKU=$(yq '.app_service.sku' config/azure_config.yaml)
APP_INSIGHTS_NAME=$(yq '.monitoring.application_insights.name' config/azure_config.yaml)

# Login to Azure
echo "Logging in to Azure..."
az login

# Set subscription
echo "Setting subscription..."
SUBSCRIPTION_ID=$(az account show --query id -o tsv)
az account set --subscription $SUBSCRIPTION_ID

# Create resource group
echo "Creating resource group: $RESOURCE_GROUP..."
az group create --name $RESOURCE_GROUP --location $LOCATION

# Create Azure Container Registry
echo "Creating Azure Container Registry: $ACR_NAME..."
az acr create --resource-group $RESOURCE_GROUP --name $ACR_NAME --sku $ACR_SKU --admin-enabled true

# Create Azure ML workspace
echo "Creating Azure ML workspace: $ML_WORKSPACE..."
az ml workspace create --resource-group $RESOURCE_GROUP --name $ML_WORKSPACE

# Create compute cluster
echo "Creating compute cluster: $COMPUTE_CLUSTER..."
az ml compute create --resource-group $RESOURCE_GROUP --workspace-name $ML_WORKSPACE \
    --name $COMPUTE_CLUSTER --type AmlCompute --min-nodes $COMPUTE_MIN_NODES \
    --max-nodes $COMPUTE_MAX_NODES --size $COMPUTE_VM_SIZE

# Create environment
echo "Creating environment..."
ENV_NAME=$(yq '.azure_ml.environment.name' config/azure_config.yaml)
CONDA_FILE="deployment/azure/conda_env.yml"

az ml environment create --resource-group $RESOURCE_GROUP --workspace-name $ML_WORKSPACE \
    --name $ENV_NAME --conda-file $CONDA_FILE \
    --image mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04:latest

# Register model
echo "Registering model..."
MODEL_NAME=$(yq '.azure_ml.model.name' config/azure_config.yaml)
MODEL_PATH="artifacts/model_trainer/model.joblib"

az ml model create --resource-group $RESOURCE_GROUP --workspace-name $ML_WORKSPACE \
    --name $MODEL_NAME --version 1 --path $MODEL_PATH \
    --description "Insurance Premium Prediction Model"

# Create online endpoint
echo "Creating online endpoint: $ENDPOINT_NAME..."
az ml online-endpoint create --resource-group $RESOURCE_GROUP --workspace-name $ML_WORKSPACE \
    --name $ENDPOINT_NAME

# Create deployment
echo "Creating deployment..."
DEPLOYMENT_NAME="$MODEL_NAME-deployment"
SCORING_FILE="deployment/azure/score.py"

az ml online-deployment create --resource-group $RESOURCE_GROUP --workspace-name $ML_WORKSPACE \
    --endpoint-name $ENDPOINT_NAME --name $DEPLOYMENT_NAME \
    --model-name $MODEL_NAME --environment-name $ENV_NAME \
    --code-path $(dirname $SCORING_FILE) --scoring-script $(basename $SCORING_FILE) \
    --instance-type Standard_DS2_v2 --instance-count 1

# Create App Service
echo "Creating App Service: $APP_SERVICE_NAME..."
az appservice plan create --resource-group $RESOURCE_GROUP --name "$APP_SERVICE_NAME-plan" \
    --sku $APP_SERVICE_SKU

az webapp create --resource-group $RESOURCE_GROUP --plan "$APP_SERVICE_NAME-plan" \
    --name $APP_SERVICE_NAME --runtime "PYTHON|3.8"

# Set up Application Insights
echo "Setting up Application Insights: $APP_INSIGHTS_NAME..."
az monitor app-insights component create --resource-group $RESOURCE_GROUP \
    --app $APP_INSIGHTS_NAME --location $LOCATION

# Get Application Insights key
APP_INSIGHTS_KEY=$(az monitor app-insights component show --resource-group $RESOURCE_GROUP \
    --app $APP_INSIGHTS_NAME --query instrumentationKey -o tsv)

# Set environment variables for the web app
echo "Setting environment variables for the web app..."
az webapp config appsettings set --resource-group $RESOURCE_GROUP --name $APP_SERVICE_NAME \
    --settings AZURE_APP_INSIGHTS_KEY=$APP_INSIGHTS_KEY \
    ENDPOINT_URL=https://$ENDPOINT_NAME.azureml.ms/score

# Deploy web app
echo "Deploying web app..."
# Package the web app
mkdir -p deployment/azure/webapp
cp -r app.py templates static requirements.txt deployment/azure/webapp/

# Create a deployment package
cd deployment/azure/webapp
zip -r ../webapp.zip .
cd ../../..

# Deploy the package
az webapp deployment source config-zip --resource-group $RESOURCE_GROUP \
    --name $APP_SERVICE_NAME --src deployment/azure/webapp.zip

# Set up alerts
echo "Setting up alerts..."
for i in $(seq 0 $(yq '.monitoring.alerts | length - 1' config/azure_config.yaml)); do
    ALERT_NAME=$(yq ".monitoring.alerts[$i].name" config/azure_config.yaml)
    ALERT_DESC=$(yq ".monitoring.alerts[$i].description" config/azure_config.yaml)
    ALERT_SEVERITY=$(yq ".monitoring.alerts[$i].severity" config/azure_config.yaml)
    ALERT_METRIC=$(yq ".monitoring.alerts[$i].metric_name" config/azure_config.yaml)
    ALERT_THRESHOLD=$(yq ".monitoring.alerts[$i].threshold" config/azure_config.yaml)
    ALERT_OPERATOR=$(yq ".monitoring.alerts[$i].operator" config/azure_config.yaml)
    ALERT_WINDOW=$(yq ".monitoring.alerts[$i].window_size" config/azure_config.yaml)
    ALERT_FREQ=$(yq ".monitoring.alerts[$i].frequency" config/azure_config.yaml)
    
    echo "Creating alert: $ALERT_NAME..."
    az monitor metrics alert create --resource-group $RESOURCE_GROUP --name $ALERT_NAME \
        --description "$ALERT_DESC" \
        --scopes /subscriptions/$SUBSCRIPTION_ID/resourceGroups/$RESOURCE_GROUP/providers/Microsoft.Insights/components/$APP_INSIGHTS_NAME \
        --condition "avg $ALERT_METRIC $ALERT_OPERATOR $ALERT_THRESHOLD" \
        --window-size $ALERT_WINDOW --evaluation-frequency $ALERT_FREQ \
        --severity $ALERT_SEVERITY
done

echo "Setup completed successfully!"
echo "Model Endpoint: https://$ENDPOINT_NAME.azureml.ms/score"
echo "Web Application: https://$APP_SERVICE_NAME.azurewebsites.net"
echo "Application Insights: $APP_INSIGHTS_NAME"
