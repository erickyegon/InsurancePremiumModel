"""
Alerting Module

This module provides alerting functionality for the MLOps monitoring system.
It supports email, Slack, and Azure Application Insights alerts.
"""
import os
import sys
import logging
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
import json
from typing import Dict, List, Optional, Union
from datetime import datetime
import yaml

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.InsurancePremiumPrediction import logger
from src.InsurancePremiumPrediction.utils.common import read_yaml

# Set up logging
logging.basicConfig(
    filename=os.path.join("logs", "alerting.log"),
    level=logging.INFO,
    format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
    filemode="a"
)
logger = logging.getLogger(__name__)

def load_config() -> Dict:
    """
    Load the alerting configuration from the model monitoring config file.
    
    Returns:
        Dictionary with alerting configuration
    """
    try:
        config = read_yaml("config/model_monitoring.yaml")
        return config["alerting"]
    except Exception as e:
        logger.error(f"Error loading alerting configuration: {e}")
        return {}

def send_email_alert(recipients: List[str], subject: str, message: str) -> bool:
    """
    Send an email alert.
    
    Args:
        recipients: List of email recipients
        subject: Email subject
        message: Email message
        
    Returns:
        True if the email was sent successfully, False otherwise
    """
    # This is a placeholder implementation
    # In a real-world scenario, you would use a proper email service
    # like SendGrid, AWS SES, or a corporate SMTP server
    
    logger.info(f"Email alert would be sent to {recipients}: {subject}")
    
    # Uncomment and configure this code to actually send emails
    """
    try:
        # Email server configuration
        smtp_server = os.environ.get("SMTP_SERVER", "smtp.gmail.com")
        smtp_port = int(os.environ.get("SMTP_PORT", 587))
        smtp_username = os.environ.get("SMTP_USERNAME", "")
        smtp_password = os.environ.get("SMTP_PASSWORD", "")
        
        # Create message
        msg = MIMEMultipart()
        msg["From"] = smtp_username
        msg["To"] = ", ".join(recipients)
        msg["Subject"] = subject
        
        # Add message body
        msg.attach(MIMEText(message, "plain"))
        
        # Connect to server and send email
        server = smtplib.SMTP(smtp_server, smtp_port)
        server.starttls()
        server.login(smtp_username, smtp_password)
        server.send_message(msg)
        server.quit()
        
        logger.info(f"Email alert sent to {recipients}")
        return True
    except Exception as e:
        logger.error(f"Error sending email alert: {e}")
        return False
    """
    
    return True

def send_slack_alert(webhook_url: str, channel: str, title: str, message: str) -> bool:
    """
    Send a Slack alert.
    
    Args:
        webhook_url: Slack webhook URL
        channel: Slack channel
        title: Alert title
        message: Alert message
        
    Returns:
        True if the alert was sent successfully, False otherwise
    """
    logger.info(f"Slack alert would be sent to {channel}: {title}")
    
    # Uncomment and configure this code to actually send Slack alerts
    """
    try:
        payload = {
            "channel": channel,
            "username": "MLOps Monitoring Bot",
            "text": f"*{title}*\n{message}",
            "icon_emoji": ":robot_face:"
        }
        
        response = requests.post(webhook_url, json=payload)
        if response.status_code == 200:
            logger.info(f"Slack alert sent to {channel}")
            return True
        else:
            logger.error(f"Error sending Slack alert: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error sending Slack alert: {e}")
        return False
    """
    
    return True

def send_azure_app_insights_alert(instrumentation_key: str, title: str, message: str, severity: str) -> bool:
    """
    Send an alert to Azure Application Insights.
    
    Args:
        instrumentation_key: Azure App Insights instrumentation key
        title: Alert title
        message: Alert message
        severity: Alert severity (verbose, information, warning, error, critical)
        
    Returns:
        True if the alert was sent successfully, False otherwise
    """
    logger.info(f"Azure App Insights alert would be sent: {title} ({severity})")
    
    # Uncomment and configure this code to actually send Azure App Insights alerts
    """
    try:
        # Map severity to App Insights severity level
        severity_map = {
            "verbose": 0,
            "information": 1,
            "warning": 2,
            "error": 3,
            "critical": 4
        }
        severity_level = severity_map.get(severity.lower(), 1)
        
        # Create telemetry payload
        payload = {
            "name": "Microsoft.ApplicationInsights.Event",
            "time": datetime.utcnow().isoformat() + "Z",
            "iKey": instrumentation_key,
            "tags": {
                "ai.cloud.role": "MLOps Monitoring",
                "ai.cloud.roleInstance": "Insurance Premium Model"
            },
            "data": {
                "baseType": "EventData",
                "baseData": {
                    "ver": 2,
                    "name": title,
                    "properties": {
                        "message": message,
                        "severity": severity
                    },
                    "measurements": {
                        "severityLevel": severity_level
                    }
                }
            }
        }
        
        # Send telemetry to App Insights
        response = requests.post(
            "https://dc.services.visualstudio.com/v2/track",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            logger.info(f"Azure App Insights alert sent: {title}")
            return True
        else:
            logger.error(f"Error sending Azure App Insights alert: {response.status_code} - {response.text}")
            return False
    except Exception as e:
        logger.error(f"Error sending Azure App Insights alert: {e}")
        return False
    """
    
    return True

def send_alert(title: str, message: str, severity: str = "warning") -> None:
    """
    Send an alert through all configured channels.
    
    Args:
        title: Alert title
        message: Alert message
        severity: Alert severity (warning, critical)
    """
    config = load_config()
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    full_message = f"[{timestamp}] {message}"
    
    # Log the alert
    if severity == "critical":
        logger.critical(f"{title}: {message}")
    else:
        logger.warning(f"{title}: {message}")
    
    # Send email alerts
    if config.get("email", {}).get("enabled", False):
        if severity in config["email"].get("severity_levels", []):
            recipients = config["email"].get("recipients", [])
            if recipients:
                send_email_alert(recipients, title, full_message)
    
    # Send Slack alerts
    if config.get("slack", {}).get("enabled", False):
        if severity in config["slack"].get("severity_levels", []):
            webhook_url = config["slack"].get("webhook_url", "")
            channel = config["slack"].get("channel", "#model-monitoring")
            if webhook_url:
                send_slack_alert(webhook_url, channel, title, full_message)
    
    # Send Azure App Insights alerts
    if config.get("azure_app_insights", {}).get("enabled", False):
        if severity in config["azure_app_insights"].get("severity_levels", []):
            instrumentation_key = config["azure_app_insights"].get("instrumentation_key", "")
            if instrumentation_key:
                # Replace environment variables
                if instrumentation_key.startswith("${") and instrumentation_key.endswith("}"):
                    env_var = instrumentation_key[2:-1]
                    instrumentation_key = os.environ.get(env_var, "")
                
                if instrumentation_key:
                    send_azure_app_insights_alert(instrumentation_key, title, full_message, severity)


if __name__ == "__main__":
    # Example usage
    send_alert(
        title="Test Alert",
        message="This is a test alert from the MLOps monitoring system.",
        severity="warning"
    )
