"""
Performance Tracker Module

This module tracks the performance of the Insurance Premium Prediction model over time.
It collects and stores metrics, generates visualizations, and provides insights into model performance.
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union
import json
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.InsurancePremiumPrediction import logger
from src.InsurancePremiumPrediction.utils.common import read_yaml, create_directories
from mlops.monitoring.alerting import send_alert

class PerformanceTracker:
    """
    Class for tracking model performance over time.
    """
    
    def __init__(self, config_path: str = "config/model_monitoring.yaml"):
        """
        Initialize the PerformanceTracker with configuration.
        
        Args:
            config_path: Path to the model monitoring configuration file
        """
        self.config = read_yaml(config_path)
        self.metrics_history = []
        self.predictions_log = []
        
        # Create directories for metrics and visualizations
        metrics_dir = Path("metrics")
        viz_dir = Path("reports/visualizations")
        create_directories([metrics_dir, viz_dir])
        self.metrics_dir = metrics_dir
        self.viz_dir = viz_dir
        
        # Set up logging
        logging.basicConfig(
            filename=os.path.join("logs", "performance_tracker.log"),
            level=logging.INFO,
            format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
            filemode="a"
        )
        self.logger = logging.getLogger(__name__)
        
        # Load metrics history if it exists
        self._load_metrics_history()
        self._load_predictions_log()
    
    def _load_metrics_history(self) -> None:
        """
        Load metrics history from file.
        """
        metrics_file = os.path.join(self.metrics_dir, "metrics_history.json")
        if os.path.exists(metrics_file):
            try:
                with open(metrics_file, "r") as f:
                    self.metrics_history = json.load(f)
                self.logger.info(f"Loaded metrics history with {len(self.metrics_history)} entries")
            except Exception as e:
                self.logger.error(f"Error loading metrics history: {e}")
                self.metrics_history = []
    
    def _save_metrics_history(self) -> None:
        """
        Save metrics history to file.
        """
        metrics_file = os.path.join(self.metrics_dir, "metrics_history.json")
        try:
            with open(metrics_file, "w") as f:
                json.dump(self.metrics_history, f, indent=2)
            self.logger.info(f"Saved metrics history with {len(self.metrics_history)} entries")
        except Exception as e:
            self.logger.error(f"Error saving metrics history: {e}")
    
    def _load_predictions_log(self) -> None:
        """
        Load predictions log from file.
        """
        predictions_file = os.path.join(self.metrics_dir, "predictions_log.json")
        if os.path.exists(predictions_file):
            try:
                with open(predictions_file, "r") as f:
                    self.predictions_log = json.load(f)
                self.logger.info(f"Loaded predictions log with {len(self.predictions_log)} entries")
            except Exception as e:
                self.logger.error(f"Error loading predictions log: {e}")
                self.predictions_log = []
    
    def _save_predictions_log(self) -> None:
        """
        Save predictions log to file.
        """
        predictions_file = os.path.join(self.metrics_dir, "predictions_log.json")
        try:
            with open(predictions_file, "w") as f:
                json.dump(self.predictions_log, f, indent=2)
            self.logger.info(f"Saved predictions log with {len(self.predictions_log)} entries")
        except Exception as e:
            self.logger.error(f"Error saving predictions log: {e}")
    
    def log_prediction(self, input_data: Dict, prediction: float, actual: Optional[float] = None) -> None:
        """
        Log a prediction for tracking.
        
        Args:
            input_data: Input data used for prediction
            prediction: Predicted value
            actual: Actual value (if available)
        """
        # Create a log entry
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "input_data": input_data,
            "prediction": float(prediction)
        }
        
        if actual is not None:
            log_entry["actual"] = float(actual)
            log_entry["error"] = float(actual - prediction)
            log_entry["absolute_error"] = float(abs(actual - prediction))
            log_entry["squared_error"] = float((actual - prediction) ** 2)
        
        # Add to predictions log
        self.predictions_log.append(log_entry)
        
        # Save periodically (every 100 predictions)
        if len(self.predictions_log) % 100 == 0:
            self._save_predictions_log()
            
        self.logger.info(f"Logged prediction: {prediction:.2f}" + (f", actual: {actual:.2f}" if actual is not None else ""))
    
    def compute_metrics(self, period: str = "day") -> Dict:
        """
        Compute performance metrics for a specific period.
        
        Args:
            period: Time period for metrics calculation ("day", "week", "month")
            
        Returns:
            Dictionary with computed metrics
        """
        if not self.predictions_log:
            self.logger.warning("No predictions available for metrics computation")
            return {}
        
        # Determine cutoff date based on period
        now = datetime.now()
        if period == "day":
            cutoff = now - timedelta(days=1)
        elif period == "week":
            cutoff = now - timedelta(weeks=1)
        elif period == "month":
            cutoff = now - timedelta(days=30)
        else:
            self.logger.error(f"Invalid period: {period}")
            return {}
        
        # Filter predictions within the period
        recent_predictions = [
            p for p in self.predictions_log 
            if "actual" in p and datetime.fromisoformat(p["timestamp"]) >= cutoff
        ]
        
        if not recent_predictions:
            self.logger.warning(f"No predictions with actual values available for the {period} period")
            return {}
        
        # Extract actual and predicted values
        actuals = [p["actual"] for p in recent_predictions]
        predictions = [p["prediction"] for p in recent_predictions]
        
        # Compute metrics
        rmse = np.sqrt(mean_squared_error(actuals, predictions))
        mae = mean_absolute_error(actuals, predictions)
        r2 = r2_score(actuals, predictions)
        
        # Compute additional metrics
        errors = [p["error"] for p in recent_predictions]
        mean_error = np.mean(errors)
        median_error = np.median(errors)
        error_std = np.std(errors)
        
        # Create metrics dictionary
        metrics = {
            "timestamp": now.isoformat(),
            "period": period,
            "sample_size": len(recent_predictions),
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "mean_error": mean_error,
            "median_error": median_error,
            "error_std": error_std
        }
        
        self.logger.info(f"Computed {period} metrics: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
        return metrics
    
    def update_metrics(self) -> Dict:
        """
        Update metrics for all periods and save to history.
        
        Returns:
            Dictionary with all computed metrics
        """
        all_metrics = {}
        
        for period in ["day", "week", "month"]:
            metrics = self.compute_metrics(period)
            if metrics:
                all_metrics[period] = metrics
        
        if all_metrics:
            # Add to metrics history
            entry = {
                "timestamp": datetime.now().isoformat(),
                "metrics": all_metrics
            }
            self.metrics_history.append(entry)
            self._save_metrics_history()
            
            # Check for performance degradation
            self._check_performance_degradation(all_metrics)
        
        return all_metrics
    
    def _check_performance_degradation(self, metrics: Dict) -> None:
        """
        Check for performance degradation and send alerts if necessary.
        
        Args:
            metrics: Dictionary with computed metrics
        """
        if not self.metrics_history or len(self.metrics_history) < 2:
            return
        
        # Get previous metrics
        prev_entry = self.metrics_history[-2]
        
        # Check each period
        for period, current in metrics.items():
            if period in prev_entry["metrics"]:
                prev = prev_entry["metrics"][period]
                
                # Check RMSE
                rmse_increase = (current["rmse"] - prev["rmse"]) / prev["rmse"] if prev["rmse"] > 0 else 0
                
                # Check MAE
                mae_increase = (current["mae"] - prev["mae"]) / prev["mae"] if prev["mae"] > 0 else 0
                
                # Check R²
                r2_decrease = (prev["r2"] - current["r2"]) / prev["r2"] if prev["r2"] > 0 else 0
                
                # Get thresholds from config
                thresholds = self.config["retraining_triggers"]["performance_thresholds"]
                rmse_threshold = thresholds["rmse_increase"] / 100
                mae_threshold = thresholds["mae_increase"] / 100
                r2_threshold = thresholds["r2_decrease"] / 100
                
                # Check for degradation
                degradation_reasons = []
                
                if rmse_increase > rmse_threshold:
                    degradation_reasons.append(f"RMSE increased by {rmse_increase:.2%} (threshold: {rmse_threshold:.2%})")
                
                if mae_increase > mae_threshold:
                    degradation_reasons.append(f"MAE increased by {mae_increase:.2%} (threshold: {mae_threshold:.2%})")
                
                if r2_decrease > r2_threshold:
                    degradation_reasons.append(f"R² decreased by {r2_decrease:.2%} (threshold: {r2_threshold:.2%})")
                
                if degradation_reasons:
                    self.logger.warning(f"Performance degradation detected for {period} period: {', '.join(degradation_reasons)}")
                    send_alert(
                        title=f"Performance Degradation ({period})",
                        message=f"Performance degradation detected: {', '.join(degradation_reasons)}",
                        severity="warning"
                    )
    
    def generate_performance_visualizations(self) -> List[str]:
        """
        Generate visualizations of model performance over time.
        
        Returns:
            List of paths to generated visualization files
        """
        if not self.metrics_history:
            self.logger.warning("No metrics history available for visualization")
            return []
        
        # Convert metrics history to DataFrame
        data = []
        for entry in self.metrics_history:
            timestamp = datetime.fromisoformat(entry["timestamp"])
            for period, metrics in entry["metrics"].items():
                data.append({
                    "timestamp": timestamp,
                    "period": period,
                    "rmse": metrics["rmse"],
                    "mae": metrics["mae"],
                    "r2": metrics["r2"],
                    "sample_size": metrics["sample_size"]
                })
        
        df = pd.DataFrame(data)
        
        # Generate visualizations
        viz_files = []
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # RMSE over time
        plt.figure(figsize=(12, 6))
        for period in df["period"].unique():
            period_df = df[df["period"] == period]
            plt.plot(period_df["timestamp"], period_df["rmse"], marker="o", label=period)
        
        plt.title("RMSE Over Time")
        plt.xlabel("Date")
        plt.ylabel("RMSE")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        rmse_file = os.path.join(self.viz_dir, f"rmse_over_time_{timestamp_str}.png")
        plt.savefig(rmse_file)
        plt.close()
        viz_files.append(rmse_file)
        
        # MAE over time
        plt.figure(figsize=(12, 6))
        for period in df["period"].unique():
            period_df = df[df["period"] == period]
            plt.plot(period_df["timestamp"], period_df["mae"], marker="o", label=period)
        
        plt.title("MAE Over Time")
        plt.xlabel("Date")
        plt.ylabel("MAE")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        mae_file = os.path.join(self.viz_dir, f"mae_over_time_{timestamp_str}.png")
        plt.savefig(mae_file)
        plt.close()
        viz_files.append(mae_file)
        
        # R² over time
        plt.figure(figsize=(12, 6))
        for period in df["period"].unique():
            period_df = df[df["period"] == period]
            plt.plot(period_df["timestamp"], period_df["r2"], marker="o", label=period)
        
        plt.title("R² Over Time")
        plt.xlabel("Date")
        plt.ylabel("R²")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        r2_file = os.path.join(self.viz_dir, f"r2_over_time_{timestamp_str}.png")
        plt.savefig(r2_file)
        plt.close()
        viz_files.append(r2_file)
        
        # Error distribution
        if self.predictions_log and any("error" in p for p in self.predictions_log):
            errors = [p["error"] for p in self.predictions_log if "error" in p]
            
            plt.figure(figsize=(12, 6))
            sns.histplot(errors, kde=True)
            plt.title("Error Distribution")
            plt.xlabel("Error (Actual - Predicted)")
            plt.ylabel("Frequency")
            plt.grid(True)
            plt.tight_layout()
            
            error_file = os.path.join(self.viz_dir, f"error_distribution_{timestamp_str}.png")
            plt.savefig(error_file)
            plt.close()
            viz_files.append(error_file)
        
        self.logger.info(f"Generated {len(viz_files)} performance visualizations")
        return viz_files


if __name__ == "__main__":
    # Example usage
    tracker = PerformanceTracker()
    
    # Log some predictions (in a real scenario, these would come from the model)
    for i in range(10):
        input_data = {"feature1": i, "feature2": i*2}
        prediction = i * 1.5
        actual = i * 1.5 + np.random.normal(0, 0.5)
        tracker.log_prediction(input_data, prediction, actual)
    
    # Update metrics
    metrics = tracker.update_metrics()
    print(f"Updated metrics: {metrics}")
    
    # Generate visualizations
    viz_files = tracker.generate_performance_visualizations()
    print(f"Generated visualizations: {viz_files}")
