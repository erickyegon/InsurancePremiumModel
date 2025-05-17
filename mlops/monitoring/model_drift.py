"""
Model Drift Detection Module

This module implements model performance monitoring and drift detection for the Insurance Premium Prediction model.
It tracks model performance metrics over time and detects when performance degrades.
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
import yaml
import logging
import joblib
import json
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.InsurancePremiumPrediction import logger
from src.InsurancePremiumPrediction.utils.common import read_yaml, create_directories
from mlops.monitoring.alerting import send_alert

class ModelDriftDetector:
    """
    Class for detecting model performance drift in the Insurance Premium Prediction model.
    """
    
    def __init__(self, config_path: str = "config/model_monitoring.yaml"):
        """
        Initialize the ModelDriftDetector with configuration.
        
        Args:
            config_path: Path to the model monitoring configuration file
        """
        self.config = read_yaml(config_path)
        self.model = None
        self.performance_history = []
        self.baseline_metrics = None
        
        # Create directories for reports and metrics history
        report_dir = Path("reports/model_drift")
        metrics_dir = Path("metrics")
        create_directories([report_dir, metrics_dir])
        self.report_dir = report_dir
        self.metrics_dir = metrics_dir
        
        # Set up logging
        logging.basicConfig(
            filename=os.path.join("logs", "model_drift.log"),
            level=logging.INFO,
            format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
            filemode="a"
        )
        self.logger = logging.getLogger(__name__)
        
        # Load performance history if it exists
        self._load_performance_history()
    
    def load_model(self, model_path: str) -> None:
        """
        Load the model for evaluation.
        
        Args:
            model_path: Path to the model file
        """
        try:
            self.model = joblib.load(model_path)
            self.logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            self.logger.error(f"Error loading model: {e}")
            raise
    
    def set_baseline_metrics(self, metrics: Dict[str, float]) -> None:
        """
        Set baseline metrics for comparison.
        
        Args:
            metrics: Dictionary of baseline metrics (rmse, mae, r2)
        """
        self.baseline_metrics = metrics
        self.logger.info(f"Baseline metrics set: {metrics}")
    
    def load_baseline_metrics(self, metrics_path: str) -> None:
        """
        Load baseline metrics from a file.
        
        Args:
            metrics_path: Path to the metrics file
        """
        try:
            with open(metrics_path, 'r') as f:
                self.baseline_metrics = json.load(f)
            self.logger.info(f"Baseline metrics loaded from {metrics_path}")
        except Exception as e:
            self.logger.error(f"Error loading baseline metrics: {e}")
            raise
    
    def _load_performance_history(self) -> None:
        """
        Load performance history from the metrics directory.
        """
        history_path = os.path.join(self.metrics_dir, "performance_history.json")
        if os.path.exists(history_path):
            try:
                with open(history_path, 'r') as f:
                    self.performance_history = json.load(f)
                self.logger.info(f"Performance history loaded with {len(self.performance_history)} entries")
            except Exception as e:
                self.logger.error(f"Error loading performance history: {e}")
                self.performance_history = []
    
    def _save_performance_history(self) -> None:
        """
        Save performance history to the metrics directory.
        """
        history_path = os.path.join(self.metrics_dir, "performance_history.json")
        try:
            with open(history_path, 'w') as f:
                json.dump(self.performance_history, f, indent=2)
            self.logger.info(f"Performance history saved with {len(self.performance_history)} entries")
        except Exception as e:
            self.logger.error(f"Error saving performance history: {e}")
    
    def evaluate_model(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate the model on the given data.
        
        Args:
            X: Features
            y: Target values
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model must be loaded before evaluation")
        
        # Make predictions
        y_pred = self.model.predict(X)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # Create metrics dictionary
        metrics = {
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "sample_size": len(X),
            "timestamp": datetime.now().isoformat()
        }
        
        self.logger.info(f"Model evaluated on {len(X)} samples: RMSE={rmse:.4f}, MAE={mae:.4f}, R²={r2:.4f}")
        return metrics
    
    def detect_performance_drift(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Detect model performance drift by comparing current metrics with baseline.
        
        Args:
            X: Features
            y: Target values
            
        Returns:
            Dictionary with drift detection results
        """
        # Check if we have enough samples
        min_samples = self.config["performance_monitoring"]["min_samples"]
        if len(X) < min_samples:
            self.logger.warning(f"Not enough samples for performance evaluation: {len(X)} < {min_samples}")
            return {
                "drift_detected": False,
                "reason": f"Not enough samples for evaluation: {len(X)} < {min_samples}",
                "metrics": None,
                "comparison": None
            }
        
        # Evaluate model on current data
        current_metrics = self.evaluate_model(X, y)
        
        # If no baseline metrics, set current as baseline
        if self.baseline_metrics is None:
            self.logger.info("No baseline metrics available, setting current metrics as baseline")
            self.baseline_metrics = {k: v for k, v in current_metrics.items() if k in ["rmse", "mae", "r2"]}
            return {
                "drift_detected": False,
                "reason": "Baseline metrics established",
                "metrics": current_metrics,
                "comparison": None
            }
        
        # Compare with baseline
        comparison = {}
        drift_detected = False
        drift_reasons = []
        
        # Get thresholds from config
        thresholds = {}
        for metric in self.config["performance_monitoring"]["metrics"]:
            thresholds[metric["name"]] = metric["threshold"]
        
        # Check RMSE
        rmse_increase = (current_metrics["rmse"] - self.baseline_metrics["rmse"]) / self.baseline_metrics["rmse"]
        comparison["rmse_increase"] = rmse_increase
        comparison["rmse_threshold"] = self.config["retraining_triggers"]["performance_thresholds"]["rmse_increase"] / 100
        
        if rmse_increase > comparison["rmse_threshold"]:
            drift_detected = True
            drift_reasons.append(f"RMSE increased by {rmse_increase:.2%} (threshold: {comparison['rmse_threshold']:.2%})")
        
        # Check MAE
        mae_increase = (current_metrics["mae"] - self.baseline_metrics["mae"]) / self.baseline_metrics["mae"]
        comparison["mae_increase"] = mae_increase
        comparison["mae_threshold"] = self.config["retraining_triggers"]["performance_thresholds"]["mae_increase"] / 100
        
        if mae_increase > comparison["mae_threshold"]:
            drift_detected = True
            drift_reasons.append(f"MAE increased by {mae_increase:.2%} (threshold: {comparison['mae_threshold']:.2%})")
        
        # Check R²
        r2_decrease = (self.baseline_metrics["r2"] - current_metrics["r2"]) / self.baseline_metrics["r2"]
        comparison["r2_decrease"] = r2_decrease
        comparison["r2_threshold"] = self.config["retraining_triggers"]["performance_thresholds"]["r2_decrease"] / 100
        
        if r2_decrease > comparison["r2_threshold"]:
            drift_detected = True
            drift_reasons.append(f"R² decreased by {r2_decrease:.2%} (threshold: {comparison['r2_threshold']:.2%})")
        
        # Check absolute thresholds
        for metric_name in ["rmse", "mae", "r2"]:
            if metric_name in thresholds:
                threshold = thresholds[metric_name]
                comparison[f"{metric_name}_absolute_threshold"] = threshold
                
                # For R², lower is worse; for RMSE and MAE, higher is worse
                if metric_name == "r2":
                    if current_metrics[metric_name] < threshold:
                        drift_detected = True
                        drift_reasons.append(f"{metric_name.upper()} ({current_metrics[metric_name]:.4f}) below absolute threshold ({threshold:.4f})")
                else:
                    if current_metrics[metric_name] > threshold:
                        drift_detected = True
                        drift_reasons.append(f"{metric_name.upper()} ({current_metrics[metric_name]:.4f}) above absolute threshold ({threshold:.4f})")
        
        # Add metrics to performance history
        history_entry = {
            **current_metrics,
            "drift_detected": drift_detected,
            "drift_reasons": drift_reasons if drift_detected else []
        }
        self.performance_history.append(history_entry)
        self._save_performance_history()
        
        # Log results
        if drift_detected:
            self.logger.warning(f"Model performance drift detected: {', '.join(drift_reasons)}")
            send_alert(
                title="Model Performance Drift Detected",
                message=f"Model performance drift detected: {', '.join(drift_reasons)}",
                severity="warning"
            )
        else:
            self.logger.info("No significant model performance drift detected")
        
        return {
            "drift_detected": drift_detected,
            "reason": "; ".join(drift_reasons) if drift_detected else "No significant drift detected",
            "metrics": current_metrics,
            "comparison": comparison
        }
    
    def should_retrain(self) -> Tuple[bool, str]:
        """
        Determine if model retraining is needed based on performance drift.
        
        Returns:
            Tuple of (should_retrain, reason)
        """
        if not self.performance_history:
            return False, "No performance history available"
        
        # Get the latest performance entry
        latest = self.performance_history[-1]
        
        # Check if drift was detected
        if latest.get("drift_detected", False):
            return True, f"Performance drift detected: {'; '.join(latest.get('drift_reasons', []))}"
        
        # Check time-based trigger
        if len(self.performance_history) >= 2:
            first_timestamp = datetime.fromisoformat(self.performance_history[0]["timestamp"])
            latest_timestamp = datetime.fromisoformat(latest["timestamp"])
            days_since_first = (latest_timestamp - first_timestamp).days
            
            time_trigger = self.config["retraining_triggers"]["time_based_trigger"]
            if days_since_first >= time_trigger:
                return True, f"Time-based trigger: {days_since_first} days since first evaluation (threshold: {time_trigger} days)"
        
        return False, "No retraining triggers activated"
    
    def generate_performance_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate a report of the model performance history.
        
        Args:
            output_path: Path to save the report. If None, use the default path.
            
        Returns:
            Path to the generated report
        """
        if not self.performance_history:
            raise ValueError("No performance history available")
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.report_dir, f"performance_report_{timestamp}.html")
        
        # Create report directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert performance history to DataFrame for easier plotting
        history_df = pd.DataFrame(self.performance_history)
        history_df["timestamp"] = pd.to_datetime(history_df["timestamp"])
        
        # Create plots
        fig, axes = plt.subplots(3, 1, figsize=(10, 15))
        
        # RMSE plot
        history_df.plot(x="timestamp", y="rmse", ax=axes[0], marker="o")
        if self.baseline_metrics:
            axes[0].axhline(y=self.baseline_metrics["rmse"], color="r", linestyle="--", label="Baseline")
        axes[0].set_title("RMSE over time")
        axes[0].set_ylabel("RMSE")
        axes[0].grid(True)
        
        # MAE plot
        history_df.plot(x="timestamp", y="mae", ax=axes[1], marker="o")
        if self.baseline_metrics:
            axes[1].axhline(y=self.baseline_metrics["mae"], color="r", linestyle="--", label="Baseline")
        axes[1].set_title("MAE over time")
        axes[1].set_ylabel("MAE")
        axes[1].grid(True)
        
        # R² plot
        history_df.plot(x="timestamp", y="r2", ax=axes[2], marker="o")
        if self.baseline_metrics:
            axes[2].axhline(y=self.baseline_metrics["r2"], color="r", linestyle="--", label="Baseline")
        axes[2].set_title("R² over time")
        axes[2].set_ylabel("R²")
        axes[2].grid(True)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(self.report_dir, f"performance_plot_{timestamp}.png")
        plt.savefig(plot_path)
        plt.close()
        
        # Generate HTML report
        with open(output_path, "w") as f:
            f.write("<html><head><title>Model Performance Report</title>")
            f.write("<style>body{font-family:Arial,sans-serif;margin:20px;}")
            f.write("table{border-collapse:collapse;width:100%;}")
            f.write("th,td{border:1px solid #ddd;padding:8px;text-align:left;}")
            f.write("th{background-color:#f2f2f2;}")
            f.write(".drift{color:red;font-weight:bold;}")
            f.write("</style></head><body>")
            
            # Header
            f.write(f"<h1>Model Performance Report</h1>")
            f.write(f"<p>Generated on: {datetime.now().isoformat()}</p>")
            
            # Performance plot
            f.write("<h2>Performance Metrics Over Time</h2>")
            f.write(f"<img src='{os.path.relpath(plot_path, os.path.dirname(output_path))}' alt='Performance Plot' style='width:100%;max-width:800px;'>")
            
            # Latest metrics
            latest = self.performance_history[-1]
            f.write("<h2>Latest Performance Metrics</h2>")
            f.write("<table>")
            f.write("<tr><th>Metric</th><th>Value</th><th>Baseline</th><th>Change</th></tr>")
            
            for metric in ["rmse", "mae", "r2"]:
                current = latest[metric]
                baseline = self.baseline_metrics[metric] if self.baseline_metrics else "N/A"
                
                if self.baseline_metrics:
                    if metric == "r2":
                        change = (baseline - current) / baseline if baseline != 0 else float("inf")
                        change_text = f"{change:.2%} decrease" if change > 0 else f"{-change:.2%} increase"
                    else:
                        change = (current - baseline) / baseline if baseline != 0 else float("inf")
                        change_text = f"{change:.2%} increase" if change > 0 else f"{-change:.2%} decrease"
                else:
                    change_text = "N/A"
                
                drift_class = "drift" if latest.get("drift_detected", False) and any(metric.upper() in reason for reason in latest.get("drift_reasons", [])) else ""
                
                f.write("<tr>")
                f.write(f"<td>{metric.upper()}</td>")
                f.write(f"<td>{current:.4f}</td>")
                f.write(f"<td>{baseline:.4f if isinstance(baseline, float) else baseline}</td>")
                f.write(f"<td class='{drift_class}'>{change_text}</td>")
                f.write("</tr>")
            
            f.write("</table>")
            
            # Drift status
            f.write("<h2>Drift Status</h2>")
            if latest.get("drift_detected", False):
                f.write("<p class='drift'>Performance drift detected:</p>")
                f.write("<ul>")
                for reason in latest.get("drift_reasons", []):
                    f.write(f"<li class='drift'>{reason}</li>")
                f.write("</ul>")
            else:
                f.write("<p>No significant performance drift detected.</p>")
            
            # Retraining recommendation
            should_retrain, reason = self.should_retrain()
            f.write("<h2>Retraining Recommendation</h2>")
            if should_retrain:
                f.write(f"<p class='drift'>Model retraining recommended: {reason}</p>")
            else:
                f.write(f"<p>Model retraining not needed: {reason}</p>")
            
            # Performance history table
            f.write("<h2>Performance History</h2>")
            f.write("<table>")
            f.write("<tr><th>Timestamp</th><th>RMSE</th><th>MAE</th><th>R²</th><th>Sample Size</th><th>Drift</th></tr>")
            
            for entry in reversed(self.performance_history):
                drift_class = "drift" if entry.get("drift_detected", False) else ""
                drift_text = "Yes" if entry.get("drift_detected", False) else "No"
                
                f.write("<tr>")
                f.write(f"<td>{entry['timestamp']}</td>")
                f.write(f"<td>{entry['rmse']:.4f}</td>")
                f.write(f"<td>{entry['mae']:.4f}</td>")
                f.write(f"<td>{entry['r2']:.4f}</td>")
                f.write(f"<td>{entry['sample_size']}</td>")
                f.write(f"<td class='{drift_class}'>{drift_text}</td>")
                f.write("</tr>")
            
            f.write("</table>")
            
            f.write("</body></html>")
        
        self.logger.info(f"Performance report generated at {output_path}")
        return output_path


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    # Load model and data
    detector = ModelDriftDetector()
    detector.load_model("artifacts/model_trainer/model.joblib")
    
    # Load baseline metrics
    detector.load_baseline_metrics("metrics/baseline_metrics.json")
    
    # Load evaluation data
    data = pd.read_csv("data/test_data.csv")
    X = data.drop("Annual_Premium_Amount", axis=1)
    y = data["Annual_Premium_Amount"]
    
    # Detect performance drift
    results = detector.detect_performance_drift(X, y)
    
    # Generate report
    report_path = detector.generate_performance_report()
    print(f"Performance report generated at: {report_path}")
    
    # Check if retraining is needed
    should_retrain, reason = detector.should_retrain()
    if should_retrain:
        print(f"Model retraining recommended: {reason}")
    else:
        print(f"Model retraining not needed: {reason}")
