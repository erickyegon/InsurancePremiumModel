"""
Data Drift Detection Module

This module implements data drift detection for the Insurance Premium Prediction model.
It compares current data distributions with a reference dataset to detect drift.
"""
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Union
import yaml
import logging
from scipy import stats
from sklearn.metrics import mutual_info_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.InsurancePremiumPrediction import logger
from src.InsurancePremiumPrediction.utils.common import read_yaml, create_directories
from mlops.monitoring.alerting import send_alert

class DataDriftDetector:
    """
    Class for detecting data drift in the Insurance Premium Prediction model.
    """
    
    def __init__(self, config_path: str = "config/model_monitoring.yaml"):
        """
        Initialize the DataDriftDetector with configuration.
        
        Args:
            config_path: Path to the model monitoring configuration file
        """
        self.config = read_yaml(config_path)
        self.reference_data = None
        self.current_data = None
        self.drift_results = {}
        self.drift_detected = False
        
        # Create directories for reports
        report_dir = Path("reports/data_drift")
        create_directories([report_dir])
        self.report_dir = report_dir
        
        # Set up logging
        logging.basicConfig(
            filename=os.path.join("logs", "data_drift.log"),
            level=logging.INFO,
            format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
            filemode="a"
        )
        self.logger = logging.getLogger(__name__)
    
    def load_reference_data(self, reference_path: Optional[str] = None) -> None:
        """
        Load the reference dataset.
        
        Args:
            reference_path: Path to the reference dataset. If None, use the path from config.
        """
        if reference_path is None:
            reference_path = self.config["data_drift"]["reference_dataset"]
        
        try:
            self.reference_data = pd.read_csv(reference_path)
            self.logger.info(f"Reference data loaded from {reference_path} with shape {self.reference_data.shape}")
        except Exception as e:
            self.logger.error(f"Error loading reference data: {e}")
            raise
    
    def load_current_data(self, current_path: str) -> None:
        """
        Load the current dataset for drift detection.
        
        Args:
            current_path: Path to the current dataset
        """
        try:
            self.current_data = pd.read_csv(current_path)
            self.logger.info(f"Current data loaded from {current_path} with shape {self.current_data.shape}")
        except Exception as e:
            self.logger.error(f"Error loading current data: {e}")
            raise
    
    def detect_drift_numerical(self, feature: str) -> Dict:
        """
        Detect drift in numerical features using statistical tests.
        
        Args:
            feature: Name of the numerical feature
            
        Returns:
            Dictionary with drift detection results
        """
        if self.reference_data is None or self.current_data is None:
            raise ValueError("Reference and current data must be loaded before drift detection")
        
        ref_values = self.reference_data[feature].dropna().values
        curr_values = self.current_data[feature].dropna().values
        
        # Kolmogorov-Smirnov test
        ks_statistic, ks_pvalue = stats.ks_2samp(ref_values, curr_values)
        
        # Calculate mean and std differences
        ref_mean, curr_mean = np.mean(ref_values), np.mean(curr_values)
        ref_std, curr_std = np.std(ref_values), np.std(curr_values)
        mean_diff_pct = abs(ref_mean - curr_mean) / (abs(ref_mean) if abs(ref_mean) > 0 else 1)
        std_diff_pct = abs(ref_std - curr_std) / (abs(ref_std) if abs(ref_std) > 0 else 1)
        
        # Get threshold from config
        threshold = self.config["data_drift"]["drift_thresholds"]["numerical_features"].get(
            feature, 0.1  # Default threshold if not specified
        )
        
        # Determine if drift is detected
        is_drift = ks_pvalue < 0.05 and ks_statistic > threshold
        
        return {
            "feature": feature,
            "type": "numerical",
            "ks_statistic": ks_statistic,
            "ks_pvalue": ks_pvalue,
            "ref_mean": ref_mean,
            "curr_mean": curr_mean,
            "mean_diff_pct": mean_diff_pct,
            "ref_std": ref_std,
            "curr_std": curr_std,
            "std_diff_pct": std_diff_pct,
            "threshold": threshold,
            "drift_detected": is_drift,
            "severity": "high" if is_drift and ks_statistic > threshold * 2 else "medium" if is_drift else "low"
        }
    
    def detect_drift_categorical(self, feature: str) -> Dict:
        """
        Detect drift in categorical features using chi-squared test.
        
        Args:
            feature: Name of the categorical feature
            
        Returns:
            Dictionary with drift detection results
        """
        if self.reference_data is None or self.current_data is None:
            raise ValueError("Reference and current data must be loaded before drift detection")
        
        # Get unique categories from both datasets
        ref_counts = self.reference_data[feature].value_counts(normalize=True)
        curr_counts = self.current_data[feature].value_counts(normalize=True)
        
        # Align categories
        all_categories = sorted(set(ref_counts.index) | set(curr_counts.index))
        ref_dist = np.array([ref_counts.get(cat, 0) for cat in all_categories])
        curr_dist = np.array([curr_counts.get(cat, 0) for cat in all_categories])
        
        # Calculate Jensen-Shannon divergence (symmetric KL divergence)
        js_divergence = stats.entropy(ref_dist, curr_dist) / 2 + stats.entropy(curr_dist, ref_dist) / 2
        
        # Chi-squared test if enough data
        if len(all_categories) > 1 and min(len(self.reference_data), len(self.current_data)) > 5 * len(all_categories):
            # Create contingency table
            ref_abs_counts = self.reference_data[feature].value_counts()
            curr_abs_counts = self.current_data[feature].value_counts()
            
            # Align categories
            contingency = np.array([
                [ref_abs_counts.get(cat, 0) for cat in all_categories],
                [curr_abs_counts.get(cat, 0) for cat in all_categories]
            ])
            
            chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
            chi2_result = {"chi2": chi2, "p_value": p_value, "dof": dof}
        else:
            chi2_result = {"chi2": None, "p_value": None, "dof": None}
        
        # Get threshold from config
        threshold = self.config["data_drift"]["drift_thresholds"]["categorical_features"].get(
            feature, 0.1  # Default threshold if not specified
        )
        
        # Determine if drift is detected
        is_drift = js_divergence > threshold
        
        return {
            "feature": feature,
            "type": "categorical",
            "js_divergence": js_divergence,
            "chi2_results": chi2_result,
            "ref_distribution": dict(ref_counts),
            "curr_distribution": dict(curr_counts),
            "threshold": threshold,
            "drift_detected": is_drift,
            "severity": "high" if is_drift and js_divergence > threshold * 2 else "medium" if is_drift else "low"
        }
    
    def detect_drift(self, current_data_path: str, numerical_features: List[str], categorical_features: List[str]) -> Dict:
        """
        Detect drift across all features.
        
        Args:
            current_data_path: Path to the current dataset
            numerical_features: List of numerical features
            categorical_features: List of categorical features
            
        Returns:
            Dictionary with drift detection results for all features
        """
        # Load data
        if self.reference_data is None:
            self.load_reference_data()
        self.load_current_data(current_data_path)
        
        # Detect drift for each feature
        results = {}
        drift_count = 0
        total_features = len(numerical_features) + len(categorical_features)
        
        # Numerical features
        for feature in numerical_features:
            if feature in self.reference_data.columns and feature in self.current_data.columns:
                results[feature] = self.detect_drift_numerical(feature)
                if results[feature]["drift_detected"]:
                    drift_count += 1
                    self.logger.warning(f"Drift detected in numerical feature {feature} with statistic {results[feature]['ks_statistic']:.4f}")
            else:
                self.logger.warning(f"Feature {feature} not found in both datasets")
        
        # Categorical features
        for feature in categorical_features:
            if feature in self.reference_data.columns and feature in self.current_data.columns:
                results[feature] = self.detect_drift_categorical(feature)
                if results[feature]["drift_detected"]:
                    drift_count += 1
                    self.logger.warning(f"Drift detected in categorical feature {feature} with JS divergence {results[feature]['js_divergence']:.4f}")
            else:
                self.logger.warning(f"Feature {feature} not found in both datasets")
        
        # Calculate overall drift
        drift_percentage = drift_count / total_features if total_features > 0 else 0
        overall_threshold = self.config["data_drift"]["overall_drift_threshold"]
        overall_drift_detected = drift_percentage > overall_threshold
        
        # Store results
        self.drift_results = {
            "timestamp": datetime.now().isoformat(),
            "feature_results": results,
            "drift_count": drift_count,
            "total_features": total_features,
            "drift_percentage": drift_percentage,
            "overall_threshold": overall_threshold,
            "overall_drift_detected": overall_drift_detected
        }
        
        self.drift_detected = overall_drift_detected
        
        if overall_drift_detected:
            self.logger.warning(f"Overall drift detected: {drift_percentage:.2%} of features have drifted (threshold: {overall_threshold:.2%})")
            # Check if retraining should be triggered
            if drift_percentage > self.config["retraining_triggers"]["data_drift_threshold"]:
                self.logger.critical(f"Data drift exceeds retraining threshold: {drift_percentage:.2%} > {self.config['retraining_triggers']['data_drift_threshold']:.2%}")
                send_alert(
                    title="Model Retraining Required",
                    message=f"Data drift exceeds retraining threshold: {drift_percentage:.2%} > {self.config['retraining_triggers']['data_drift_threshold']:.2%}",
                    severity="critical"
                )
        else:
            self.logger.info(f"No significant overall drift detected: {drift_percentage:.2%} of features have drifted (threshold: {overall_threshold:.2%})")
        
        return self.drift_results
    
    def generate_drift_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate a report of the drift detection results.
        
        Args:
            output_path: Path to save the report. If None, use the default path.
            
        Returns:
            Path to the generated report
        """
        if not self.drift_results:
            raise ValueError("Drift detection must be run before generating a report")
        
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(self.report_dir, f"drift_report_{timestamp}.html")
        
        # Create report directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Generate HTML report
        with open(output_path, "w") as f:
            f.write("<html><head><title>Data Drift Report</title>")
            f.write("<style>body{font-family:Arial,sans-serif;margin:20px;}")
            f.write("table{border-collapse:collapse;width:100%;}")
            f.write("th,td{border:1px solid #ddd;padding:8px;text-align:left;}")
            f.write("th{background-color:#f2f2f2;}")
            f.write(".high{color:red;font-weight:bold;}")
            f.write(".medium{color:orange;}")
            f.write(".low{color:green;}")
            f.write("</style></head><body>")
            
            # Header
            f.write(f"<h1>Data Drift Report</h1>")
            f.write(f"<p>Generated on: {self.drift_results['timestamp']}</p>")
            
            # Overall summary
            f.write("<h2>Overall Drift Summary</h2>")
            f.write("<table>")
            f.write("<tr><th>Metric</th><th>Value</th></tr>")
            f.write(f"<tr><td>Features with drift</td><td>{self.drift_results['drift_count']} / {self.drift_results['total_features']}</td></tr>")
            f.write(f"<tr><td>Drift percentage</td><td>{self.drift_results['drift_percentage']:.2%}</td></tr>")
            f.write(f"<tr><td>Drift threshold</td><td>{self.drift_results['overall_threshold']:.2%}</td></tr>")
            
            drift_class = "high" if self.drift_results['overall_drift_detected'] else "low"
            drift_text = "Yes" if self.drift_results['overall_drift_detected'] else "No"
            f.write(f"<tr><td>Overall drift detected</td><td class='{drift_class}'>{drift_text}</td></tr>")
            f.write("</table>")
            
            # Feature details
            f.write("<h2>Feature Drift Details</h2>")
            
            # Numerical features
            numerical_features = [f for f, r in self.drift_results["feature_results"].items() if r["type"] == "numerical"]
            if numerical_features:
                f.write("<h3>Numerical Features</h3>")
                f.write("<table>")
                f.write("<tr><th>Feature</th><th>KS Statistic</th><th>p-value</th><th>Mean Diff %</th><th>Std Diff %</th><th>Threshold</th><th>Drift</th></tr>")
                
                for feature in numerical_features:
                    result = self.drift_results["feature_results"][feature]
                    drift_class = result["severity"]
                    drift_text = "Yes" if result["drift_detected"] else "No"
                    
                    f.write("<tr>")
                    f.write(f"<td>{feature}</td>")
                    f.write(f"<td>{result['ks_statistic']:.4f}</td>")
                    f.write(f"<td>{result['ks_pvalue']:.4f}</td>")
                    f.write(f"<td>{result['mean_diff_pct']:.2%}</td>")
                    f.write(f"<td>{result['std_diff_pct']:.2%}</td>")
                    f.write(f"<td>{result['threshold']:.2f}</td>")
                    f.write(f"<td class='{drift_class}'>{drift_text}</td>")
                    f.write("</tr>")
                
                f.write("</table>")
            
            # Categorical features
            categorical_features = [f for f, r in self.drift_results["feature_results"].items() if r["type"] == "categorical"]
            if categorical_features:
                f.write("<h3>Categorical Features</h3>")
                f.write("<table>")
                f.write("<tr><th>Feature</th><th>JS Divergence</th><th>Chi² p-value</th><th>Threshold</th><th>Drift</th></tr>")
                
                for feature in categorical_features:
                    result = self.drift_results["feature_results"][feature]
                    drift_class = result["severity"]
                    drift_text = "Yes" if result["drift_detected"] else "No"
                    chi2_pvalue = result["chi2_results"]["p_value"]
                    
                    f.write("<tr>")
                    f.write(f"<td>{feature}</td>")
                    f.write(f"<td>{result['js_divergence']:.4f}</td>")
                    f.write(f"<td>{chi2_pvalue:.4f if chi2_pvalue is not None else 'N/A'}</td>")
                    f.write(f"<td>{result['threshold']:.2f}</td>")
                    f.write(f"<td class='{drift_class}'>{drift_text}</td>")
                    f.write("</tr>")
                
                f.write("</table>")
            
            f.write("</body></html>")
        
        self.logger.info(f"Drift report generated at {output_path}")
        return output_path
    
    def should_retrain(self) -> Tuple[bool, str]:
        """
        Determine if model retraining is needed based on drift results.
        
        Returns:
            Tuple of (should_retrain, reason)
        """
        if not self.drift_results:
            return False, "No drift detection results available"
        
        # Check if drift percentage exceeds retraining threshold
        drift_percentage = self.drift_results["drift_percentage"]
        retraining_threshold = self.config["retraining_triggers"]["data_drift_threshold"]
        
        if drift_percentage > retraining_threshold:
            return True, f"Data drift exceeds retraining threshold: {drift_percentage:.2%} > {retraining_threshold:.2%}"
        
        return False, "Data drift does not exceed retraining threshold"


if __name__ == "__main__":
    # Example usage
    from src.InsurancePremiumPrediction.constants import NUMERICAL_FEATURES, CATEGORICAL_FEATURES
    
    detector = DataDriftDetector()
    detector.load_reference_data()
    
    # Detect drift with current data
    current_data_path = "data/current_data.csv"
    results = detector.detect_drift(current_data_path, NUMERICAL_FEATURES, CATEGORICAL_FEATURES)
    
    # Generate report
    report_path = detector.generate_drift_report()
    print(f"Drift report generated at: {report_path}")
    
    # Check if retraining is needed
    should_retrain, reason = detector.should_retrain()
    if should_retrain:
        print(f"Model retraining recommended: {reason}")
    else:
        print(f"Model retraining not needed: {reason}")
