"""
Model Registry Module

This module provides functionality for versioning and registering models.
It tracks model versions, metadata, and performance metrics.
"""
import os
import sys
import json
import shutil
import logging
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
from pathlib import Path
import joblib
import yaml

# Add the src directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.InsurancePremiumPrediction import logger
from src.InsurancePremiumPrediction.utils.common import read_yaml, create_directories

class ModelRegistry:
    """
    Class for managing model versions and metadata.
    """
    
    def __init__(self, registry_dir: str = "model_registry"):
        """
        Initialize the ModelRegistry.
        
        Args:
            registry_dir: Directory to store model registry
        """
        self.registry_dir = Path(registry_dir)
        create_directories([self.registry_dir])
        
        # Create subdirectories
        self.models_dir = self.registry_dir / "models"
        self.metadata_dir = self.registry_dir / "metadata"
        create_directories([self.models_dir, self.metadata_dir])
        
        # Registry index file
        self.index_file = self.registry_dir / "index.json"
        self.index = self._load_index()
        
        # Set up logging
        logging.basicConfig(
            filename=os.path.join("logs", "model_registry.log"),
            level=logging.INFO,
            format="[%(asctime)s: %(levelname)s: %(module)s]: %(message)s",
            filemode="a"
        )
        self.logger = logging.getLogger(__name__)
    
    def _load_index(self) -> Dict:
        """
        Load the registry index.
        
        Returns:
            Dictionary with registry index
        """
        if self.index_file.exists():
            try:
                with open(self.index_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading registry index: {e}")
                return {"models": {}, "latest_version": {}}
        else:
            return {"models": {}, "latest_version": {}}
    
    def _save_index(self) -> None:
        """
        Save the registry index.
        """
        try:
            with open(self.index_file, "w") as f:
                json.dump(self.index, f, indent=2)
        except Exception as e:
            self.logger.error(f"Error saving registry index: {e}")
    
    def register_model(self, model_path: str, model_name: str, version: Optional[str] = None,
                      metadata: Optional[Dict] = None, metrics: Optional[Dict] = None,
                      description: Optional[str] = None, tags: Optional[List[str]] = None,
                      promote_to_production: bool = False) -> Dict:
        """
        Register a model in the registry.
        
        Args:
            model_path: Path to the model file
            model_name: Name of the model
            version: Version of the model (if None, auto-generated)
            metadata: Model metadata
            metrics: Model performance metrics
            description: Model description
            tags: Model tags
            promote_to_production: Whether to promote the model to production
            
        Returns:
            Dictionary with registration information
        """
        # Generate version if not provided
        if version is None:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            version = f"v{timestamp}"
        
        # Create model directory
        model_dir = self.models_dir / model_name / version
        create_directories([model_dir])
        
        # Copy model file
        model_filename = os.path.basename(model_path)
        model_dest = model_dir / model_filename
        shutil.copy2(model_path, model_dest)
        
        # Create metadata
        if metadata is None:
            metadata = {}
        
        metadata.update({
            "name": model_name,
            "version": version,
            "created_at": datetime.now().isoformat(),
            "file_path": str(model_dest),
            "description": description or "",
            "tags": tags or [],
            "metrics": metrics or {}
        })
        
        # Save metadata
        metadata_file = self.metadata_dir / f"{model_name}_{version}.json"
        with open(metadata_file, "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Update index
        if model_name not in self.index["models"]:
            self.index["models"][model_name] = {}
        
        self.index["models"][model_name][version] = {
            "metadata_path": str(metadata_file),
            "model_path": str(model_dest),
            "created_at": metadata["created_at"]
        }
        
        # Update latest version
        self.index["latest_version"][model_name] = version
        
        # Promote to production if requested
        if promote_to_production:
            self.promote_to_production(model_name, version)
        
        # Save index
        self._save_index()
        
        self.logger.info(f"Registered model {model_name} version {version}")
        return {
            "model_name": model_name,
            "version": version,
            "model_path": str(model_dest),
            "metadata_path": str(metadata_file)
        }
    
    def promote_to_production(self, model_name: str, version: str) -> Dict:
        """
        Promote a model version to production.
        
        Args:
            model_name: Name of the model
            version: Version of the model
            
        Returns:
            Dictionary with promotion information
        """
        if model_name not in self.index["models"] or version not in self.index["models"][model_name]:
            raise ValueError(f"Model {model_name} version {version} not found in registry")
        
        # Update production version
        self.index["production"] = {
            "model_name": model_name,
            "version": version,
            "promoted_at": datetime.now().isoformat()
        }
        
        # Save index
        self._save_index()
        
        self.logger.info(f"Promoted model {model_name} version {version} to production")
        return self.index["production"]
    
    def get_model(self, model_name: str, version: Optional[str] = None) -> Any:
        """
        Get a model from the registry.
        
        Args:
            model_name: Name of the model
            version: Version of the model (if None, latest version)
            
        Returns:
            Loaded model
        """
        if model_name not in self.index["models"]:
            raise ValueError(f"Model {model_name} not found in registry")
        
        if version is None:
            version = self.index["latest_version"].get(model_name)
            if version is None:
                raise ValueError(f"No versions found for model {model_name}")
        
        if version not in self.index["models"][model_name]:
            raise ValueError(f"Version {version} not found for model {model_name}")
        
        model_path = self.index["models"][model_name][version]["model_path"]
        
        try:
            model = joblib.load(model_path)
            self.logger.info(f"Loaded model {model_name} version {version}")
            return model
        except Exception as e:
            self.logger.error(f"Error loading model {model_name} version {version}: {e}")
            raise
    
    def get_production_model(self) -> Any:
        """
        Get the production model.
        
        Returns:
            Loaded production model
        """
        if "production" not in self.index:
            raise ValueError("No production model set")
        
        model_name = self.index["production"]["model_name"]
        version = self.index["production"]["version"]
        
        return self.get_model(model_name, version)
    
    def get_model_metadata(self, model_name: str, version: Optional[str] = None) -> Dict:
        """
        Get model metadata.
        
        Args:
            model_name: Name of the model
            version: Version of the model (if None, latest version)
            
        Returns:
            Dictionary with model metadata
        """
        if model_name not in self.index["models"]:
            raise ValueError(f"Model {model_name} not found in registry")
        
        if version is None:
            version = self.index["latest_version"].get(model_name)
            if version is None:
                raise ValueError(f"No versions found for model {model_name}")
        
        if version not in self.index["models"][model_name]:
            raise ValueError(f"Version {version} not found for model {model_name}")
        
        metadata_path = self.index["models"][model_name][version]["metadata_path"]
        
        try:
            with open(metadata_path, "r") as f:
                metadata = json.load(f)
            return metadata
        except Exception as e:
            self.logger.error(f"Error loading metadata for model {model_name} version {version}: {e}")
            raise
    
    def list_models(self) -> List[str]:
        """
        List all models in the registry.
        
        Returns:
            List of model names
        """
        return list(self.index["models"].keys())
    
    def list_versions(self, model_name: str) -> List[str]:
        """
        List all versions of a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            List of version strings
        """
        if model_name not in self.index["models"]:
            raise ValueError(f"Model {model_name} not found in registry")
        
        return list(self.index["models"][model_name].keys())
    
    def get_latest_version(self, model_name: str) -> str:
        """
        Get the latest version of a model.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Latest version string
        """
        if model_name not in self.index["latest_version"]:
            raise ValueError(f"No versions found for model {model_name}")
        
        return self.index["latest_version"][model_name]
    
    def delete_model(self, model_name: str, version: Optional[str] = None) -> None:
        """
        Delete a model from the registry.
        
        Args:
            model_name: Name of the model
            version: Version of the model (if None, delete all versions)
        """
        if model_name not in self.index["models"]:
            raise ValueError(f"Model {model_name} not found in registry")
        
        if version is None:
            # Delete all versions
            for ver in list(self.index["models"][model_name].keys()):
                self._delete_model_version(model_name, ver)
            
            # Remove from index
            del self.index["models"][model_name]
            if model_name in self.index["latest_version"]:
                del self.index["latest_version"][model_name]
            
            # Remove from production if it's the production model
            if "production" in self.index and self.index["production"]["model_name"] == model_name:
                del self.index["production"]
            
            self.logger.info(f"Deleted all versions of model {model_name}")
        else:
            # Delete specific version
            if version not in self.index["models"][model_name]:
                raise ValueError(f"Version {version} not found for model {model_name}")
            
            self._delete_model_version(model_name, version)
            
            # Update latest version if needed
            if model_name in self.index["latest_version"] and self.index["latest_version"][model_name] == version:
                if self.index["models"][model_name]:
                    # Set latest to the most recent remaining version
                    versions = list(self.index["models"][model_name].keys())
                    versions.sort()
                    self.index["latest_version"][model_name] = versions[-1]
                else:
                    # No versions left
                    del self.index["latest_version"][model_name]
            
            # Update production if needed
            if "production" in self.index and self.index["production"]["model_name"] == model_name and self.index["production"]["version"] == version:
                del self.index["production"]
            
            self.logger.info(f"Deleted model {model_name} version {version}")
        
        # Save index
        self._save_index()
    
    def _delete_model_version(self, model_name: str, version: str) -> None:
        """
        Delete a specific model version.
        
        Args:
            model_name: Name of the model
            version: Version of the model
        """
        # Get paths
        model_path = self.index["models"][model_name][version]["model_path"]
        metadata_path = self.index["models"][model_name][version]["metadata_path"]
        
        # Delete files
        try:
            if os.path.exists(model_path):
                os.remove(model_path)
            
            if os.path.exists(metadata_path):
                os.remove(metadata_path)
        except Exception as e:
            self.logger.error(f"Error deleting files for model {model_name} version {version}: {e}")
        
        # Remove from index
        del self.index["models"][model_name][version]
        
        # Remove model directory if empty
        model_dir = os.path.dirname(model_path)
        if os.path.exists(model_dir) and not os.listdir(model_dir):
            try:
                os.rmdir(model_dir)
            except Exception as e:
                self.logger.error(f"Error removing directory {model_dir}: {e}")


if __name__ == "__main__":
    # Example usage
    registry = ModelRegistry()
    
    # Register a model
    model_info = registry.register_model(
        model_path="artifacts/model_trainer/model.joblib",
        model_name="insurance_premium_model",
        metadata={
            "algorithm": "XGBoost",
            "hyperparameters": {
                "learning_rate": 0.1,
                "max_depth": 5
            }
        },
        metrics={
            "rmse": 1050.32,
            "mae": 798.45,
            "r2": 0.92
        },
        description="Insurance Premium Prediction Model",
        tags=["regression", "insurance", "premium"],
        promote_to_production=True
    )
    
    print(f"Registered model: {model_info}")
    
    # List models
    models = registry.list_models()
    print(f"Models in registry: {models}")
    
    # Get model metadata
    metadata = registry.get_model_metadata("insurance_premium_model")
    print(f"Model metadata: {metadata}")
    
    # Get production model
    try:
        production_model = registry.get_production_model()
        print("Production model loaded successfully")
    except Exception as e:
        print(f"Error loading production model: {e}")
