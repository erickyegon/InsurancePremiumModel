"""
Data Ingestion Component

This module contains the DataIngestion class which is responsible for:
1. Downloading or copying the data from the source
2. Splitting the data into train, test, and validation sets
3. Saving the data to the specified locations
"""
import os
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

from InsurancePremiumPrediction import logger
from InsurancePremiumPrediction.entity.config_entity import DataIngestionConfig
from InsurancePremiumPrediction.constants import RANDOM_STATE, TEST_SIZE, VALIDATION_SIZE


class DataIngestion:
    """
    Class for data ingestion operations.

    This class handles:
    1. Getting data from source
    2. Splitting data into train, test, and validation sets
    3. Saving the data to the specified locations
    """

    def __init__(self, config: DataIngestionConfig):
        """
        Initialize the DataIngestion class.

        Args:
            config (DataIngestionConfig): Configuration for data ingestion.
        """
        self.config = config
        # Create root directory
        os.makedirs(self.config.root_dir, exist_ok=True)

    def get_data(self):
        """
        Get data from the source and save it locally.

        This method copies the data from the source to the local data file.
        Handles both CSV and Excel files.

        Returns:
            None
        """
        logger.info("Getting data from source")

        if not os.path.exists(self.config.local_data_file):
            # Create parent directory if it doesn't exist
            os.makedirs(os.path.dirname(
                self.config.local_data_file), exist_ok=True)

            # Copy data from source to local
            if os.path.exists(self.config.source_data):
                # Check file extension
                file_extension = os.path.splitext(
                    self.config.source_data)[1].lower()

                if file_extension == '.csv':
                    # If source is a CSV file
                    df = pd.read_csv(self.config.source_data)
                elif file_extension in ['.xlsx', '.xls']:
                    # If source is an Excel file
                    df = pd.read_excel(self.config.source_data)
                    logger.info(f"Read Excel file with shape: {df.shape}")
                    logger.info(
                        f"Columns in the Excel file: {df.columns.tolist()}")

                    # Rename columns with spaces to use underscores
                    if 'Number Of Dependants' in df.columns:
                        df = df.rename(
                            columns={'Number Of Dependants': 'Number_Of_Dependants'})

                    if 'Medical History' in df.columns:
                        df = df.rename(
                            columns={'Medical History': 'Medical_History'})

                    logger.info(
                        f"Columns after renaming: {df.columns.tolist()}")

                    # Print a sample of the data
                    logger.info(f"Sample data from Excel file:\n{df.head(2)}")
                else:
                    logger.error(f"Unsupported file format: {file_extension}")
                    raise ValueError(
                        f"Unsupported file format: {file_extension}")

                # Save as CSV
                df.to_csv(self.config.local_data_file, index=False)
                logger.info(
                    f"Data copied from {self.config.source_data} to {self.config.local_data_file}")
            else:
                # If source file doesn't exist
                logger.error(
                    f"Source data not found at {self.config.source_data}")
                raise FileNotFoundError(
                    f"Source data not found at {self.config.source_data}")
        else:
            logger.info(
                f"Data already exists at {self.config.local_data_file}")

    def split_data(self):
        """
        Split the data into train, test, and validation sets.

        This method:
        1. Reads the data from the local data file
        2. Splits it into train, test, and validation sets
        3. Saves the splits to the specified locations

        Returns:
            None
        """
        logger.info("Splitting data into train, test, and validation sets")

        # Read the data
        df = pd.read_csv(self.config.local_data_file)

        # Split into train and temp sets (80% train, 20% temp)
        train_df, temp_df = train_test_split(
            df,
            test_size=TEST_SIZE + VALIDATION_SIZE,
            random_state=RANDOM_STATE
        )

        # Split temp into test and validation sets
        test_df, val_df = train_test_split(
            temp_df,
            test_size=VALIDATION_SIZE / (TEST_SIZE + VALIDATION_SIZE),
            random_state=RANDOM_STATE
        )

        # Create directories if they don't exist
        os.makedirs(os.path.dirname(
            self.config.train_data_path), exist_ok=True)
        os.makedirs(os.path.dirname(self.config.test_data_path), exist_ok=True)
        os.makedirs(os.path.dirname(
            self.config.validation_data_path), exist_ok=True)

        # Save the splits
        train_df.to_csv(self.config.train_data_path, index=False)
        test_df.to_csv(self.config.test_data_path, index=False)
        val_df.to_csv(self.config.validation_data_path, index=False)

        logger.info(
            f"Data split complete. Shapes: Train: {train_df.shape}, Test: {test_df.shape}, Validation: {val_df.shape}")

    def initiate_data_ingestion(self):
        """
        Initiate the data ingestion process.

        This method:
        1. Gets the data from the source
        2. Splits the data into train, test, and validation sets

        Returns:
            tuple: Paths to the train, test, and validation data files
        """
        logger.info("Initiating data ingestion")
        self.get_data()
        self.split_data()
        logger.info("Data ingestion completed")

        return (
            self.config.train_data_path,
            self.config.test_data_path,
            self.config.validation_data_path
        )
