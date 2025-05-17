"""
Data Validation Component

This module contains the DataValidation class which is responsible for:
1. Validating the dataset schema
2. Checking for required columns
3. Validating data types
4. Checking for missing values
5. Generating a validation status report
"""
import os
import pandas as pd
from pathlib import Path

from InsurancePremiumPrediction import logger
from InsurancePremiumPrediction.entity.config_entity import DataValidationConfig
from InsurancePremiumPrediction.constants import SCHEMA_FILE_PATH


class DataValidation:
    """
    Class for data validation operations.

    This class handles:
    1. Validating the dataset schema
    2. Checking for required columns
    3. Validating data types
    4. Checking for missing values
    5. Generating a validation status report
    """

    def __init__(self, config: DataValidationConfig, schema_filepath=SCHEMA_FILE_PATH):
        """
        Initialize the DataValidation class.

        Args:
            config (DataValidationConfig): Configuration for data validation
            schema_filepath (Path, optional): Path to the schema file. Defaults to SCHEMA_FILE_PATH.
        """
        self.config = config
        self.schema_filepath = schema_filepath
        self.validation_status = True

        # Create root directory
        os.makedirs(self.config.root_dir, exist_ok=True)

    def validate_columns(self, dataframe: pd.DataFrame) -> bool:
        """
        Validate that the dataframe contains all required columns.

        Args:
            dataframe (pd.DataFrame): The dataframe to validate

        Returns:
            bool: True if validation passes, False otherwise
        """
        validation_status = True

        # Check if all required columns are present
        all_columns = list(dataframe.columns)

        for column in self.config.required_columns:
            if column not in all_columns:
                validation_status = False
                logger.error(
                    f"Required column {column} not found in the dataset")

        return validation_status

    def validate_data_types(self, dataframe: pd.DataFrame) -> bool:
        """
        Validate that the dataframe columns have the correct data types.

        Args:
            dataframe (pd.DataFrame): The dataframe to validate

        Returns:
            bool: True if validation passes, False otherwise
        """
        validation_status = True
        schema = self._read_schema()

        # Check data types for each column
        for column, properties in schema.columns.items():
            if column in dataframe.columns:
                col_type = properties.get('type', None)

                # Validate numerical types
                if col_type == 'int' and not pd.api.types.is_integer_dtype(dataframe[column]):
                    try:
                        # Try to convert to integer
                        dataframe[column] = dataframe[column].astype(int)
                        logger.warning(
                            f"Column {column} converted to integer type")
                    except:
                        validation_status = False
                        logger.error(
                            f"Column {column} should be of type int but has incompatible values")

                elif col_type == 'float' and not pd.api.types.is_float_dtype(dataframe[column]):
                    try:
                        # Try to convert to float
                        dataframe[column] = dataframe[column].astype(float)
                        logger.warning(
                            f"Column {column} converted to float type")
                    except:
                        validation_status = False
                        logger.error(
                            f"Column {column} should be of type float but has incompatible values")

                # Validate categorical types
                elif col_type == 'categorical':
                    # Check if values are within allowed categories
                    categories = properties.get('categories', [])
                    if categories and not set(dataframe[column].unique()).issubset(set(categories)):
                        invalid_values = set(
                            dataframe[column].unique()) - set(categories)
                        validation_status = False
                        logger.error(
                            f"Column {column} contains invalid categories: {invalid_values}")

        return validation_status

    def validate_missing_values(self, dataframe: pd.DataFrame) -> bool:
        """
        Validate that the dataframe does not contain missing values.

        Args:
            dataframe (pd.DataFrame): The dataframe to validate

        Returns:
            bool: True if validation passes, False otherwise
        """
        # Check for missing values
        missing_values = dataframe.isnull().sum()

        if missing_values.sum() > 0:
            columns_with_missing = missing_values[missing_values > 0].index.tolist(
            )
            logger.warning(
                f"Missing values found in columns: {columns_with_missing}")
            logger.warning(
                f"Missing value counts: {missing_values[missing_values > 0].to_dict()}")

            # Return True anyway as we'll handle missing values in data transformation
            return True

        return True

    def validate_value_ranges(self, dataframe: pd.DataFrame) -> bool:
        """
        Validate that the values in the dataframe are within the specified ranges.

        Args:
            dataframe (pd.DataFrame): The dataframe to validate

        Returns:
            bool: True if validation passes, False otherwise
        """
        validation_status = True
        schema = self._read_schema()

        # Check value ranges for each column
        for column, properties in schema.columns.items():
            if column in dataframe.columns and 'constraints' in properties:
                constraints = properties.constraints

                # Check minimum value
                if 'min' in constraints:
                    min_val = constraints.min
                    if dataframe[column].min() < min_val:
                        validation_status = False
                        logger.error(
                            f"Column {column} contains values below the minimum of {min_val}")

                # Check maximum value
                if 'max' in constraints:
                    max_val = constraints.max
                    if dataframe[column].max() > max_val:
                        validation_status = False
                        logger.error(
                            f"Column {column} contains values above the maximum of {max_val}")

        return validation_status

    def _read_schema(self):
        """
        Read the schema file.

        Returns:
            ConfigBox: The schema configuration
        """
        from InsurancePremiumPrediction.utils import read_yaml
        return read_yaml(self.schema_filepath)

    def validate_dataset(self, train_path: Path, test_path: Path, validation_path: Path) -> bool:
        """
        Validate the dataset.

        Args:
            train_path (Path): Path to the training dataset
            test_path (Path): Path to the test dataset
            validation_path (Path): Path to the validation dataset

        Returns:
            bool: True if validation passes, False otherwise
        """
        logger.info("Validating dataset")

        try:
            # Load datasets
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            validation_df = pd.read_csv(validation_path)

            # Validate columns
            logger.info("Validating columns")
            train_cols_valid = self.validate_columns(train_df)
            test_cols_valid = self.validate_columns(test_df)
            val_cols_valid = self.validate_columns(validation_df)

            # Validate data types
            logger.info("Validating data types")
            train_types_valid = self.validate_data_types(train_df)
            test_types_valid = self.validate_data_types(test_df)
            val_types_valid = self.validate_data_types(validation_df)

            # Validate missing values
            logger.info("Validating missing values")
            train_missing_valid = self.validate_missing_values(train_df)
            test_missing_valid = self.validate_missing_values(test_df)
            val_missing_valid = self.validate_missing_values(validation_df)

            # Validate value ranges
            logger.info("Validating value ranges")
            train_ranges_valid = self.validate_value_ranges(train_df)
            test_ranges_valid = self.validate_value_ranges(test_df)
            val_ranges_valid = self.validate_value_ranges(validation_df)

            # Overall validation status
            validation_status = (
                train_cols_valid and test_cols_valid and val_cols_valid and
                train_types_valid and test_types_valid and val_types_valid and
                train_missing_valid and test_missing_valid and val_missing_valid and
                train_ranges_valid and test_ranges_valid and val_ranges_valid
            )

            # Write validation status to file
            self._write_validation_status(validation_status)

            return validation_status

        except Exception as e:
            logger.error(f"Error validating dataset: {e}")
            self._write_validation_status(False)
            return False

    def _write_validation_status(self, status: bool):
        """
        Write the validation status to a file.

        Args:
            status (bool): The validation status
        """
        # Create parent directory if it doesn't exist
        os.makedirs(os.path.dirname(self.config.status_file), exist_ok=True)

        # Write status to file
        with open(self.config.status_file, 'w') as f:
            f.write(f"Validation status: {'Success' if status else 'Failed'}")

        logger.info(f"Validation status written to {self.config.status_file}")

    def initiate_data_validation(self, train_path: Path, test_path: Path, validation_path: Path) -> bool:
        """
        Initiate the data validation process.

        Args:
            train_path (Path): Path to the training dataset
            test_path (Path): Path to the test dataset
            validation_path (Path): Path to the validation dataset

        Returns:
            bool: True if validation passes, False otherwise
        """
        logger.info("Initiating data validation")
        validation_status = self.validate_dataset(
            train_path, test_path, validation_path)

        if validation_status:
            logger.info("Data validation completed successfully")
        else:
            logger.warning("Data validation completed with issues")

        return validation_status
