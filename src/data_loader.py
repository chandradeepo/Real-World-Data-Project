import os
import pandas as pd
from typing import Optional
from src.config import RAW_DATA_PATH, TARGET_COLUMN, NUMERICAL_FEATURES, CATEGORICAL_FEATURES
from src.utils import setup_logger

logger = setup_logger("DataLoader")

def load_raw_data(file_path: str = RAW_DATA_PATH) -> pd.DataFrame:
    """Loads the raw heart disease dataset from the specified path."""
    logger.info(f"Attempting to load raw data from: {file_path}")
    
    if not os.path.exists(file_path):
        error_msg = f"Raw data file not found at {file_path}. Please check your directories."
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)
        
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Successfully loaded dataset with shape: {df.shape}")
        return df
    except Exception as e:
        logger.error(f"Error reading the CSV file: {e}")
        raise e

def validate_data(df: pd.DataFrame) -> bool:
    """Validates that the required columns are present in the dataset."""
    expected_columns = [TARGET_COLUMN] + NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    missing_cols = [col for col in expected_columns if col not in df.columns]
    
    if missing_cols:
        logger.warning(f"Missing expected columns in the dataset: {missing_cols}")
        return False
        
    logger.info("Dataset validation passed. All expected features and target column are present.")
    return True

if __name__ == "__main__":
    # Test block to verify loading works correctly when run directly
    try:
        data = load_raw_data()
        if validate_data(data):
            print("\n--- Data Sample ---")
            print(data.head())
    except Exception as e:
        print(f"Execution failed: {e}")