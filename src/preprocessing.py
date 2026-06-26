import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import RAW_DATA_PATH, CLEANED_DATA_PATH, RANDOM_STATE, TEST_SIZE, TARGET_COLUMN
from src.utils import setup_logger

logger = setup_logger("Preprocessing")

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Performs basic cleaning operations on the raw dataframe."""
    initial_shape = df.shape
    
    # Check and drop duplicate rows
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        logger.info(f"Found {duplicates} duplicate rows. Removing them...")
        df = df.drop_duplicates().reset_index(drop=True)
    
    # Check for missing values (even though heart.csv is typically complete)
    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        logger.warning(f"Found {missing_values} missing values. Imputing with column modes/medians...")
        # Simple fallback imputation
        for col in df.columns:
            if df[col].isnull().any():
                df[col] = df[col].fillna(df[col].mode()[0])
    
    logger.info(f"Data cleaning complete. Shape changed from {initial_shape} to {df.shape}")
    return df

def split_features_target(df: pd.DataFrame, target_col: str = TARGET_COLUMN):
    """Splits the dataframe into features (X) and target (y)."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    return X, y

def process_and_save_data():
    """Main function to clean raw data and write it to the processed folder."""
    if not os.path.exists(RAW_DATA_PATH):
        logger.error(f"Cannot pre-process; raw file missing at {RAW_DATA_PATH}")
        return
        
    df_raw = pd.read_csv(RAW_DATA_PATH)
    df_cleaned = clean_data(df_raw)
    
    # Ensure processed directory exists and save
    os.makedirs(os.path.dirname(CLEANED_DATA_PATH), exist_ok=True)
    df_cleaned.to_csv(CLEANED_DATA_PATH, index=False)
    logger.info(f"Saved cleaned dataset to: {CLEANED_DATA_PATH}")

if __name__ == "__main__":
    process_and_save_data()