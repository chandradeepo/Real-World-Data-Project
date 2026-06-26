import os
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from src.config import (
    CLEANED_DATA_PATH, MODELS_DIR, TARGET_COLUMN, 
    NUMERICAL_FEATURES, CATEGORICAL_FEATURES, RANDOM_STATE
)
from src.utils import setup_logger, save_artifact

logger = setup_logger("FeatureEngineering")

def build_preprocessor() -> ColumnTransformer:
    """Creates a ColumnTransformer for preprocessing numerical and categorical features."""
    logger.info("Building feature engineering pipelines...")
    
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, NUMERICAL_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ]
    )
    return preprocessor

def run_feature_engineering():
    """Fits the preprocessor on the dataset and returns transformed features."""
    if not os.path.exists(CLEANED_DATA_PATH):
        logger.error(f"Cleaned data not found at {CLEANED_DATA_PATH}. Please run preprocessing first.")
        return None, None, None

    df = pd.read_csv(CLEANED_DATA_PATH)
    
    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    
    preprocessor = build_preprocessor()
    
    logger.info("Fitting and transforming features...")
    X_transformed = preprocessor.fit_transform(X)
    
    # Extract feature names for downstream tracking/transparency
    try:
        cat_encoder = preprocessor.named_transformers_['cat']
        encoded_cat_features = cat_encoder.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
        all_features = NUMERICAL_FEATURES + encoded_cat_features
    except Exception as e:
        logger.warning(f"Could not extract feature names: {e}")
        all_features = [f"feature_{i}" for i in range(X_transformed.shape[1])]

    # Save the fitted preprocessor pipeline to use during model deployment/testing
    preprocessor_path = os.path.join(MODELS_DIR, "preprocessor.joblib")
    save_artifact(preprocessor, preprocessor_path)
    
    logger.info(f"Feature engineering complete. Transformed shape: {X_transformed.shape}")
    return X_transformed, y, all_features

if __name__ == "__main__":
    X_trans, y, feature_names = run_feature_engineering()