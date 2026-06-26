import os
from pathlib import Path

# Base Directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data Paths
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
PROCESSED_DATA_DIR = os.path.join(BASE_DIR, "data", "processed")

RAW_DATA_PATH = os.path.join(RAW_DATA_DIR, "heart.csv")
CLEANED_DATA_PATH = os.path.join(PROCESSED_DATA_DIR, "cleaned_heart.csv")

# Output Paths
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")

# Model Hyperparameters & Settings
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Feature Definitions based on heart.csv
TARGET_COLUMN = "target"
CATEGORICAL_FEATURES = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]
NUMERICAL_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak"]

def create_directories():
    """Utility to ensure project directories exist before running pipeline."""
    dirs = [PROCESSED_DATA_DIR, FIGURES_DIR, REPORTS_DIR, MODELS_DIR]
    for d in dirs:
        os.makedirs(d, exist_ok=True)