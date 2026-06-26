import os
import joblib
import logging
import sys

def setup_logger(name: str = "DiseasePrediction") -> logging.Logger:
    """Sets up a standardized logging configuration for the pipeline."""
    logger = logging.getLogger(name)
    
    # If logger already has handlers, don't add them again
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter(
            '[%(asctime)s] %(levelname)s [%(name)s.%(funcName)s:%(lineno)d] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )

        # Stream Handler (Console output)
        stream_handler = logging.StreamHandler(sys.stdout)
        stream_handler.setFormatter(formatter)
        logger.addHandler(stream_handler)

    return logger

# Initialize a default logger for utility operations
logger = setup_logger("Utils")

def save_artifact(artifact, filepath: str):
    """Saves a python object (model, scaler, etc.) using joblib."""
    try:
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(artifact, filepath)
        logger.info(f"Successfully saved artifact to: {filepath}")
    except Exception as e:
        logger.error(f"Failed to save artifact to {filepath}. Error: {e}")
        raise e

def load_artifact(filepath: str):
    """Loads a python object using joblib."""
    if not os.path.exists(filepath):
        logger.error(f"Artifact file not found at: {filepath}")
        raise FileNotFoundError(f"No artifact found at {filepath}")
    try:
        artifact = joblib.load(filepath)
        logger.info(f"Successfully loaded artifact from: {filepath}")
        return artifact
    except Exception as e:
        logger.error(f"Failed to load artifact from {filepath}. Error: {e}")
        raise e