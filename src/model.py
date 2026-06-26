import os
from sklearn.ensemble import RandomForestClassifier
from src.config import MODELS_DIR, RANDOM_STATE
from src.utils import setup_logger, save_artifact, load_artifact

logger = setup_logger("Model")

def initialize_model(n_estimators: int = 100, max_depth: int = None) -> RandomForestClassifier:
    """Initializes a Random Forest Classifier with chosen hyperparameters."""
    logger.info(f"Initializing RandomForestClassifier (n_estimators={n_estimators}, max_depth={max_depth})")
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=RANDOM_STATE,
        n_jobs=-1
    )
    return model

def train_model(model, X_train, y_train):
    """Trains the initialized machine learning model on provided training features."""
    logger.info(f"Starting model training on {X_train.shape[0]} samples...")
    try:
        model.fit(X_train, y_train)
        logger.info("Model training completed successfully.")
        return model
    except Exception as e:
        logger.error(f"Error occurred during model training: {e}")
        raise e

def save_trained_model(model, filename: str = "heart_disease_model.joblib"):
    """Saves the trained model object into the dedicated models directory."""
    filepath = os.path.join(MODELS_DIR, filename)
    logger.info(f"Saving trained model to {filepath}")
    save_artifact(model, filepath)

def load_trained_model(filename: str = "heart_disease_model.joblib"):
    """Loads a previously saved model artifact from the models directory."""
    filepath = os.path.join(MODELS_DIR, filename)
    logger.info(f"Loading trained model from {filepath}")
    return load_artifact(filepath)

if __name__ == "__main__":
    # Diagnostic test block to verify initialization setup works
    test_model = initialize_model()
    print("Model parameters:", test_model.get_params())