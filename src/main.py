import os
from sklearn.model_selection import train_test_split

# Core Configuration and Utilities
from src.config import create_directories, RANDOM_STATE, TEST_SIZE, CLEANED_DATA_PATH
from src.utils import setup_logger

# Pipeline Modules
from src.data_loader import load_raw_data, validate_data
from src.preprocessing import clean_data
from src.feature_engineering import run_feature_engineering
from src.model import initialize_model, train_model, save_trained_model
from src.evaluation import evaluate_predictions, save_evaluation_report
from src.visualization import plot_confusion_matrix, plot_feature_importances, plot_roc_curve

logger = setup_logger("MainPipeline")

def run_pipeline():
    logger.info("Initializing Machine Learning Pipeline...")
    
    # Step 0: Bootstrap required directories
    create_directories()
    
    # Step 1: Load and Validate Data
    try:
        raw_df = load_raw_data()
        if not validate_data(raw_df):
            logger.error("Data validation failed. Halting pipeline.")
            return
    except Exception as e:
        logger.error(f"Pipeline broke during data ingest: {e}")
        return

    # Step 2: Clean Data and Save it to disk
    cleaned_df = clean_data(raw_df)
    cleaned_df.to_csv(CLEANED_DATA_PATH, index=False)
    logger.info(f"Saved cleaned dataset to: {CLEANED_DATA_PATH}")
    
    # Step 3: Feature Engineering
    X_transformed, y, feature_names = run_feature_engineering()
    if X_transformed is None:
        logger.error("Feature engineering failed. Halting pipeline.")
        return

    # Step 4: Train/Test Split
    logger.info(f"Splitting data with test size = {TEST_SIZE} and random state = {RANDOM_STATE}")
    X_train, X_test, y_train, y_test = train_test_split(
        X_transformed, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"Train set shape: {X_train.shape}, Test set shape: {X_test.shape}")

    # Step 5: Model Training
    model = initialize_model()
    model = train_model(model, X_train, y_train)
    save_trained_model(model)

    # Step 6: Prediction & Inference
    logger.info("Running evaluation predictions on test set...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # Step 7: Evaluate Metrics & Save Text Report
    metrics = evaluate_predictions(y_test, y_pred, y_prob)
    feature_importances = model.feature_importances_
    save_evaluation_report(metrics, feature_names, feature_importances)

    # Step 8: Generate and Save Diagnostic Plots
    plot_confusion_matrix(y_test, y_pred)
    plot_feature_importances(feature_names, feature_importances)
    plot_roc_curve(y_test, y_prob)

    logger.info("=======================================================")
    logger.info("  PIPELINE EXECUTED SUCCESSFULLY - ALL ARTIFACTS SAVED ")
    logger.info("=======================================================")

if __name__ == "__main__":
    run_pipeline()