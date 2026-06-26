import os
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from src.config import REPORTS_DIR
from src.utils import setup_logger

logger = setup_logger("Evaluation")

def evaluate_predictions(y_true, y_pred, y_prob=None) -> dict:
    """Calculates evaluation metrics and returns them as a dictionary."""
    logger.info("Calculating evaluation metrics...")
    
    metrics = {
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "classification_report": classification_report(y_true, y_pred)
    }
    
    if y_prob is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
        logger.info(f"Calculated ROC-AUC Score: {metrics['roc_auc']:.4f}")
        
    return metrics

def save_evaluation_report(metrics, feature_names, feature_importances, filename: str = "model_evaluation_report.txt"):
    """Formats and writes a comprehensive evaluation report to a text file."""
    filepath = os.path.join(REPORTS_DIR, filename)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    report_lines = [
        "=========================================",
        "        MODEL EVALUATION REPORT          ",
        "=========================================\n",
        "CLASSIFICATION REPORT:",
        metrics["classification_report"],
        "CONFUSION MATRIX:",
        str(metrics["confusion_matrix"]),
        "\n"
    ]
    
    if "roc_auc" in metrics:
        report_lines.append(f"ROC-AUC SCORE: {metrics['roc_auc']:.4f}\n")
        
    report_lines.append("TOP FEATURE IMPORTANCES:")
    # Combine feature names with their importance values and sort them
    feat_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importances
    }).sort_values(by='Importance', ascending=False)
    
    report_lines.append(feat_imp_df.to_string(index=False))
    
    with open(filepath, "w") as f:
        f.write("\n".join(report_lines))
        
    logger.info(f"Evaluation text report saved to: {filepath}")

if __name__ == "__main__":
    logger.info("Evaluation module initialized.")