import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc
from src.config import FIGURES_DIR
from src.utils import setup_logger

logger = setup_logger("Visualization")

# Set styling for clean, professional charts
sns.set_theme(style="whitegrid")

def plot_confusion_matrix(y_true, y_pred, filename: str = "confusion_matrix.png"):
    """Generates and saves a heatmap of the confusion matrix."""
    logger.info("Generating confusion matrix plot...")
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Healthy', 'Disease'], 
                yticklabels=['Healthy', 'Disease'])
    
    plt.title('Confusion Matrix', fontsize=14, pad=15)
    plt.ylabel('Actual Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    filepath = os.path.join(FIGURES_DIR, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()
    logger.info(f"Confusion matrix saved to {filepath}")

def plot_feature_importances(feature_names, importances, top_n: int = 10, filename: str = "feature_importances.png"):
    """Generates and saves a horizontal bar chart of the top feature importances."""
    logger.info("Generating feature importance plot...")
    
    feat_imp_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    }).sort_values(by='Importance', ascending=False).head(top_n)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feat_imp_df, hue='Feature', legend=False, palette='viridis')
    
    plt.title(f'Top {top_n} Feature Importances', fontsize=14, pad=15)
    plt.xlabel('Relative Importance Score', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    
    filepath = os.path.join(FIGURES_DIR, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()
    logger.info(f"Feature importance chart saved to {filepath}")

def plot_roc_curve(y_true, y_prob, filename: str = "roc_curve.png"):
    """Generates and saves an ROC Curve plot with AUC value."""
    if y_prob is None:
        logger.warning("Skipping ROC curve plot: no prediction probabilities provided.")
        return

    logger.info("Generating ROC curve plot...")
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(7, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, pad=15)
    plt.legend(loc="lower right", fontsize=11)
    plt.tight_layout()
    
    filepath = os.path.join(FIGURES_DIR, filename)
    plt.savefig(filepath, dpi=300)
    plt.close()
    logger.info(f"ROC Curve saved to {filepath}")