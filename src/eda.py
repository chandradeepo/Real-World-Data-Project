import os
import pandas as pd
from src.config import CLEANED_DATA_PATH, REPORTS_DIR, TARGET_COLUMN
from src.utils import setup_logger

logger = setup_logger("EDA")

def generate_summary_statistics(df: pd.DataFrame) -> str:
    """Generates basic summary statistics and returns them as a structured string."""
    logger.info("Generating summary statistics for the dataset.")
    
    summary = []
    summary.append("=== DATASET OVERVIEW ===")
    summary.append(f"Total Rows: {df.shape[0]}")
    summary.append(f"Total Columns: {df.shape[1]}\n")
    
    summary.append("=== TARGET DISTRIBUTION ===")
    target_counts = df[TARGET_COLUMN].value_counts()
    target_pct = df[TARGET_COLUMN].value_counts(normalize=True) * 100
    for val, count in target_counts.items():
        summary.append(f"Class {val}: {count} ({target_pct[val]:.2f}%)")
    summary.append("")

    summary.append("=== MISSING VALUES ===")
    missing = df.isnull().sum()
    summary.append(missing[missing > 0].to_string() if missing.sum() > 0 else "No missing values found.")
    summary.append("")

    summary.append("=== NUMERICAL FEATURES STATS ===")
    summary.append(df.describe().T.to_string())
    
    return "\n".join(summary)

def run_eda_pipeline():
    """Main pipeline execution for exploratory data analysis."""
    if not os.path.exists(CLEANED_DATA_PATH):
        logger.error(f"Cleaned data file not found at {CLEANED_DATA_PATH}. Please run preprocessing first.")
        return

    df = pd.read_csv(CLEANED_DATA_PATH)
    
    # Generate report text
    report_text = generate_summary_statistics(df)
    
    # Save the report to the outputs directory
    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(REPORTS_DIR, "eda_summary.txt")
    
    with open(report_path, "w") as f:
        f.write(report_text)
        
    logger.info(f"EDA Summary Report successfully saved to: {report_path}")

if __name__ == "__main__":
    run_eda_pipeline()