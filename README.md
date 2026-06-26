# Disease Prediction using Machine Learning

A production-ready, modular Machine Learning pipeline designed to predict the presence of heart disease based on clinical features using a Random Forest Classifier.

## 📂 Project Structure

```text
Disease-Prediction-ML/
│
├── data/
│   ├── raw/
│   │   └── heart.csv               # Original dataset
│   └── processed/
│       └── cleaned_heart.csv       # Cleaned, deduplicated dataset
│
├── src/
│   ├── main.py                     # Pipeline orchestrator
│   ├── config.py                   # Centralized paths and hyperparameters
│   ├── data_loader.py              # Ingests and validates data
│   ├── preprocessing.py            # Handles deduplication and cleaning
│   ├── eda.py                      # Generates baseline summary statistics
│   ├── feature_engineering.py      # Scales/encodes features and saves preprocessor
│   ├── model.py                    # Handles model initialization and training
│   ├── evaluation.py               # Generates validation text reports
│   ├── visualization.py            # Generates evaluation charts and graphs
│   └── utils.py                    # Core logging and artifact handlers
│
├── outputs/
│   ├── figures/                    # Saved plots (Confusion Matrix, ROC Curve, etc.)
│   ├── reports/                    # text summary and metrics profiles
│   └── models/                     # Saved model and preprocessor joblib artifacts
│
├── requirements.txt                # External dependencies
└── .gitignore                      # Files ignored by git version control