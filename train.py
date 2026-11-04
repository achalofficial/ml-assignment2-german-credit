"""
ML Assignment 2 - Model training script.
Dataset: UCI Statlog (German Credit Data) - 1000 instances, 20 features, binary target.
Trains 5 classifiers on the same dataset, evaluates each with 6 metrics,
and saves full preprocessing+model pipelines so the Streamlit app can score
raw test CSVs without retraining.
"""
import warnings
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

# Harmless macOS Accelerate-BLAS matmul warnings (results are finite/correct) - silence them.
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef, confusion_matrix
)

RANDOM_STATE = 42
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
MODEL_DIR.mkdir(exist_ok=True)

COLUMNS = [
    "checking_status", "duration", "credit_history", "purpose", "credit_amount",
    "savings_status", "employment", "installment_commitment", "personal_status",
    "other_parties", "residence_since", "property_magnitude", "age",
    "other_payment_plans", "housing", "existing_credits", "job",
    "num_dependents", "own_telephone", "foreign_worker", "class",
]

NUMERIC_COLS = [
    "duration", "credit_amount", "installment_commitment",
    "residence_since", "age", "existing_credits", "num_dependents",
]
CATEGORICAL_COLS = [c for c in COLUMNS[:-1] if c not in NUMERIC_COLS]
TARGET_COL = "credit_risk"  # 1 = good, 0 = bad


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATA_DIR / "german.data", sep=" ", header=None, names=COLUMNS)
    # original class: 1 = good credit risk, 2 = bad credit risk -> map to 1/0
    df[TARGET_COL] = df["class"].map({1: 1, 2: 0})
    df = df.drop(columns=["class"])
    return df


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ],
        sparse_threshold=0,  # always output a dense array (GaussianNB needs dense input)
    )


def main():
    df = load_dataset()
    df.to_csv(DATA_DIR / "german_credit.csv", index=False)

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # Save the held-out test split as the demo test_data.csv for the Streamlit app
    test_df = X_test.copy()
    test_df[TARGET_COL] = y_test.values
    test_df.to_csv(BASE_DIR / "test_data.csv", index=False)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=2000, solver="liblinear", random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=6, random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(n_neighbors=11),
        "Naive Bayes": GaussianNB(),
        "Support Vector Machine": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
    }

    file_names = {
        "Logistic Regression": "logistic_regression.joblib",
        "Decision Tree": "decision_tree.joblib",
        "KNN": "knn.joblib",
        "Naive Bayes": "naive_bayes.joblib",
        "Support Vector Machine": "svm.joblib",
    }

    results = {}
    for name, clf in models.items():
        pipe = Pipeline(steps=[("preprocess", build_preprocessor()), ("clf", clf)])
        pipe.fit(X_train, y_train)

        y_pred = pipe.predict(X_test)
        y_proba = pipe.predict_proba(X_test)[:, 1]

        metrics = {
            "Accuracy": accuracy_score(y_test, y_pred),
            "AUC": roc_auc_score(y_test, y_proba),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1": f1_score(y_test, y_pred, zero_division=0),
            "MCC": matthews_corrcoef(y_test, y_pred),
        }
        results[name] = metrics
        joblib.dump(pipe, MODEL_DIR / file_names[name])
        print(f"{name:25s} -> {metrics}")

    with open(MODEL_DIR / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\nSaved pipelines to:", MODEL_DIR)
    print("Saved test_data.csv to:", BASE_DIR / "test_data.csv")


if __name__ == "__main__":
    main()
