"""
ML Assignment 2 - Streamlit demo app.
Loads pre-trained classifier pipelines (saved by train.py) and scores an
uploaded test CSV, WITHOUT retraining, showing evaluation metrics and a
confusion matrix / classification report for the selected model.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score,
    recall_score, f1_score, matthews_corrcoef,
    confusion_matrix, classification_report,
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
TARGET_COL = "credit_risk"

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

st.set_page_config(page_title="German Credit Risk - Classifier Demo", layout="wide")

st.title("German Credit Risk - Multi-Model Classifier Demo")
st.caption(
    "UCI Statlog (German Credit Data) - 1000 instances, 20 features, binary target "
    "(1 = good credit risk, 0 = bad credit risk). Upload a test CSV to score it "
    "against a pre-trained model (no retraining happens here)."
)

with st.sidebar:
    st.header("1. Upload test data")
    uploaded_file = st.file_uploader("Upload test CSV (must include the 'credit_risk' target column)", type=["csv"])
    st.caption("A ready-made `test_data.csv` (20% held-out split) ships with this repo.")

    st.header("2. Choose model")
    model_choice = st.selectbox("Classification model", list(MODEL_FILES.keys()))

    st.header("3. Compare all models")
    show_all = st.checkbox("Also show comparison table across all 5 models", value=True)


@st.cache_resource
def load_model(name: str):
    return joblib.load(MODEL_DIR / MODEL_FILES[name])


def evaluate(pipe, X: pd.DataFrame, y: pd.Series) -> dict:
    y_pred = pipe.predict(X)
    y_proba = pipe.predict_proba(X)[:, 1]
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_proba),
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }, y_pred


if uploaded_file is None:
    st.info("Upload a test CSV from the sidebar to begin (or use the repo's `test_data.csv`).")
    st.stop()

df = pd.read_csv(uploaded_file)

if TARGET_COL not in df.columns:
    st.error(f"Uploaded CSV must contain the target column '{TARGET_COL}'.")
    st.stop()

X_test = df.drop(columns=[TARGET_COL])
y_test = df[TARGET_COL]

st.subheader(f"Results - {model_choice}")
pipe = load_model(model_choice)
metrics, y_pred = evaluate(pipe, X_test, y_test)

cols = st.columns(6)
for col, (metric_name, value) in zip(cols, metrics.items()):
    col.metric(metric_name, f"{value:.3f}")

left, right = st.columns(2)

with left:
    st.markdown("**Confusion Matrix**")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=["Actual: Bad (0)", "Actual: Good (1)"],
        columns=["Pred: Bad (0)", "Pred: Good (1)"],
    )
    st.dataframe(cm_df, use_container_width=True)

with right:
    st.markdown("**Classification Report**")
    report = classification_report(y_test, y_pred, target_names=["Bad (0)", "Good (1)"], output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose().round(3), use_container_width=True)

if show_all:
    st.subheader("Comparison across all 5 models (on the same uploaded test data)")
    rows = []
    for name in MODEL_FILES:
        m_pipe = load_model(name)
        m_metrics, _ = evaluate(m_pipe, X_test, y_test)
        rows.append({"Model": name, **{k: round(v, 3) for k, v in m_metrics.items()}})
    st.dataframe(pd.DataFrame(rows).set_index("Model"), use_container_width=True)

st.caption("Models were trained offline in `train.py` on an 80/20 stratified split and saved as scikit-learn "
           "pipelines (preprocessing + classifier) in `model/`. This app only loads and scores them.")
