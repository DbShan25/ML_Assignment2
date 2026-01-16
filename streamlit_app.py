# =====================================================
# ML Assignment 2 – Streamlit App (Cloud Safe)
# Employee Attrition Classification
# =====================================================

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    ConfusionMatrixDisplay,
    classification_report
)

# -----------------------------------------------------
# Page configuration (FIRST Streamlit call)
# -----------------------------------------------------
st.set_page_config(
    page_title="Employee Attrition Prediction",
    layout="wide"
)

st.title("🏢 Employee Attrition – ML Model Demonstration")
st.write("✅ App initialized successfully")

# -----------------------------------------------------
# Path setup (Cloud safe)
# -----------------------------------------------------
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "model"

# -----------------------------------------------------
# Load scaler safely
# -----------------------------------------------------
try:
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    st.success("Scaler loaded")
except Exception as e:
    st.error(f"Scaler loading failed: {e}")
    st.stop()

# -----------------------------------------------------
# Model selection
# -----------------------------------------------------
model_map = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "KNN": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl"
}

selected_model = st.selectbox("Select ML Model", list(model_map.keys()))

# -----------------------------------------------------
# Load selected model safely
# -----------------------------------------------------
try:
    model_path = MODEL_DIR / model_map[selected_model]
    model = joblib.load(model_path)
    st.success(f"{selected_model} loaded")
except Exception as e:
    st.error(f"Model loading failed: {e}")
    st.stop()

# -----------------------------------------------------
# Dataset upload
# -----------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload Test Dataset (CSV only, must contain Attrition column)",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("Dataset uploaded successfully")

    if "Attrition" not in df.columns:
        st.error("CSV file must contain 'Attrition' column")
        st.stop()

    X = df.drop("Attrition", axis=1)
    y = df["Attrition"]

    # Scale features
    X_scaled = scaler.transform(X)

    # Predictions
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]

    # Metrics
    acc = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_prob)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    mcc = matthews_corrcoef(y, y_pred)

    # -------------------------------------------------
    # Metrics + Confusion Matrix
    # -------------------------------------------------
    st.subheader("📊 Evaluation Results")

    left_col, right_col = st.columns([1, 1.3])

    with left_col:
        st.metric("Accuracy", f"{acc:.4f}")
        st.metric("AUC", f"{auc:.4f}")
        st.metric("Precision", f"{prec:.4f}")
        st.metric("Recall", f"{rec:.4f}")
        st.metric("F1 Score", f"{f1:.4f}")
        st.metric("MCC", f"{mcc:.4f}")

    with right_col:
        fig, ax = plt.subplots(figsize=(3, 3))
        ConfusionMatrixDisplay.from_predictions(
            y, y_pred, ax=ax, colorbar=False, values_format="d"
        )
        ax.set_title("Confusion Matrix", fontsize=9)
        st.pyplot(fig)

    # -------------------------------------------------
    # Classification Report (Table)
    # -------------------------------------------------
    st.subheader("📄 Classification Report")

    report_df = pd.DataFrame(
        classification_report(y, y_pred, output_dict=True)
    ).transpose().round(3)

    st.dataframe(report_df, use_container_width=True)

else:
    st.info("Please upload a CSV file to evaluate the selected model.")

# -----------------------------------------------------
# Footer
# -----------------------------------------------------
st.markdown(
    """
    ---
    **ML Assignment 2 – BITS Pilani (WILP)**  
    Employee Attrition Prediction using Multiple Classification Models
    """
)
