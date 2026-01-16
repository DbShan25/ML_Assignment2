# =====================================================
# ML Assignment 2 – Streamlit App
# Employee Attrition Classification
# =====================================================

import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

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
# Page configuration
# -----------------------------------------------------
st.set_page_config(
    page_title="Employee Attrition Prediction",
    layout="wide"
)

st.title("🏢 Employee Attrition – ML Model Demonstration")

# -----------------------------------------------------
# Load scaler
# -----------------------------------------------------
@st.cache_resource
def load_scaler():
    return joblib.load("model/scaler.pkl")

scaler = load_scaler()

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

selected_model = st.selectbox(
    "Select ML Model",
    list(model_map.keys())
)

@st.cache_resource
def load_model(model_file):
    return joblib.load(f"model/{model_file}")

model = load_model(model_map[selected_model])

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
    else:
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
        # Metrics + Confusion Matrix (Side-by-Side)
        # -------------------------------------------------
        st.subheader("📊 Evaluation Results")

        left_col, right_col = st.columns([1, 1.3])

        # LEFT: Metrics (Vertical)
        with left_col:
            st.markdown("### 📈 Evaluation Metrics")
            st.metric("Accuracy", f"{acc:.4f}")
            st.metric("AUC", f"{auc:.4f}")
            st.metric("Precision", f"{prec:.4f}")
            st.metric("Recall", f"{rec:.4f}")
            st.metric("F1 Score", f"{f1:.4f}")
            st.metric("MCC", f"{mcc:.4f}")

        # RIGHT: Compact Confusion Matrix
        with right_col:
            st.markdown("### 📌 Confusion Matrix")

            fig, ax = plt.subplots(figsize=(3, 3))
            ConfusionMatrixDisplay.from_predictions(
                y,
                y_pred,
                ax=ax,
                colorbar=False,
                values_format="d"
            )

            ax.set_title("Confusion Matrix", fontsize=9)
            ax.set_xlabel("")
            ax.set_ylabel("")
            ax.tick_params(axis="both", labelsize=7)

            for text in ax.texts:
                text.set_fontsize(10)

            st.pyplot(fig, use_container_width=False)

        # -------------------------------------------------
        # Classification Report (TABLE FORMAT – SAME PAGE)
        # -------------------------------------------------
        st.subheader("📄 Classification Report")

        report_dict = classification_report(y, y_pred, output_dict=True)
        report_df = pd.DataFrame(report_dict).transpose().round(3)

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
