# ML Assignment 2 – Employee Attrition Classification

Machine Learning Assignment 2 – Employee Attrition Classification using multiple ML models with Streamlit deployment  
(**BITS Pilani WILP**).

---

## Problem Statement
Employee attrition is a critical challenge for organizations as it leads to increased hiring costs and loss of skilled talent.  
The objective of this project is to build, evaluate, and compare multiple machine learning classification models to predict employee attrition.

---

## Dataset Description
The Employee Attrition dataset is sourced from Kaggle and contains employee demographic, job-related, and performance-related attributes.

- Number of instances: > 1000  
- Number of features: > 30  
- Target variable: **Attrition (Yes / No)**  

---

## Data Preparation and Model Training
- Data preprocessing and exploratory analysis were performed using a Jupyter Notebook (`Employee_Attrition.ipynb`).
- Categorical features were encoded and numerical features were standardized using a scaler.
- The dataset was split into training and test sets to ensure unbiased evaluation.
- Six classification models were trained on the same training data for fair comparison.
- Trained models and the scaler were saved as `.pkl` files using `joblib` for reuse during deployment.

---

## Models Used and Evaluation Metrics
The following models were evaluated using **Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC)**:

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|------|---------|-----|----------|--------|----|-----|
| Logistic Regression | 0.698 | 0.753 | 0.716 | 0.696 | 0.706 | 0.397 |
| Decision Tree | 0.812 | 0.811 | 0.818 | 0.821 | 0.819 | 0.623 |
| KNN | 0.745 | 0.818 | 0.719 | 0.837 | 0.773 | 0.493 |
| Naive Bayes | 0.664 | 0.749 | 0.656 | 0.743 | 0.697 | 0.326 |
| Random Forest | 0.885 | 0.941 | 0.931 | 0.840 | 0.883 | 0.774 |
| XGBoost | 0.877 | 0.944 | 0.905 | 0.852 | 0.878 | 0.755 |

---

## Model Performance Observations
- **Logistic Regression** shows moderate performance, indicating linear decision boundaries are insufficient for capturing complex attrition patterns.
- **Decision Tree** captures non-linear relationships effectively but shows slightly lower generalization compared to ensemble models.
- **KNN** achieves high recall but lower precision due to sensitivity to local data distributions.
- **Naive Bayes** demonstrates lower performance due to strong feature independence assumptions.
- **Random Forest** achieves the best overall performance with the highest Accuracy, AUC, F1 Score, and MCC.
- **XGBoost** shows performance comparable to Random Forest with well-balanced metrics and strong generalization.

---

## Streamlit Deployment
An interactive Streamlit application was developed to demonstrate model predictions and evaluation results.

The application allows users to:
- Select one of the trained ML models
- Upload a test dataset (CSV format)
- View evaluation metrics
- Visualize the confusion matrix and classification report

Trained models and the scaler are dynamically loaded during runtime.

---

## How to Run the Application
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py


## Required CSV Input Features
The uploaded CSV file must contain the following feature columns used during model training, along with the target variable for evaluation.

### Feature Columns:
- `Over_Time`
- `Stock_Option_Level`
- `Marital_Status`
- `Job_Satisfaction`
- `Monthly_Income`
- `Distance_From_Home`
- `Job_Involvement`
- `Years_in_Current_Role`

### Target Column:
- `Attrition`

> **Note:**  
> The uploaded CSV must include the above features in the same format as used during training.  
> The `Attrition` column is required for evaluation purposes to compute metrics such as Accuracy, AUC, Precision, Recall, F1-score, and MCC. The model does not use the target column during prediction.

