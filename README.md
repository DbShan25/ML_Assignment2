# ML Assignment 2 – Employee Attrition Classification
Machine Learning Assignment 2 – Employee Attrition Classification using multiple ML models with Streamlit deployment (BITS Pilani WILP).

## Problem Statement
Employee attrition is a critical challenge for organizations as it leads to increased hiring costs and loss of skilled talent. The objective of this project is to build and evaluate multiple machine learning classification models to predict employee attrition.

## Dataset Description
The Employee Attrition dataset is sourced from Kaggle and contains employee demographic, job-related, and performance-related attributes.  
- Number of instances: > 1000  
- Number of features: > 30  
- Target variable: Attrition (Yes / No)

## Models Used and Evaluation Metrics
The following six classification models were implemented and evaluated using Accuracy, AUC, Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC):

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|------|---------|-----|----------|--------|----|-----|
| Logistic Regression | 0.698 | 0.753 | 0.716 | 0.696 | 0.706 | 0.397 |
| Decision Tree | 0.812 | 0.811 | 0.818 | 0.821 | 0.819 | 0.623 |
| KNN | 0.745 | 0.818 | 0.719 | 0.837 | 0.773 | 0.493 |
| Naive Bayes | 0.664 | 0.749 | 0.656 | 0.743 | 0.697 | 0.326 |
| Random Forest | 0.885 | 0.941 | 0.931 | 0.840 | 0.883 | 0.774 |
| XGBoost | 0.877 | 0.944 | 0.905 | 0.852 | 0.878 | 0.755 |

## Model Performance Observations

| Model | Observation |
|------|-------------|
| Logistic Regression | Shows moderate performance, indicating linear boundaries are insufficient for capturing complex attrition patterns. |
| Decision Tree | Captures non-linear relationships effectively but is slightly prone to overfitting compared to ensembles. |
| KNN | Provides good recall but lower precision due to sensitivity to local neighborhoods. |
| Naive Bayes | Lower performance due to independence assumptions among features. |
| Random Forest | Best performing model with highest Accuracy, AUC, F1, and MCC. |
| XGBoost | Comparable to Random Forest with strong generalization and balanced metrics. |

## Streamlit Deployment
An interactive Streamlit application was developed to demonstrate model predictions, evaluation metrics, and confusion matrix visualization. The application allows users to upload test data and select different ML models.

