# Credit Risk Scoring System

## Overview
The Credit Risk Scoring System is a machine learning project designed to predict the likelihood of loan defaults. By leveraging explainable AI techniques, this system provides actionable insights for financial institutions to make informed lending decisions while mitigating risk.

## Problem Statement
Financial institutions need to evaluate the creditworthiness of applicants quickly and accurately. Traditional credit scoring methods may miss subtle patterns in customer data, potentially leading to higher default rates. This project aims to enhance the decision-making process using data-driven predictions and explainability.

## Workflow
1. **Data Collection & Processing**  
   - Aggregated historical loan and customer data from various sources.  
   - Cleaned and transformed data using **Pandas** and **Databricks Delta Lake** for efficient storage and processing.

2. **Feature Engineering**  
   - Created meaningful features such as debt-to-income ratio, credit history length, and payment patterns.  
   - Encoded categorical variables and normalized numerical features for modeling.

3. **Model Development**  
   - Built multiple classification models using **Scikit-learn**, including logistic regression, random forest, and gradient boosting.  
   - Optimized model performance using **MLflow** for experiment tracking and **Weights & Biases** for hyperparameter tuning.

4. **Model Evaluation & Explainability**  
   - Evaluated models using metrics like accuracy, precision, recall, and ROC-AUC.  
   - Applied **SHAP** values to explain model predictions and highlight the most influential features.

5. **Deployment & Monitoring**  
   - Registered the best-performing model in **Databricks Model Registry**.  
   - Enabled version tracking and monitoring for continuous improvement.  
   - Deployed a demo interface to visualize risk scores and feature contributions.

## Technology Stack
- **Programming & Data Processing:** Python, Pandas  
- **Machine Learning:** Scikit-learn, SHAP  
- **Platform & MLOps:** Databricks, MLflow, Model Registry, Delta Lake  
- **Experiment Tracking & Monitoring:** Weights & Biases  

## User Interaction
- Users (financial analysts or loan officers) can input applicant data and receive a credit risk score.  
- The system provides an **explainability report**, showing which factors contributed most to the prediction, improving trust and transparency.  

## Impact
- Enables faster, data-driven loan decisions.  
- Reduces the risk of defaults through predictive insights.  
- Provides actionable transparency via explainable AI, improving stakeholder confidence.