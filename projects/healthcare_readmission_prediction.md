# Healthcare Readmission Prediction

## Overview
This project focuses on predicting hospital readmissions for patients, helping healthcare providers proactively manage high-risk cases. By analyzing patient records, demographic data, and medical history, the system flags individuals who are more likely to require readmission, allowing clinicians to provide timely interventions.

## Problem Statement
Hospital readmissions are costly and can indicate gaps in patient care. Reducing preventable readmissions improves patient outcomes and reduces healthcare costs. This project addresses the challenge by building a predictive model that identifies at-risk patients before discharge.

## Workflow
1. **Data Collection & Cleaning**  
   - Gathered patient records, clinical data, and treatment histories.  
   - Handled missing values, outliers, and categorical encoding using `Pandas` and `NumPy`.

2. **Exploratory Data Analysis (EDA)**  
   - Visualized trends, correlations, and distribution of key features with `Matplotlib` and `Seaborn`.  
   - Identified factors strongly associated with readmission risk.

3. **Model Development**  
   - Built multiple machine learning models including `XGBoost` and `Random Forest`.  
   - Experimented with deep learning models using `TensorFlow` and `Keras` for sequence-based patient data.  
   - Used `Scikit-learn` for preprocessing, feature selection, and evaluation.

4. **Model Evaluation**  
   - Evaluated models based on precision, recall, F1-score, and ROC-AUC.  
   - Selected the best-performing model for deployment based on balanced predictive performance.

5. **Experiment Tracking**  
   - Tracked experiments, hyperparameters, and model versions with `Weights & Biases`.

6. **Deployment & Demo**  
   - Created a platform-independent web demo to visualize predictions and risk scores.  
   - Users can input patient features and receive an estimated readmission risk in real time.

## Tech Stack
- **Programming:** Python  
- **Machine Learning & Deep Learning:** Scikit-learn, XGBoost, TensorFlow, Keras  
- **Data Processing:** Pandas, NumPy  
- **Visualization:** Matplotlib, Seaborn  
- **Experiment Tracking:** Weights & Biases  

## User Interaction
- Upload patient data or enter individual patient details.  
- Receive a readmission risk score and recommended follow-up priority.  
- Visualize feature importance and contributing risk factors.

## Impact
- Supports proactive patient care and intervention strategies.  
- Helps healthcare providers optimize resources and reduce preventable readmissions.  
- Provides interpretable insights into key factors influencing readmission risk.
