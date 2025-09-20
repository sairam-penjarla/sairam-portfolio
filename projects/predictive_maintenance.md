# Predictive Maintenance for Industrial Equipment

## Overview
This project focuses on **predictive maintenance** for industrial equipment using IoT sensor data. By analyzing sensor readings and detecting anomalies, the system predicts potential failures before they occur. This helps organizations **reduce downtime, optimize maintenance schedules, and save costs**.

## Workflow
1. **Data Collection**  
   Gathered time-series sensor data from industrial machines, including temperature, vibration, and pressure readings.

2. **Data Preprocessing**  
   - Cleaned missing and noisy data.
   - Normalized sensor readings to improve model performance.
   - Generated features for anomaly detection, such as rolling averages and statistical metrics.

3. **Modeling**
   - Explored classical ML algorithms (e.g., Isolation Forest, Random Forest) for anomaly detection.
   - Built deep learning models (Autoencoders in PyTorch/TensorFlow) for complex pattern recognition.
   - Trained models to identify deviations from normal operational behavior.

4. **Evaluation**
   - Evaluated models using precision, recall, and F1-score.
   - Visualized anomalies and sensor trends using Matplotlib/Seaborn.
   - Monitored model training and experiments with Weights & Biases.

5. **Deployment & Demo**
   - Created a demo platform to visualize real-time anomaly alerts.
   - Users can explore sensor data, see predicted anomalies, and access maintenance recommendations.

## Tech Stack
- **Languages & Libraries:** Python, Pandas, NumPy, Scikit-learn, PyTorch, TensorFlow  
- **Experiment Tracking:** Weights & Biases  
- **Visualization:** Matplotlib, Seaborn  

## User Interaction
- Users can upload sensor data or stream data from IoT devices.  
- The platform highlights anomalies and suggests maintenance actions.  
- Provides intuitive dashboards to track equipment health over time.

## Impact
- Improved maintenance efficiency by predicting failures before they occur.  
- Helped reduce unexpected downtime, improving operational productivity.  
- Enabled data-driven decision-making for maintenance teams.