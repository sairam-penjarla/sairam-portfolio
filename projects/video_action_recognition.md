# Video Action Recognition

## Overview
This project focuses on **enhancing workplace safety** by analyzing video feeds to detect unsafe behaviors in real time. Using advanced computer vision and deep learning techniques, the system identifies actions like slips, falls, or improper use of equipment, enabling timely interventions to prevent accidents.

## Key Features
- **Real-time behavior detection:** Monitors live or recorded video streams for unsafe actions.
- **Action classification:** Recognizes multiple types of workplace activities and flags anomalies.
- **Alert system:** Can be integrated with notifications or dashboards to alert supervisors.
- **Cross-platform compatibility:** Works on both web and desktop environments.

## Workflow
1. **Data Collection & Preprocessing**
   - Video footage of workplace activities is collected.
   - Frames are extracted and normalized for model input.
   - Data augmentation is applied to improve model robustness.

2. **Model Training**
   - Deep learning models are implemented using **PyTorch** and **TensorFlow/Keras**.
   - Pretrained architectures are fine-tuned for action recognition.
   - Training metrics and experiments are tracked using **Weights & Biases**.

3. **Inference & Deployment**
   - Video streams are processed using **OpenCV** for frame extraction and real-time analysis.
   - Detected actions are classified and flagged as safe/unsafe.
   - Optional dashboard or notification system displays alerts.

## Tech Stack
- **Programming Language:** Python  
- **Computer Vision:** OpenCV  
- **Deep Learning:** PyTorch, TensorFlow, Keras  
- **Experiment Tracking:** Weights & Biases  

## User Interaction
- Upload or stream video for analysis.
- View flagged actions in real time via a web interface or desktop application.
- Optionally, receive alerts for detected unsafe behaviors.

## Impact
- Helps **reduce workplace accidents** by enabling early detection of unsafe actions.
- Provides **data-driven insights** into employee behavior and safety compliance.
- Supports **safety training and preventive measures** through recorded video analysis.
