# Medical Image Analysis

## Overview
This project focuses on **automated detection and classification of tumors** from MRI and CT scans. Using deep learning techniques, it assists radiologists by providing preliminary analysis, highlighting regions of interest, and classifying tumor types. The system is designed to improve accuracy and reduce diagnostic time without replacing expert judgment.

## Workflow
1. **Data Acquisition and Preprocessing**
   - Collected anonymized MRI and CT scan datasets.
   - Applied preprocessing steps such as normalization, resizing, and augmentation.
   - Utilized OpenCV for image enhancement and noise reduction.

2. **Model Development**
   - Built deep learning models using **TensorFlow** and **Keras** for classification tasks.
   - Experimented with **PyTorch** for alternate model architectures.
   - Implemented CNN-based architectures optimized for medical imaging.

3. **Training and Evaluation**
   - Trained models on Azure Machine Learning with GPU acceleration.
   - Monitored experiments and tracked metrics using **Weights & Biases**.
   - Evaluated performance with accuracy, precision, recall, and F1-score metrics.

4. **Deployment**
   - Packaged models for inference in Azure ML Studio.
   - Developed a web-based demo allowing users to upload scans and receive predictions.
   - Ensured compliance with privacy guidelines for medical data.

## Tech Stack
- **Languages:** Python  
- **Frameworks:** TensorFlow, Keras, PyTorch  
- **Tools & Platforms:** Azure Machine Learning, ML Studio, Weights & Biases  
- **Libraries:** OpenCV, scikit-learn  

## User Interaction
- Users can upload MRI or CT scans through the demo portal.
- The system highlights potential tumor regions and provides classification results.
- Offers qualitative confidence scores and visual overlays for better interpretability.

## Impact
- Accelerates initial tumor screening and provides radiologists with an additional layer of analysis.
- Demonstrates practical application of deep learning in medical imaging.
- Provides a foundation for future expansion into multi-modal medical image analysis.