# Satellite Image Analysis

## Overview
This project focuses on analyzing satellite imagery to monitor crop health, detect deforestation, and track environmental changes. By leveraging deep learning and computer vision techniques, the system helps in making data-driven decisions for agriculture and environmental management.

## Workflow
1. **Data Acquisition:** Satellite images were collected from publicly available sources and Azure storage services.  
2. **Preprocessing:** Images were cleaned, normalized, and resized. Cloud cover and noise were filtered out using OpenCV.  
3. **Model Training:**  
   - Deep learning models (CNNs) were trained using TensorFlow and PyTorch.  
   - Keras was used for building and fine-tuning model architectures efficiently.  
   - Weights & Biases tracked experiments, metrics, and model performance.  
4. **Prediction & Analysis:**  
   - Crop health was assessed using vegetation indices extracted from images.  
   - Deforestation and land-use changes were detected via segmentation models.  
5. **Deployment:** The analysis pipeline was deployed on Azure Cloud for scalable and automated inference.

## Technology Stack
- **Languages & Frameworks:** Python, TensorFlow, Keras, PyTorch  
- **Image Processing:** OpenCV  
- **Cloud Services:** Azure Cognitive Services, Azure Storage  
- **Experiment Tracking:** Weights & Biases  

## User Interaction
- Users can upload satellite imagery through a simple web interface.  
- The system provides visualizations highlighting areas of concern (e.g., crop stress or deforested regions).  
- Reports and actionable insights can be downloaded for further analysis.

## Impact
- Supported better monitoring of agricultural fields, enabling timely intervention.  
- Assisted environmental agencies in tracking deforestation and land-use changes.  
- Provided a scalable, automated solution for large-scale satellite image analysis.
