# Quality Control System

**Domain:** Computer Vision  
**Platform:** Azure  
**Tech Stack:** Python, OpenCV, PyTorch, TensorFlow, Azure Cognitive Services, Azure ML Studio  

---

## Project Overview

The **Quality Control System** is an AI-powered solution designed to automate defect detection in manufacturing pipelines. It helps companies improve product quality, reduce waste, and minimize manual inspection efforts. Using computer vision techniques, the system identifies anomalies in real-time, allowing timely intervention and better decision-making on the production floor.

---

## Workflow

1. **Data Collection**  
   High-resolution images and videos are captured from production lines using industrial cameras. These images are labeled to identify defects such as scratches, dents, or misalignments.

2. **Data Preprocessing**  
   - Image resizing, normalization, and augmentation.  
   - Noise reduction using OpenCV filters.  

3. **Model Development**  
   - A convolutional neural network (CNN) is trained using PyTorch and TensorFlow.  
   - Transfer learning is applied to speed up training and improve accuracy.  
   - Model is validated using a test dataset to ensure robust defect detection.

4. **Deployment**  
   - The trained model is deployed using **Azure ML Studio**.  
   - **Azure Cognitive Services** is integrated to handle real-time inference and scalability.  
   - A web-based demo allows users to upload images and see defect predictions in seconds.

---

## User Interaction

- Users can upload images or video snippets of the production line.  
- The system highlights detected defects and provides a confidence score for each prediction.  
- Reports can be exported to track quality metrics over time.

---

## Impact

- Reduced manual inspection workload by automating repetitive quality checks.  
- Early detection of defects prevents defective products from reaching customers.  
- Streamlined production process with improved overall efficiency.