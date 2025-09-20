# License Plate Recognition with Context

## Overview
This project automates toll and parking billing by detecting and recognizing vehicle license plates. Beyond simple plate recognition, the system incorporates contextual information, such as vehicle type and orientation, to improve accuracy and reduce misidentification in real-world scenarios.

## Workflow
1. **Data Acquisition**  
   - Collected images and video feeds from toll booths and parking entrances.  
   - Annotated license plates and vehicle metadata for training.  

2. **Preprocessing**  
   - Applied image enhancements such as resizing, normalization, and noise reduction using OpenCV.  
   - Extracted relevant regions of interest (license plates) for model input.  

3. **Model Training**  
   - Leveraged a combination of TensorFlow and PyTorch models for license plate detection and OCR.  
   - Used Weights & Biases to track experiments, monitor model performance, and fine-tune hyperparameters.  

4. **Contextual Recognition**  
   - Integrated vehicle attributes (type, orientation, color) to reduce false positives.  
   - Combined multiple frames for improved plate reading accuracy in motion.  

5. **Deployment**  
   - Developed a cross-platform solution that can run on cloud servers or edge devices.  
   - Supports real-time inference for toll booths or parking facilities.  

## Tech Stack
- **Python** – core programming and data processing  
- **OpenCV** – image processing and plate extraction  
- **TensorFlow & PyTorch** – detection and recognition models  
- **Weights & Biases** – experiment tracking and model optimization  

## User Interaction
- Users can upload images or stream live video to the system.  
- The system automatically identifies license plates and associated vehicle information.  
- Recognized plates are logged with timestamps for billing or access control purposes.  
- Optionally, results can be visualized through a web dashboard or exported to external systems.  

## Impact
- Reduces manual monitoring and billing errors.  
- Improves throughput at toll booths and parking facilities.  
- Provides accurate vehicle identification even in challenging conditions, such as low light or motion blur.  