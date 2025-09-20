# Retail Shelf Monitoring

## Overview
Retail Shelf Monitoring is a computer vision solution designed to help stores automatically detect stockouts, misplaced products, and compliance issues on retail shelves. The system provides store managers with actionable insights to improve inventory management and ensure product availability.

## Workflow
1. **Data Collection**  
   Images of store shelves are captured using standard cameras or mobile devices.
2. **Preprocessing**  
   Images are cleaned and standardized using OpenCV to correct lighting and orientation issues.
3. **Object Detection & Classification**  
   PyTorch and TensorFlow models identify products, detect missing items, and flag misplaced or incorrectly arranged products.
4. **Analysis & Reporting**  
   Insights are aggregated and visualized, highlighting stockouts and shelf compliance issues for store managers.
5. **Deployment**  
   The solution runs on Azure, leveraging Azure Computer Vision and ML Studio for model hosting, inference, and monitoring.

## Tech Stack
- **Programming & Libraries:** Python, OpenCV, PyTorch, TensorFlow  
- **Cloud & AI Services:** Azure Computer Vision, Azure ML Studio  
- **Experiment Tracking & Collaboration:** Weights & Biases  

## User Interaction
- Users can upload shelf images through a simple interface or integrate with store cameras.
- The system provides annotated images and dashboards showing stock levels and misplacements.
- Alerts are generated for low stock or compliance issues.

## Impact
- Reduces manual shelf auditing time by automating detection of stockouts and misplaced items.
- Improves inventory accuracy and product availability, enhancing customer satisfaction.
- Supports decision-making with visual dashboards and actionable insights.