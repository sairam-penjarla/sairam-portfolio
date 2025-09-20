# Handwritten Form Reader

## Overview
The Handwritten Form Reader is an AI-powered solution designed to digitize handwritten forms, such as government or bank documents. By combining Optical Character Recognition (OCR) with deep learning models, the system can accurately extract structured data from scanned forms, reducing manual entry and increasing operational efficiency.

## Workflow
1. **Input Processing**  
   Users upload scanned or photographed forms in common image formats (JPEG, PNG, PDF).  
2. **Preprocessing**  
   Images are processed using OpenCV to enhance readability—noise reduction, skew correction, and contrast adjustment.  
3. **Text Extraction**  
   Tesseract OCR is used for initial text recognition, and deep learning models further refine and classify extracted data.  
4. **Data Structuring**  
   Recognized fields (e.g., name, date, ID numbers) are mapped into structured formats like CSV or JSON.  
5. **Validation & Review**  
   Optional manual verification can be performed for low-confidence fields, ensuring high accuracy.  
6. **Output**  
   Digitized data is ready for integration into databases or document management systems.

## Technology Stack
- **Programming:** Python  
- **Deep Learning:** TensorFlow, Keras  
- **Computer Vision:** OpenCV  
- **OCR:** Tesseract OCR  
- **Experiment Tracking:** Weights & Biases  
- **Document Processing:** Document Intelligence libraries for field recognition and structured extraction  

## User Interaction
- Simple interface to upload forms and download processed results  
- Visual feedback showing detected fields and confidence scores  
- Optional integration for batch processing large volumes of documents  

## Impact
- Significantly reduces time and errors in manual data entry  
- Facilitates digitization of paper-based workflows for banks, government offices, and enterprises  
- Provides scalable, cross-platform automation for document processing tasks  