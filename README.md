# Plant Disease Detector - Frontend & Backend Integration Plan

## Information Gathered:
- ML Project: Plant Disease Detection using TensorFlow Lite
- Models: MobileNetV2.tfliteQuant, DenseNet169.tfliteQuant (in project)
- Labels: 38 plant disease classes in Labels.txt
- Test images available in test/ folder

## Implementation Plan:

### 1. Backend (Flask) - app.py
- Flask server setup
- Load TFLite model on startup
- POST /predict endpoint - accepts image file, returns classification
- Image preprocessing (resize to 224x224, normalize)
- Return top prediction with confidence score

### 2. Frontend (HTML/CSS/JS) - templates/index.html
- Clean, modern UI with plant-themed design
- Image upload functionality
- Drag & drop support
- Display prediction results with:
  - Disease name
  - Confidence percentage
  - Visual indicator
- Loading state during prediction

### 3. Dependencies - requirements.txt
- flask
- tensorflow
- pillow
- numpy

## Files to Create:
1. app.py - Backend Flask application
2. templates/index.html - Frontend interface  
3. requirements.txt - Python dependencies

## Followup Steps:
1. Install dependencies: pip install -r requirements.txt
2. Run backend: python app.py
3. Open browser to http://localhost:5000
4. Upload test image and verify prediction works
