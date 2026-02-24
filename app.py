"""
Plant Disease Detector - Flask Backend
This server provides an API for plant disease detection using TensorFlow Lite models.
"""

import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from PIL import Image
import tensorflow as tf
import io

app = Flask(__name__)

# Configuration
MODEL_PATH = 'MobileNetV2.tfliteQuant'
LABELS_PATH = 'Labels.txt'
IMAGE_SIZE = 224

# Load labels
def load_labels():
    labels = []
    with open(LABELS_PATH, 'r') as f:
        for line in f:
            labels.append(line.strip())
    return labels

# Load TFLite model
def load_model():
    interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    return interpreter

# Preprocess image
def preprocess_image(image, target_size=(IMAGE_SIZE, IMAGE_SIZE)):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image = image.resize(target_size)
    image_array = np.array(image, dtype=np.float32)
    image_array = image_array / 255.0  # Normalize to [0, 1]
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Make prediction
def predict(interpreter, image_array):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    interpreter.set_tensor(input_details[0]['index'], image_array)
    interpreter.invoke()
    
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0]

# Global variables
interpreter = None
labels = None

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict_api():
    """API endpoint for disease prediction"""
    global interpreter, labels
    
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'No image selected'}), 400
    
    try:
        # Read and preprocess image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes))
        image_array = preprocess_image(image)
        
        # Make prediction
        predictions = predict(interpreter, image_array)
        
        # Get top prediction
        top_index = np.argmax(predictions)
        top_confidence = float(predictions[top_index])
        top_label = labels[top_index]
        
        # Get top 3 predictions for better UX
        top_3_indices = np.argsort(predictions)[-3:][::-1]
        top_3_predictions = [
            {
                'label': labels[i],
                'confidence': float(predictions[i])
            }
            for i in top_3_indices
        ]
        
        return jsonify({
            'success': True,
            'prediction': {
                'label': top_label,
                'confidence': top_confidence,
                'all_predictions': top_3_predictions
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': interpreter is not None,
        'labels_loaded': labels is not None
    })

if __name__ == '__main__':
    print("Loading model and labels...")
    try:
        labels = load_labels()
        print(f"Loaded {len(labels)} labels")
        
        interpreter = load_model()
        print(f"Loaded model from {MODEL_PATH}")
        
        print("Starting Flask server...")
        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        print(f"Error starting server: {e}")
        print("Make sure you have installed the dependencies and the model file exists.")

