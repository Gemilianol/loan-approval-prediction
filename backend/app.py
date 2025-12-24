from flask import Flask, request, jsonify
# Needed for cross-origin requests during development
from flask_cors import CORS
import pandas as pd
import numpy as np
import json
from src.components.model_predict import search_model_uri, load_mlflow_model
from src.pipelines.training_pipeline import train_model_pipeline
from src.pipelines.predict_pipeline import predict_pipeline
from src.utils.logger import logger

# Create a Flask application instance
app = Flask(__name__)

# Enable CORS for all routes (important for development)
# At the beginning, Flask and React apps will be running 
# on localhost but in differents ports so CORS allows 
# to communicate each other.
CORS(app)

# # --- Essential CORS Configuration ---
# # 1. Allow ALL origins for simplicity (good for initial local development)
# # CORS(app)

# # 2. **BEST PRACTICE for local development:** #    Only allow your specific React frontend origin
# CORS(app, resources={r"/predict": {"origins": "http://localhost:5173"}}) 
# # or for all routes:
# # CORS(app, origins="http://localhost:5173")

# Minimal logging + health endpoint for K8s
logger.info("Starting app...")

# Cache the model loaded.
MODEL_URI = search_model_uri()
_MODEL = load_mlflow_model(MODEL_URI)

if _MODEL is None:
    train_model_pipeline(force_retrain=True)
    MODEL_URI = search_model_uri()
    _MODEL = load_mlflow_model(MODEL_URI)

@app.route('/health', methods=['GET'])
def health():
    """
    Health endpoint.

    Returns:
        JSON: JSON object + Status code as int.
    """
    return jsonify({'status': 'ok'}), 200
    
@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint to predict the input data (X) received from the user through the Frontend.
    
    Returns:
        JSON: JSON object with the prediction. 
    """
    
    global _MODEL
    
    # Handle preflight request
    # if request.method == 'OPTIONS':
    #     return jsonify({'status': 'ok'}), 200
    
    try:
        # Parses the JSON data sent from the frontend into a Python dictionary.
        data = request.get_json()
        # Safeguard
        if not data:
            return jsonify({'Error': 'No data received'}), 400
    
        pred = predict_pipeline(data, _MODEL)
        
        return jsonify({'Result': pred})
        
        # return jsonify({'Result': ['Approved' if pred[0]== 1 else 'Rejected']})
    
    except Exception as e:
        logger.debug('Something happened through the prediction process => %s', e)
        raise RuntimeError(f'Something happened through the prediction process => {e}') from e
    
if __name__ == '__main__':
    app.run(port=2000, debug=True)