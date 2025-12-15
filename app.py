from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend

# Global variables for models
lstm_model = None
tokenizer = None
max_length = 200

def load_models():
    """Load the trained LSTM model and tokenizer"""
    global lstm_model, tokenizer
    
    try:
        # Load LSTM model
        lstm_model_path = 'lstm_model.keras'
        if os.path.exists(lstm_model_path):
            lstm_model = tf.keras.models.load_model(lstm_model_path)
            print(f"✓ Loaded LSTM model from {lstm_model_path}")
        else:
            raise FileNotFoundError(f"Model file not found: {lstm_model_path}")
        
        # Load tokenizer
        tokenizer_path = 'lstm_tokenizer.pkl'
        if os.path.exists(tokenizer_path):
            with open(tokenizer_path, 'rb') as f:
                tokenizer = pickle.load(f)
            print(f"✓ Loaded tokenizer from {tokenizer_path}")
        else:
            raise FileNotFoundError(f"Tokenizer file not found: {tokenizer_path}")
        
        print("✓ All models loaded successfully!")
        return True
    except Exception as e:
        print(f"Error loading models: {e}")
        return False

@app.route('/')
def home():
    """Health check endpoint"""
    return jsonify({
        'status': 'success',
        'message': 'Phishing Email Detection API is running',
        'model_loaded': lstm_model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Predict if an email is phishing or safe"""
    try:
        # Get email text from request
        data = request.get_json()
        
        if not data or 'email_text' not in data:
            return jsonify({
                'error': 'Missing email_text in request body'
            }), 400
        
        email_text = data['email_text']
        
        if not email_text or not isinstance(email_text, str):
            return jsonify({
                'error': 'email_text must be a non-empty string'
            }), 400
        
        # Check if models are loaded
        if lstm_model is None or tokenizer is None:
            return jsonify({
                'error': 'Models not loaded. Please check server logs.'
            }), 500
        
        # Preprocess email text
        # Convert to sequence
        sequence = tokenizer.texts_to_sequences([email_text])
        
        # Pad sequence
        padded_sequence = pad_sequences(
            sequence, 
            maxlen=max_length, 
            padding='post', 
            truncating='post'
        )
        
        # Make prediction
        prediction_proba = lstm_model.predict(padded_sequence, verbose=0)
        phishing_probability = float(prediction_proba[0][0])
        safe_probability = 1.0 - phishing_probability
        
        # Determine prediction
        is_phishing = phishing_probability > 0.5
        prediction_label = 'Phishing' if is_phishing else 'Safe'
        
        # Determine confidence level
        confidence = phishing_probability if is_phishing else safe_probability
        
        if confidence >= 0.9:
            confidence_level = 'Very High'
        elif confidence >= 0.75:
            confidence_level = 'High'
        elif confidence >= 0.6:
            confidence_level = 'Medium'
        else:
            confidence_level = 'Low'
        
        return jsonify({
            'prediction': prediction_label,
            'is_phishing': bool(is_phishing),
            'phishing_probability': round(phishing_probability, 4),
            'safe_probability': round(safe_probability, 4),
            'confidence': round(confidence, 4),
            'confidence_level': confidence_level
        })
    
    except Exception as e:
        return jsonify({
            'error': f'Prediction error: {str(e)}'
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': lstm_model is not None and tokenizer is not None
    })

if __name__ == '__main__':
    print("="*60)
    print("Loading Phishing Email Detection Models...")
    print("="*60)
    
    if load_models():
        print("\n" + "="*60)
        print("Starting Flask server...")
        print("="*60)
        print("API will be available at: http://localhost:5000")
        print("Frontend should connect to: http://localhost:5000/predict")
        print("="*60 + "\n")
        
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        print("Failed to load models. Please check the model files.")
        exit(1)

