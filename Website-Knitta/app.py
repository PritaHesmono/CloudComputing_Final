from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import pandas as pd
import logging

# Initialize Flask
app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.ERROR)

# Load the .pkl model
with open('model/model_lstm.pkl', 'rb') as file:
    model = pickle.load(file)

# Main page
@app.route('/')
def index():
    return render_template('predict.html')

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check for uploaded file
        file = request.files.get('dataset')
        if not file:
            return jsonify({'error': 'No file uploaded'}), 400
        
        # Read CSV data
        data = pd.read_csv(file)
        
        # Validate required column
        feature_column = 'feature_column'  # Replace with the correct column name
        if feature_column not in data.columns:
            return jsonify({'error': f'Missing required column: {feature_column}'}), 400
        
        # Preprocess data
        input_data = data[feature_column].values
        input_data = np.expand_dims(input_data, axis=0)  # Adjust this to match your model's expected input
        
        # Predict using the loaded .pkl model
        predictions = model.predict(input_data)  # Replace with model-specific prediction method
        output = predictions.tolist()
        
        return jsonify({'predictions': output})
    
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        return jsonify({'error': 'An error occurred during prediction'}), 500

if __name__ == '__main__':
    app.run(debug=True)