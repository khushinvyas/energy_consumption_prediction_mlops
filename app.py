import os
import pandas as pd
import joblib
from flask import Flask, render_template, request
import numpy as np
import yaml
import boto3
from botocore.exceptions import BotoCoreError, ClientError

# Initialize Flask app
app = Flask(__name__)

# Load params and trained model
with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

features = params['train']['features']
model_path = "models/model.pkl"
model = None
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    # Try download from S3 if configured
    s3_bucket = os.getenv('S3_BUCKET')
    s3_key = os.getenv('S3_MODEL_KEY', 'models/model.pkl')
    if s3_bucket:
        try:
            os.makedirs('models', exist_ok=True)
            s3 = boto3.client('s3', region_name=os.getenv('AWS_REGION'))
            s3.download_file(s3_bucket, s3_key, model_path)
            model = joblib.load(model_path)
        except (BotoCoreError, ClientError, FileNotFoundError):
            model = None

@app.route('/')
def home():
    """Render the home page with the input form"""
    return render_template('index.html', features=features)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle the prediction request"""
    # Get the input values from the form
    input_values = []
    for feature in features:
        value = request.form.get(feature, None)
        value = float(value) if value is not None and value != '' else 0.0
        input_values.append(value)
    
    # Convert to numpy array and reshape for model
    input_array = np.array(input_values).reshape(1, -1)
    
    # Make prediction
    if model is None:
        return render_template('index.html', 
                              prediction_text='Model not found. Please train and mount models/model.pkl.',
                              input_values=dict(zip(features, input_values)),
                              features=features)
    prediction = model.predict(input_array)[0]
    
    # Return the prediction to the user
    return render_template('index.html', 
                          prediction_text=f'Predicted Global Active Power: {prediction:.4f} kW',
                          input_values=dict(zip(features, input_values)),
                          features=features)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)