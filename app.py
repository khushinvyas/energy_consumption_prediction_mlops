import os
import pandas as pd
import joblib
from flask import Flask, render_template, request
import numpy as np

# Initialize Flask app
app = Flask(__name__)

# Load the trained model
model_path = "models/model.joblib"
model = joblib.load(model_path)

# Define the features expected by the model
features = [
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3"
]

@app.route('/')
def home():
    """Render the home page with the input form"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle the prediction request"""
    # Get the input values from the form
    input_values = []
    for feature in features:
        value = float(request.form[feature])
        input_values.append(value)
    
    # Convert to numpy array and reshape for model
    input_array = np.array(input_values).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(input_array)[0]
    
    # Return the prediction to the user
    return render_template('index.html', 
                          prediction_text=f'Predicted Global Active Power: {prediction:.4f} kW',
                          input_values=dict(zip(features, input_values)))

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)