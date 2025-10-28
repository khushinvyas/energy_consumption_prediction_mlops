# evaluate.py
import pandas as pd
import joblib
import os
import json
import yaml
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_params():
    """Loads parameters from params.yaml."""
    try:
        with open('params.yaml', 'r') as f:
            params = yaml.safe_load(f)
        return params
    except FileNotFoundError:
        logging.error("params.yaml not found. Please ensure it's in the root directory.")
        exit(1)
    except yaml.YAMLError as e:
        logging.error(f"Error reading params.yaml: {e}")
        exit(1)

def evaluate_model(model_path, processed_data_dir, metrics_output_dir, params):
    """
    Loads model and test data, makes predictions, calculates and saves metrics.
    """
    logging.info(f"Loading model from {model_path}")
    model = joblib.load(model_path)

    logging.info(f"Loading test data from {processed_data_dir}")
    X_test = pd.read_csv(os.path.join(processed_data_dir, 'X_test.csv'), index_col='datetime', parse_dates=True)
    y_test = pd.read_csv(os.path.join(processed_data_dir, 'y_test.csv'), index_col='datetime', parse_dates=True)
    
    target_column = params['preprocess']['target_column']
    if isinstance(y_test, pd.DataFrame) and len(y_test.columns) == 1:
        y_test = y_test.iloc[:, 0]
    elif isinstance(y_test, pd.DataFrame) and target_column in y_test.columns:
        y_test = y_test[target_column]
    else:
        logging.error(f"Could not correctly load y_test as a Series. Expected column: {target_column}")
        exit(1)


    logging.info("Making predictions on the test set.")
    predictions = model.predict(X_test)

    # Calculate metrics
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)

    metrics = {
        'mae': mae,
        'rmse': rmse,
        'r2_score': r2
    }

    # Create output directory if it doesn't exist
    os.makedirs(metrics_output_dir, exist_ok=True)

    # Save metrics
    metrics_path = os.path.join(metrics_output_dir, 'metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)

    logging.info(f"Metrics calculated and saved to {metrics_path}")
    logging.info(f"MAE: {mae:.4f}, RMSE: {rmse:.4f}, R2 Score: {r2:.4f}")

if __name__ == "__main__":
    params = load_params()
    model_file = 'models/model.pkl'
    processed_data_directory = 'data/processed'
    metrics_output_directory = 'metrics'
    evaluate_model(model_file, processed_data_directory, metrics_output_directory, params)