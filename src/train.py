# train.py
import pandas as pd
import numpy as np
import joblib
import os
import yaml
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
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

def train_model(processed_data_dir, model_output_dir, params):
    """
    Loads processed data, trains a model, and saves it.
    """
    # Load parameters
    model_name = params['train']['model']['name']
    model_params = params['train']['model']['params']
    target_column = params['preprocess']['target_column'] # Need target column for features later

    logging.info(f"Training model: {model_name} with parameters: {model_params}")

    # Load data
    X_train = pd.read_csv(os.path.join(processed_data_dir, 'X_train.csv'), index_col='datetime', parse_dates=True)
    y_train = pd.read_csv(os.path.join(processed_data_dir, 'y_train.csv'), index_col='datetime', parse_dates=True)
    
    # Ensure y_train is a Series, not a DataFrame
    if isinstance(y_train, pd.DataFrame) and len(y_train.columns) == 1:
        y_train = y_train.iloc[:, 0]
    elif isinstance(y_train, pd.DataFrame) and target_column in y_train.columns:
        y_train = y_train[target_column]
    else:
        logging.error(f"Could not correctly load y_train as a Series. Expected column: {target_column}")
        exit(1)


    # Initialize model
    if model_name == "RandomForestRegressor":
        model = RandomForestRegressor(**model_params)
    elif model_name == "XGBoostRegressor":
        model = XGBRegressor(**model_params)
    else:
        logging.error(f"Unsupported model type: {model_name}")
        exit(1)

    # Train model
    model.fit(X_train, y_train)
    logging.info(f"{model_name} training complete.")

    # Create output directory if it doesn't exist
    os.makedirs(model_output_dir, exist_ok=True)

    # Save model
    model_path = os.path.join(model_output_dir, 'model.pkl')
    joblib.dump(model, model_path)
    logging.info(f"Model saved to {model_path}")

if __name__ == "__main__":
    params = load_params()
    processed_data_directory = 'data/processed'
    model_output_directory = 'models'
    train_model(processed_data_directory, model_output_directory, params)