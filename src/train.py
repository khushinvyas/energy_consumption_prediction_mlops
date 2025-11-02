# src/train.py
import pandas as pd
import numpy as np
import joblib
import os
import yaml
import logging
import argparse # Import for command-line arguments
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor # Import new model
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_params():
    """Loads parameters from params.yaml."""
    with open('params.yaml', 'r') as f:
        return yaml.safe_load(f)

def get_model_instance(model_name, params):
    """Initializes a model instance based on its name and parameters."""
    if model_name == "RandomForestRegressor":
        return RandomForestRegressor(**params)
    elif model_name == "XGBoostRegressor":
        return XGBRegressor(**params)
    elif model_name == "LightGBMRegressor":
        return LGBMRegressor(**params)
    else:
        raise ValueError(f"Unsupported model type: {model_name}")

def perform_time_series_cross_validation(X, y, model_config):
    """Performs TCV for a single, specified model."""
    model_name = model_config['name']
    logging.info(f"--- Starting TCV for {model_name} ---")
    
    model = get_model_instance(model_name, model_config['params'])
    
    tscv = TimeSeriesSplit(n_splits=5) # Hardcoding 5 splits for simplicity
    mae_scores, r2_scores = [], []

    for i, (train_index, val_index) in enumerate(tscv.split(X)):
        X_train_fold, X_val_fold = X.iloc[train_index], X.iloc[val_index]
        y_train_fold, y_val_fold = y.iloc[train_index], y.iloc[val_index]
        
        model.fit(X_train_fold, y_train_fold)
        predictions = model.predict(X_val_fold)
        
        mae = mean_absolute_error(y_val_fold, predictions)
        r2 = r2_score(y_val_fold, predictions)
        mae_scores.append(mae)
        r2_scores.append(r2)
        logging.info(f"Fold {i+1}/5 -> MAE: {mae:.4f}, R2 Score: {r2:.4f}")

    logging.info(f"--- TCV Summary for {model_name} ---")
    logging.info(f"Average MAE: {np.mean(mae_scores):.4f} (std: {np.std(mae_scores):.4f})")
    logging.info(f"Average R2 Score: {np.mean(r2_scores):.4f} (std: {np.std(r2_scores):.4f})")
    logging.info(f"--- Finished TCV for {model_name} ---\n")

def train_model(model_config, params):
    """Loads data and trains a single, specified model."""
    processed_data_dir = 'data/processed'
    model_output_dir = 'models'
    
    model_name = model_config['name']
    model_params = model_config['params']
    model_filename = model_config['file_name']

    # Load data
    X_train = pd.read_csv(os.path.join(processed_data_dir, 'X_train.csv'), index_col='datetime', parse_dates=True)
    y_train = pd.read_csv(os.path.join(processed_data_dir, 'y_train.csv'), index_col='datetime', parse_dates=True).iloc[:, 0]

    # Optional: Run Time-Series Cross-Validation
    if params.get('validation', {}).get('run_tcv', False):
        perform_time_series_cross_validation(X_train.copy(), y_train.copy(), model_config)

    # Train the final model
    logging.info(f"Training final model: {model_name}")
    model = get_model_instance(model_name, model_params)
    model.fit(X_train, y_train)
    logging.info(f"Final training for {model_name} complete.")

    # Save the model
    os.makedirs(model_output_dir, exist_ok=True)
    model_path = os.path.join(model_output_dir, model_filename)
    joblib.dump(model, model_path)
    logging.info(f"Model saved to {model_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", help="The name of the model to train from params.yaml")
    args = parser.parse_args()

    all_params = load_params()
    
    # Find the specific model configuration from the list
    model_to_train = next((m for m in all_params['train']['models'] if m['name'] == args.model_name), None)

    if model_to_train:
        train_model(model_to_train, all_params)
    else:
        logging.error(f"Model '{args.model_name}' not found in params.yaml")
        exit(1)