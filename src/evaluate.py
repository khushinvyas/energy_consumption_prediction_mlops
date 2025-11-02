# src/evaluate.py
import pandas as pd
import joblib
import os
import json
import yaml
import numpy as np
import logging
import argparse # Import for command-line arguments
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_params():
    """Loads parameters from params.yaml."""
    with open('params.yaml', 'r') as f:
        return yaml.safe_load(f)

def evaluate_model(model_config, params):
    """Loads a specified model and evaluates it."""
    model_name = model_config['name']
    model_filename = model_config['file_name']
    
    model_path = os.path.join('models', model_filename)
    processed_data_dir = 'data/processed'
    metrics_output_dir = 'metrics'
    plots_dir = params['validation']['plots_dir']

    logging.info(f"--- Evaluating model: {model_name} ---")
    model = joblib.load(model_path)

    # Load test data
    X_test = pd.read_csv(os.path.join(processed_data_dir, 'X_test.csv'), index_col='datetime', parse_dates=True)
    y_test = pd.read_csv(os.path.join(processed_data_dir, 'y_test.csv'), index_col='datetime', parse_dates=True).iloc[:, 0]
    
    predictions = model.predict(X_test)

    # Calculate metrics
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    r2 = r2_score(y_test, predictions)
    metrics = {'mae': mae, 'rmse': rmse, 'r2_score': r2}

    # Save metrics to a model-specific file
    os.makedirs(metrics_output_dir, exist_ok=True)
    metrics_path = os.path.join(metrics_output_dir, f"{model_name}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=4)
    logging.info(f"Metrics for {model_name}: MAE={mae:.4f}, RMSE={rmse:.4f}, R2={r2:.4f}")

    # Residual analysis plots
    os.makedirs(plots_dir, exist_ok=True)
    residuals = y_test - predictions

    plt.figure(figsize=(15, 6))
    plt.plot(y_test.index, residuals, marker='.', linestyle='None', alpha=0.6)
    plt.axhline(y=0, color='r', linestyle='--'); plt.title(f'{model_name} - Residuals Over Time')
    plt.savefig(os.path.join(plots_dir, f'{model_name}_residuals_over_time.png'))
    plt.close()

    plt.figure(figsize=(10, 6))
    sns.histplot(residuals, kde=True); plt.title(f'{model_name} - Distribution of Residuals')
    plt.savefig(os.path.join(plots_dir, f'{model_name}_residuals_histogram.png'))
    plt.close()
    
    logging.info(f"Validation plots for {model_name} saved to {plots_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", help="The name of the model to evaluate from params.yaml")
    args = parser.parse_args()

    all_params = load_params()
    model_to_evaluate = next((m for m in all_params['train']['models'] if m['name'] == args.model_name), None)

    if model_to_evaluate:
        evaluate_model(model_to_evaluate, all_params)
    else:
        logging.error(f"Model '{args.model_name}' not found in params.yaml")
        exit(1)