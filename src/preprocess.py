# preprocess.py
import pandas as pd
import numpy as np
import os
import yaml
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

def preprocess_data(input_path, output_dir, params):
    """
    Loads raw data, performs feature engineering, and splits into train/test sets.
    """
    logging.info(f"Loading data from {input_path}")
    df = pd.read_csv(input_path, index_col='datetime', parse_dates=True)

    # Ensure all columns are numeric, converting non-numeric to NaN then ffilling
    for col in df.columns:
        if df[col].dtype == 'object':
            df[col] = pd.to_numeric(df[col], errors='coerce')
    df.ffill(inplace=True) # Fill any NaNs that resulted from coercion

    # Feature Engineering: Time-based features
    df['hour_of_day'] = df.index.hour
    df['day_of_week'] = df.index.dayofweek
    df['month'] = df.index.month
    df['year'] = df.index.year

    # Define target and features
    target_column = params['preprocess']['target_column']
    features_to_use = params['train']['features']

    # Ensure target column and features exist
    if target_column not in df.columns:
        logging.error(f"Target column '{target_column}' not found in the dataset.")
        exit(1)
    for feature in features_to_use:
        if feature not in df.columns:
            logging.warning(f"Feature '{feature}' not found in the dataset. It will be ignored.")
    
    # Filter features to only include those present in the dataframe
    actual_features = [f for f in features_to_use if f in df.columns]
    
    X = df[actual_features]
    y = df[target_column]

    # Time-series split (chronological)
    test_split_ratio = params['preprocess']['test_split_ratio']
    test_size = int(len(df) * test_split_ratio)

    X_train = X.iloc[:-test_size]
    X_test = X.iloc[-test_size:]
    y_train = y.iloc[:-test_size]
    y_test = y.iloc[-test_size:]

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Save processed data
    X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=True)
    X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=True)
    y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=True)
    y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=True)

    logging.info(f"Data preprocessing complete. Saved to {output_dir}")
    logging.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    logging.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

if __name__ == "__main__":
    params = load_params()
    input_file = 'data/raw/household_power_consumption_hourly.csv'
    output_directory = 'data/processed'
    preprocess_data(input_file, output_directory, params)