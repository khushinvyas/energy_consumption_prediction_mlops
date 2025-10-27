import os
import pandas as pd
import joblib
import json
import yaml
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import sys

def load_params():
    try:
        with open("params.yaml", "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print("ERROR: params.yaml not found.")
        sys.exit(1)

def main():
    try:
        params = load_params()
        
        # Define paths
        processed_dir = params["data"]["processed_data_dir"]
        train_file_path = os.path.join(processed_dir, params["data"]["train_file"])
        test_file_path = os.path.join(processed_dir, params["data"]["test_file"])
        model_path = "models/model.joblib"
        reports_dir = "reports"
        metrics_path = os.path.join(reports_dir, "metrics.json")
        
        # --- VALIDATION: Check if data and model exist ---
        if not os.path.exists(train_file_path):
            print(f"ERROR: Training data file not found at '{train_file_path}'.")
            sys.exit(1)
        if not os.path.exists(test_file_path):
            print(f"ERROR: Test data file not found at '{test_file_path}'.")
            sys.exit(1)
        if not os.path.exists(model_path):
            print(f"ERROR: Model file not found at '{model_path}'.")
            sys.exit(1)

        # Load the trained model
        print(f"Loading model from {model_path}")
        model = joblib.load(model_path)
        
        # Define features and target
        features = params["featurization"]["features_to_use"]
        target = params["preprocessing"]["target_col"]

        # --- Evaluate on TRAINING set ---
        print("\n--- Evaluating on TRAINING data ---")
        train_df = pd.read_csv(train_file_path)
        X_train = train_df[features]
        y_train = train_df[target]
        y_train_pred = model.predict(X_train)
        
        train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
        train_r2 = r2_score(y_train, y_train_pred)
        print(f"Training RMSE: {train_rmse:.4f}")
        print(f"Training R² Score: {train_r2:.4f}")

        # --- Evaluate on TEST set ---
        print("\n--- Evaluating on TEST data ---")
        test_df = pd.read_csv(test_file_path)
        X_test = test_df[features]
        y_test = test_df[target]
        y_test_pred = model.predict(X_test)
        
        test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
        test_r2 = r2_score(y_test, y_test_pred)
        print(f"Test RMSE: {test_rmse:.4f}")
        print(f"Test R² Score: {test_r2:.4f}")

        # --- Comparison and Diagnosis ---
        print("\n--- Overfitting Diagnosis ---")
        r2_diff = train_r2 - test_r2
        print(f"Difference between Training and Test R²: {r2_diff:.4f}")

        if r2_diff > 0.05:  # A 5% difference is a good rule of thumb
            print("WARNING: The large gap between training and test scores suggests OVERFITTING.")
        else:
            print("The scores are similar. The model appears to be generalizing well.")

        # Save the TEST metrics to JSON file (for DVC tracking)
        # We save the test metrics as the "official" result of the pipeline
        metrics = {
            "train_rmse": float(train_rmse),
            "train_r2_score": float(train_r2),
            "test_rmse": float(test_rmse),
            "test_r2_score": float(test_r2)
        }
        
        os.makedirs(reports_dir, exist_ok=True)
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        
        print(f"\nAll metrics saved to {metrics_path}")

    except Exception as e:
        print(f"An unexpected error occurred during model evaluation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()