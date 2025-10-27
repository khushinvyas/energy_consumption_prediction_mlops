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
        test_file_path = os.path.join(processed_dir, params["data"]["test_file"])
        model_path = "models/model.joblib"
        reports_dir = "reports"
        metrics_path = os.path.join(reports_dir, "metrics.json")
        
        # --- VALIDATION: Check if test data and model exist ---
        if not os.path.exists(test_file_path):
            print(f"ERROR: Test data file not found at '{test_file_path}'.")
            sys.exit(1)
        if not os.path.exists(model_path):
            print(f"ERROR: Model file not found at '{model_path}'.")
            print("Please run the training step first.")
            sys.exit(1)

        # Load the test data
        print(f"Loading test data from {test_file_path}")
        test_df = pd.read_csv(test_file_path)
        
        # Define features and target
        features = params["featurization"]["features_to_use"]
        target = params["preprocessing"]["target_col"]
        required_cols = features + [target]

        # --- VALIDATION: Check for required columns ---
        if not all(col in test_df.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in test_df.columns]
            print(f"ERROR: Test data is missing required columns: {missing_cols}")
            sys.exit(1)

        X_test = test_df[features]
        y_test = test_df[target]
        
        # Load the trained model
        print(f"Loading model from {model_path}")
        model = joblib.load(model_path)
        
        # Make predictions on the test set
        print("Making predictions on test set")
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Create metrics dictionary
        metrics = {
            "rmse": float(rmse),
            "r2_score": float(r2)
        }
        
        # Create reports directory if it doesn't exist
        os.makedirs(reports_dir, exist_ok=True)
        
        # Save metrics to JSON file
        print(f"Saving metrics to {metrics_path}")
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=4)
        
        print(f"Model evaluation completed. RMSE: {rmse:.4f}, R2 Score: {r2:.4f}")

    except Exception as e:
        print(f"An unexpected error occurred during model evaluation: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()