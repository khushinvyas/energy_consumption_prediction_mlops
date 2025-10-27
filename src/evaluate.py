import os
import pandas as pd
import joblib
import json
import yaml
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def load_params():
    """Load parameters from params.yaml file"""
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    # Load parameters from params.yaml
    params = load_params()
    
    # Define paths from params
    processed_dir = params["data"]["processed_data_dir"]
    test_file_path = os.path.join(processed_dir, params["data"]["test_file"])
    model_path = "models/model.joblib"
    reports_dir = "reports"
    metrics_path = os.path.join(reports_dir, "metrics.json")
    
    # Create reports directory if it doesn't exist
    os.makedirs(reports_dir, exist_ok=True)
    
    # Load the test data
    print(f"Loading test data from {test_file_path}")
    test_df = pd.read_csv(test_file_path)
    
    # Define features and target
    features = params["featurization"]["features_to_use"]
    target = params["preprocessing"]["target_col"]
    
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
    
    # Save metrics to JSON file
    print(f"Saving metrics to {metrics_path}")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    
    print(f"Model evaluation completed. RMSE: {rmse:.4f}, R2 Score: {r2:.4f}")

if __name__ == "__main__":
    main()