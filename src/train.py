import os
import pandas as pd
import joblib
import yaml
from xgboost import XGBRegressor
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
        model_dir = "models"
        model_path = os.path.join(model_dir, "model.joblib")
        
        # --- VALIDATION: Check if training data exists ---
        if not os.path.exists(train_file_path):
            print(f"ERROR: Training data file not found at '{train_file_path}'.")
            print("Please run the data split step first.")
            sys.exit(1)

        # Load the training data
        print(f"Loading training data from {train_file_path}")
        train_df = pd.read_csv(train_file_path)
        
        # Define features and target
        features = params["featurization"]["features_to_use"]
        target = params["preprocessing"]["target_col"]
        required_cols = features + [target]

        # --- VALIDATION: Check for required columns ---
        if not all(col in train_df.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in train_df.columns]
            print(f"ERROR: Training data is missing required columns: {missing_cols}")
            sys.exit(1)

        X_train = train_df[features]
        y_train = train_df[target]
        
        # Get model hyperparameters
        model_params = params["model"]
        print(f"Training model with hyperparameters: {model_params}")
        
        # Initialize and train the XGBoost model
        model = XGBRegressor(
            n_estimators=model_params["n_estimators"],
            max_depth=model_params["max_depth"],
            learning_rate=model_params["learning_rate"],
            min_child_weight=model_params["min_child_weight"],
            subsample=model_params["subsample"],
            colsample_bytree=model_params["colsample_bytree"],
            random_state=params["base"]["random_state"]
        )
        
        model.fit(X_train, y_train)
        
        # Create model directory if it doesn't exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Save the trained model
        print(f"Saving trained model to {model_path}")
        joblib.dump(model, model_path)
        
        print("Model training completed successfully.")

    except Exception as e:
        print(f"An unexpected error occurred during model training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()