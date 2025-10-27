import os
import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import sys

def load_params():
    """Load parameters from params.yaml file"""
    try:
        with open("params.yaml", "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print("ERROR: params.yaml not found.")
        sys.exit(1)

def main():
    try:
        # Load parameters from params.yaml
        params = load_params()
        
        # Define paths from params
        processed_dir = params["data"]["processed_data_dir"]
        cleaned_file_path = os.path.join(processed_dir, params["data"]["cleaned_file"])
        train_file_path = os.path.join(processed_dir, params["data"]["train_file"])
        test_file_path = os.path.join(processed_dir, params["data"]["test_file"])
        
        # --- VALIDATION: Check if cleaned data file exists ---
        if not os.path.exists(cleaned_file_path):
            print(f"ERROR: Cleaned data file not found at '{cleaned_file_path}'.")
            print("Please run the data preprocessing step first.")
            sys.exit(1)
        
        # Load the cleaned data
        print(f"Loading cleaned data from {cleaned_file_path}")
        df = pd.read_csv(cleaned_file_path)
        
        # Define features and target
        features = params["featurization"]["features_to_use"]
        target = params["preprocessing"]["target_col"]
        required_cols = features + [target]

        # --- VALIDATION: Check if all required columns exist in the dataframe ---
        if not all(col in df.columns for col in required_cols):
            missing_cols = [col for col in required_cols if col not in df.columns]
            print(f"ERROR: The cleaned data is missing required columns: {missing_cols}")
            print(f"Available columns are: {df.columns.tolist()}")
            print("Please check your 'featurization' and 'preprocessing' sections in params.yaml.")
            sys.exit(1)
        print("All required columns for splitting are present.")

        # Split the data into training and testing sets
        print(f"Splitting data with test ratio: {params['data_split']['test_split_ratio']}")
        train_df, test_df = train_test_split(
            df, 
            test_size=params["data_split"]["test_split_ratio"],
            random_state=params["base"]["random_state"]
        )
        
        # Ensure the output directory exists
        os.makedirs(processed_dir, exist_ok=True)
        
        # Save the train and test sets
        print(f"Saving training data to {train_file_path}")
        train_df.to_csv(train_file_path, index=False)
        
        print(f"Saving test data to {test_file_path}")
        test_df.to_csv(test_file_path, index=False)
        
        print("Data split completed successfully.")

    except Exception as e:
        print(f"An unexpected error occurred during data splitting: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()