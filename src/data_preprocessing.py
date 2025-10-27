import os
import pandas as pd
import yaml
from ucimlrepo import fetch_ucirepo
import sys # Import sys to exit gracefully on error

def load_params():
    """Load parameters from params.yaml file"""
    try:
        with open("params.yaml", "r") as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        print("ERROR: params.yaml not found. Make sure you are running the script from the project root.")
        sys.exit(1) # Exit the script with an error code

def main():
    try:
        # Load parameters from params.yaml
        params = load_params()
        
        # Define paths from params
        processed_dir = params["data"]["processed_data_dir"]
        cleaned_file_path = os.path.join(processed_dir, params["data"]["cleaned_file"])
        
        # Create processed directory if it doesn't exist
        os.makedirs(processed_dir, exist_ok=True)

        # --- Download and load the dataset from UCI repository ---
        print("Fetching dataset from UCI ML Repository...")
        individual_household_electric_power_consumption = fetch_ucirepo(id=235) 
        X = individual_household_electric_power_consumption.data.features 
        y = individual_household_electric_power_consumption.data.targets 
        
        # Combine features and targets into a single DataFrame
        df = pd.concat([X, y], axis=1)
        print("Dataset fetched and combined successfully.")
        
        # --- VALIDATION: Check if expected columns are present ---
        # This is a critical check to prevent downstream errors.
        expected_cols = [
            "Global_active_power", "Global_reactive_power", "Voltage", 
            "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"
        ]
        if not all(col in df.columns for col in expected_cols):
            missing_cols = [col for col in expected_cols if col not in df.columns]
            print(f"ERROR: The fetched dataset is missing expected columns: {missing_cols}")
            print(f"Available columns are: {df.columns.tolist()}")
            sys.exit(1)
        print("All expected columns are present.")

        # Handle missing values marked as '?' by replacing with NaN
        print("Handling missing values...")
        for col in expected_cols:
            df[col] = df[col].replace("?", float("nan"))
        
        # Convert to numeric types
        print("Converting columns to numeric types...")
        for col in expected_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        
        # Drop rows with missing values
        print("Dropping rows with missing values...")
        initial_rows = len(df)
        df.dropna(inplace=True)
        final_rows = len(df)
        print(f"Dropped {initial_rows - final_rows} rows with missing values.")
        
        # Save the cleaned data
        print(f"Saving cleaned data to {cleaned_file_path}")
        df.to_csv(cleaned_file_path, index=False)
        
        print("Data preprocessing completed successfully.")

    except Exception as e:
        print(f"An unexpected error occurred during preprocessing: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()