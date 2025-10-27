import os
import pandas as pd
import yaml
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
        raw_data_path = os.path.join(params["data"]["raw_data_dir"], params["data"]["raw_data_file"])
        processed_dir = params["data"]["processed_data_dir"]
        cleaned_file_path = os.path.join(processed_dir, params["data"]["cleaned_file"])
        
        # Create processed directory if it doesn't exist
        os.makedirs(processed_dir, exist_ok=True)

        # --- Load the dataset from the local raw file ---
        print(f"Loading data from local file: {raw_data_path}")
        
        # --- VALIDATION: Check if the raw data file exists ---
        if not os.path.exists(raw_data_path):
            print(f"ERROR: Raw data file not found at '{raw_data_path}'.")
            print("Please make sure you have downloaded the file and placed it in the correct directory.")
            sys.exit(1)
            
        # The data is semi-colon separated
        df = pd.read_csv(
            raw_data_path,
            sep=params["preprocessing"]["separator"],
            low_memory=False
        )
        print("Dataset loaded successfully.")
        
        # --- The rest of the preprocessing logic remains the same ---
        
        # Handle missing values marked as '?' by replacing with NaN
        print("Handling missing values...")
        numeric_cols = [
            "Global_active_power", "Global_reactive_power", "Voltage", 
            "Global_intensity", "Sub_metering_1", "Sub_metering_2", "Sub_metering_3"
        ]
        
        for col in numeric_cols:
            df[col] = df[col].replace("?", float("nan"))
        
        # Convert to numeric types
        print("Converting columns to numeric types...")
        for col in numeric_cols:
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