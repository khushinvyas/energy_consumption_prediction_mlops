import os
import pandas as pd
import yaml
from ucimlrepo import fetch_ucirepo

def load_params():
    """Load parameters from params.yaml file"""
    with open("params.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    # Load parameters from params.yaml
    params = load_params()
    
    # Define paths from params
    processed_dir = params["data"]["processed_data_dir"]
    cleaned_file_path = os.path.join(processed_dir, params["data"]["cleaned_file"])
    
    # Create processed directory if it doesn't exist
    os.makedirs(processed_dir, exist_ok=True)

    # --- NEW: Download and load the dataset from UCI repository ---
    print("Fetching dataset from UCI ML Repository...")
    try:
        # fetch dataset 
        individual_household_electric_power_consumption = fetch_ucirepo(id=235) 
        
        # data (as pandas dataframes) 
        # The repository provides features and targets separately.
        # We need to combine them to replicate the original dataset structure.
        X = individual_household_electric_power_consumption.data.features 
        y = individual_household_electric_power_consumption.data.targets 
        
        # Combine features and targets into a single DataFrame
        # The target 'Global_active_power' is a column in the original dataset.
        df = pd.concat([X, y], axis=1)
        
        print("Dataset fetched successfully.")
    except Exception as e:
        print(f"Error fetching dataset: {e}")
        print("Please ensure you have an internet connection.")
        return

    # --- The rest of the preprocessing logic remains the same ---
    
    # Handle missing values marked as '?' by replacing with NaN
    print("Handling missing values")
    # Replace '?' with NaN for numeric columns
    numeric_cols = [
        "Global_active_power", 
        "Global_reactive_power", 
        "Voltage", 
        "Global_intensity",
        "Sub_metering_1",
        "Sub_metering_2",
        "Sub_metering_3"
    ]
    
    for col in numeric_cols:
        df[col] = df[col].replace("?", float("nan"))
    
    # Convert to numeric types
    print("Converting columns to numeric types")
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    
    # Drop rows with missing values
    print("Dropping rows with missing values")
    df.dropna(inplace=True)
    
    # Save the cleaned data
    print(f"Saving cleaned data to {cleaned_file_path}")
    df.to_csv(cleaned_file_path, index=False)
    
    print("Data preprocessing completed successfully")

if __name__ == "__main__":
    main()