import os
import pandas as pd
import joblib
import yaml
from xgboost import XGBRegressor
# CHANGE: Import RandomizedSearchCV instead of GridSearchCV
from sklearn.model_selection import RandomizedSearchCV 
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
            sys.exit(1)

        # Load the training data
        print(f"Loading training data from {train_file_path}")
        train_df = pd.read_csv(train_file_path)
        
        # Define features and target
        features = params["featurization"]["features_to_use"]
        target = params["preprocessing"]["target_col"]
        
        X_train = train_df[features]
        y_train = train_df[target]
        
        # --- RIGOROUS TRAINING WITH RANDOMIZED SEARCH AND CROSS-VALIDATION ---
        
        # 1. Get the base model and tuning parameters from params.yaml
        base_model_params = params["model"]
        tuning_params = params["tuning"]
        param_distributions = tuning_params["param_distributions"]
        cv_folds = tuning_params["cv_folds"]
        n_iter = tuning_params["n_iter"]
        
        print(f"Starting advanced training with {cv_folds}-fold cross-validation...")
        print(f"Will test {n_iter} random parameter combinations.")
        print(f"Parameter distributions to sample from: {param_distributions}")
        
        # 2. Initialize the base XGBoost model
        base_model = XGBRegressor(
            # Use a reasonable default for parameters not being tuned
            n_estimators=base_model_params["n_estimators"],
            max_depth=base_model_params["max_depth"],
            learning_rate=base_model_params["learning_rate"],
            min_child_weight=base_model_params["min_child_weight"],
            subsample=base_model_params["subsample"],
            colsample_bytree=base_model_params["colsample_bytree"],
            random_state=params["base"]["random_state"],
            n_jobs=-1 
        )
        
        # 3. Set up RandomizedSearchCV
        # This will test n_iter random combinations from param_distributions.
        random_search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            n_iter=n_iter,  # Number of parameter settings sampled
            cv=cv_folds,    # Number of cross-validation folds
            scoring='neg_root_mean_squared_error', # Metric to optimize
            verbose=2,      # Show progress
            random_state=params["base"]["random_state"], # For reproducibility
            n_jobs=-1       # Use all available CPU cores
        )
        
        # 4. Run the randomized search on the training data
        print("Running RandomizedSearchCV... This will take some time.")
        random_search.fit(X_train, y_train)
        
        # 5. Print the results
        print("\n--- Randomized Search Complete ---")
        print(f"Best parameters found: {random_search.best_params_}")
        print(f"Best cross-validated RMSE: {-random_search.best_score_:.4f}")
        
        # 6. Save the best model
        # RandomizedSearchCV automatically refits a model on the entire training set
        # using the best parameters it found.
        os.makedirs(model_dir, exist_ok=True)
        print(f"Saving best model to {model_path}")
        joblib.dump(random_search.best_estimator_, model_path)
        
        print("Advanced model training completed successfully.")

    except Exception as e:
        print(f"An unexpected error occurred during model training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()