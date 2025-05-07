# Copyright (c) 2024 Gabriel-Vasilică Sasu
# All rights reserved.
#
# This script performs data imputation using Support Vector Regression (SVR).
# The process is as follows:
# 1. Load two datasets: one complete original dataset (for training SVR models) and 
#    one incomplete dataset (with missing values that need to be imputed).
# 2. Identify and select only the numerical columns from the datasets, as SVR is typically used for regression on numerical features.
# 3. Standardize both the complete and incomplete numerical data using StandardScaler. This is important for SVR performance.
# 4. Train a separate SVR model for each numerical column that might have missing values:
#    a. For each target column, the other numerical columns are used as features.
#    b. The SVR model (with an RBF kernel) is trained on the complete (scaled) dataset to predict the target column.
#    c. The trained SVR model for each column is stored.
# 5. Impute missing values in the incomplete (scaled) dataset:
#    a. For each column with missing values, use its corresponding trained SVR model.
#    b. The features for prediction are the other (scaled) columns from the incomplete dataset.
#       - If these feature columns themselves have missing values at a given row, they are temporarily filled (e.g., with their mean) 
#         for the purpose of making a prediction for the target column in that row.
#    c. The predicted values from SVR are used to fill the NaNs in the target column.
# 6. The script assumes that the imputation is done on the scaled data. The imputed values in `incomplete_data` will be on the original scale
#    because `y_train` was from `complete_data` (original scale) and predictions are assigned back to `incomplete_data`.
#    However, the features `X_train` and `X_impute_filled` were scaled. For consistency, it's often better to predict on scaled target and then inverse_transform.
#    *Correction*: `y_train` is `complete_data[column]`, which is on the original scale. `X_train` is from `complete_scaled`. This means SVR is trained to predict original scale values from scaled features.
#    The `incomplete_scaled` data is used for `X_impute`, and its NaNs are filled with its mean (which would be mean of scaled data, close to 0).
#    The predictions are then directly assigned to `incomplete_data` (original scale). This approach is viable.
# 7. Save the dataset with imputed values to a new CSV file.

from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np # Imported numpy for potential use, though not explicitly used in the final version of this script.

# --- Configuration and Data Loading --- #
# Define file paths for the complete (original) dataset and the incomplete dataset.
# IMPORTANT: Replace placeholder paths with the actual paths to your data files.
file_path_complete = "../prelucrate_2.csv"  # Path to the original dataset without missing values.
complete_data = pd.read_csv(file_path_complete)

file_path_incomplete = "../prelucrate_2_mnar.csv"  # Path to the dataset with missing values (e.g., MNAR).
incomplete_data = pd.read_csv(file_path_incomplete)

# --- Data Preprocessing --- #
# Select numerical columns for training the SVR models and for imputation.
# SVR works with numerical data. Categorical columns are excluded.
numerical_columns = complete_data.select_dtypes(include=["float64", "int64"]).columns

# Standardize the numerical features of both datasets.
# Standardization (scaling to zero mean and unit variance) is crucial for SVR performance.
scaler = StandardScaler()

# Fit the scaler on the complete data and transform it.
complete_scaled_array = scaler.fit_transform(complete_data[numerical_columns])
complete_scaled = pd.DataFrame(complete_scaled_array, columns=numerical_columns)

# Transform the incomplete data using the scaler fitted on the complete data.
# This ensures consistency in scaling between training and imputation phases.
incomplete_scaled_array = scaler.transform(incomplete_data[numerical_columns])
incomplete_scaled = pd.DataFrame(incomplete_scaled_array, columns=numerical_columns)
# Note: `incomplete_scaled` will have NaNs where `incomplete_data` had NaNs, as StandardScaler propagates NaNs.

# --- SVR Model Training --- #
# Train a separate SVR model for each numerical column to predict its values based on other columns.
svm_models = {}  # Dictionary to store the trained SVR model for each column.

print("Training SVR models for each numerical column...")
for column in numerical_columns:
    print(f"  Training SVR for column: {column}")
    # Prepare training data: Features (X_train) are all other scaled numerical columns,
    # Target (y_train) is the current column from the original (unscaled) complete data.
    X_train = complete_scaled.drop(columns=[column])
    y_train = complete_data[column] # Target is on the original scale.
    
    # Initialize and train the SVR model.
    # RBF (Radial Basis Function) kernel is a common choice for SVR.
    # Hyperparameters (C, gamma, epsilon) might need tuning for optimal performance.
    svr = SVR(kernel="rbf")  
    svr.fit(X_train, y_train)
    
    # Store the trained model for the current column.
    svm_models[column] = svr
print("SVR model training completed.")

# --- Imputation using Trained SVR Models --- #
# Impute missing values in the `incomplete_data` DataFrame.

print("Imputing missing values using trained SVR models...")
for column in numerical_columns:
    # Identify rows where the current column has missing values in the original incomplete dataset.
    missing_mask = incomplete_data[column].isna()
    
    # Proceed with imputation only if there are missing values in the current column.
    if missing_mask.sum() > 0:
        print(f"  Imputing column: {column}, Number of missing values: {missing_mask.sum()}")
        # Prepare the feature set for imputation from the scaled incomplete data.
        # These are the other columns that will be used to predict the current `column`.
        X_impute_features = incomplete_scaled.drop(columns=[column])
        
        # For the rows where `column` is missing, we need to predict its value.
        # The features for these predictions (`X_impute_features[missing_mask]`) might themselves have missing values.
        # SVR cannot handle NaNs in input features. So, temporarily fill NaNs in these feature columns
        # (e.g., using the mean of each feature column from `X_impute_features`).
        # This mean is calculated from the scaled data, so it will be close to 0 if data was properly standardized.
        X_impute_filled_for_prediction = X_impute_features[missing_mask].fillna(X_impute_features.mean())
        
        # Ensure no NaNs remain in the data being fed to predict.
        if X_impute_filled_for_prediction.isnull().any().any():
            print(f"    Warning: NaNs still present in features for column {column} after fillna. Check data.")
            # As a fallback, fill any remaining NaNs with 0 (mean of standard scaled data)
            X_impute_filled_for_prediction = X_impute_filled_for_prediction.fillna(0)

        # Predict the missing values for the current `column` using its trained SVR model.
        predicted_values = svm_models[column].predict(X_impute_filled_for_prediction)
        
        # Assign the predicted values to the corresponding missing cells in the original `incomplete_data` DataFrame.
        incomplete_data.loc[missing_mask, column] = predicted_values

print("SVM-based imputation completed.")

# --- Saving the Imputed Dataset --- #
# Save the `incomplete_data` DataFrame, now with missing values imputed by SVR, to a new CSV file.
# IMPORTANT: Adjust the output filename as needed.
output_filename = "imputed_data_with_trained_svm_mnar.csv"
incomplete_data.to_csv(output_filename, index=False)

print(f"Imputed data saved to: {output_filename}")

