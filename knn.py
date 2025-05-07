# Copyright (c) 2024 Gabriel-Vasilică Sasu
# All rights reserved.
#
# This script performs data imputation using the K-Nearest Neighbors (KNN) algorithm, specifically KNeighborsRegressor from scikit-learn.
# The process is as follows:
# 1. Load the dataset that contains missing values.
# 2. Select only the numerical columns from the dataset, as KNN imputation is typically applied to numerical data.
# 3. For each numerical column that contains missing values:
#    a. The column with missing values becomes the target variable for the KNN model.
#    b. The other numerical columns are used as features to find the nearest neighbors.
#    c. Before training the KNN model or making predictions, any missing values in these feature columns (for the rows being considered)
#       are temporarily filled (e.g., using the mean of each respective feature column).
#    d. A KNeighborsRegressor model (e.g., with n_neighbors=5 and Manhattan distance metric) is trained on the rows where the target column is NOT missing.
#    e. This trained KNN model is then used to predict the missing values in the target column based on the feature values of the rows where the target is missing.
# 4. The original numerical columns in the dataset are updated with the imputed values.
# 5. The dataset, now with missing values imputed by KNN, is saved to a new CSV file.

from sklearn.neighbors import KNeighborsRegressor
import pandas as pd
import numpy as np # Imported for isnan, though pandas isna() is used directly.

# --- Configuration and Data Loading --- #
# Define the path to the dataset with missing values.
# IMPORTANT: Replace "../prelucrate_2_mnar.csv" with the actual path to your dataset.
file_path = "../prelucrate_2_mnar.csv"  # Example path for a dataset with MNAR missingness.
data = pd.read_csv(file_path)

# --- Data Preprocessing --- #
# Select numerical columns for imputation. KNN works with numerical data.
# Categorical columns are typically excluded or require specific encoding if used.
numerical_columns = data.select_dtypes(include=["float64", "int64"]).columns

# Create a copy of the numerical part of the DataFrame to work on.
# This avoids modifying the original DataFrame `data` directly during the column-wise imputation loop,
# until the very end when imputed values are assigned back.
X_imputation_target = data[numerical_columns].copy()

# --- KNN Imputation Loop --- #
# Loop over each numerical column to check for and impute missing values.
print("Starting KNN imputation process...")
for column in X_imputation_target.columns:
    # Create a boolean mask indicating missing values (True where value is NaN) for the current column.
    missing_mask = X_imputation_target[column].isna()
    
    # Proceed with imputation for this column only if it actually contains missing values.
    if missing_mask.sum() > 0:
        print(f"  Imputing column: {column}, Number of missing values: {missing_mask.sum()}")
        
        # Prepare the feature set (X_features) for the KNN model.
        # These are all other numerical columns, excluding the current `column` being imputed.
        X_features = X_imputation_target.drop(columns=[column])
        
        # Handle missing values within the feature set (X_features) before KNN.
        # KNN cannot handle NaNs in its input features. A common strategy is to fill these
        # temporary NaNs with the mean of their respective columns.
        # This fillna is applied to the entire X_features DataFrame for simplicity here.
        X_features_filled = X_features.fillna(X_features.mean())
        
        # Separate the data into training set (where current `column` is not missing)
        # and prediction set (where current `column` is missing).
        
        # Features for training the KNN model (from rows where `column` is NOT missing).
        X_train_knn = X_features_filled[~missing_mask]
        # Target values for training the KNN model (known values of `column`).
        y_train_knn = X_imputation_target.loc[~missing_mask, column]
        
        # Features for which we need to predict `column` (from rows where `column` IS missing).
        X_predict_knn = X_features_filled[missing_mask]
        
        # Initialize and train the KNeighborsRegressor model.
        # `n_neighbors=5`: Use 5 nearest neighbors.
        # `metric=\'manhattan\`: Use Manhattan distance (L1 norm) to find neighbors.
        # Other metrics like Euclidean (L2) could also be used.
        knn_imputer_model = KNeighborsRegressor(n_neighbors=5, metric="manhattan")
        knn_imputer_model.fit(X_train_knn, y_train_knn)
        
        # Predict the missing values in the current `column`.
        predicted_values = knn_imputer_model.predict(X_predict_knn)
        
        # Assign the predicted values back to the missing cells in the `column`.
        X_imputation_target.loc[missing_mask, column] = predicted_values

print("KNN imputation process completed.")

# --- Update Original DataFrame and Save --- #
# Replace the original numerical columns in the `data` DataFrame with the imputed versions from `X_imputation_target`.
data[numerical_columns] = X_imputation_target

# Save the dataset with imputed values to a new CSV file.
# IMPORTANT: Adjust the output filename as needed.
output_filename = "imputed_data_with_custom_metric_mnar.csv"
data.to_csv(output_filename, index=False)

print(f"Imputation complete with KNN (Manhattan distance). Imputed data saved to: {output_filename}")

