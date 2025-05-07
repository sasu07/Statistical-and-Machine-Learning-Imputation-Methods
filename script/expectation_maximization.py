# Copyright (c) 2024 Gabriel-Vasilică Sasu
# All rights reserved.
#
# This script implements a custom Expectation-Maximization (EM) algorithm for imputing missing data in a dataset,
# assuming the data follows a multivariate normal distribution.
# The process involves these main steps:
# 1. Load the dataset with missing values.
# 2. Select only the numerical columns for imputation, as the EM algorithm described here is based on multivariate normality.
# 3. Initialize the EM algorithm:
#    a. Perform an initial naive imputation of missing values (e.g., using the mean of each column).
#    b. Estimate the initial mean vector and covariance matrix from this naively imputed data.
# 4. Iterate through the Expectation (E-step) and Maximization (M-step) until convergence:
#    a. E-step: For each row with missing values, calculate the conditional expectation of the missing components,
#       given the observed components and the current estimates of the mean and covariance matrix.
#       These expected values are used to fill in the missing data points for this iteration.
#       The `conditional_distribution` helper function is used for this, derived from properties of multivariate normal distributions.
#    b. M-step: Re-estimate the mean vector and covariance matrix from the data that was completed in the E-step.
# 5. Convergence is checked by comparing the change in the imputed data matrix between iterations against a tolerance threshold.
# 6. Once converged, the resulting DataFrame with imputed values (for numeric columns) is returned.
# 7. Merge back any categorical or non-numeric columns that were initially excluded.
# 8. Save the fully imputed dataset to a new CSV file.


import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal
from sklearn.preprocessing import MinMaxScaler

# Helper function: Compute the conditional mean and variance
def conditional_distribution(mean, cov, index_known, index_unknown, known_values):
    """
    Calculate the conditional distribution for the missing values given the observed ones.
    Parameters:
    - mean: Mean vector of the distribution
    - cov: Covariance matrix of the distribution
    - index_known: Indices of the known (observed) values
    - index_unknown: Indices of the unknown (missing) values
    - known_values: Observed values corresponding to index_known
    
    Returns:
    - conditional_mean: Expected value of the missing data
    - conditional_cov: Covariance of the missing data
    """
    # Partition the covariance matrix
    sigma_11 = cov[np.ix_(index_known, index_known)]
    sigma_12 = cov[np.ix_(index_known, index_unknown)]
    sigma_22 = cov[np.ix_(index_unknown, index_unknown)]
    
    # Compute the conditional mean and covariance
    sigma_22_inv = np.linalg.inv(sigma_22)
    conditional_mean = mean[index_unknown] + sigma_12.T @ np.linalg.inv(sigma_11) @ (known_values - mean[index_known])
    conditional_cov = sigma_22 - sigma_12.T @ np.linalg.inv(sigma_11) @ sigma_12
    
    return conditional_mean, conditional_cov

# Custom Expectation-Maximization (EM) algorithm for missing data
def em_impute(data, max_iter=10, tol=1e-4):
    """
    EM-based imputation for missing data.
    Parameters:
    - data: DataFrame with missing values
    - max_iter: Maximum number of iterations
    - tol: Tolerance for convergence (stop when changes are smaller than tol)
    
    Returns:
    - imputed_data: DataFrame with imputed missing values
    """
    # Initial estimates: Mean imputation for the missing values
    data_filled = data.copy()
    for col in data.columns:
        data_filled[col].fillna(data[col].mean(), inplace=True)
    
    # Convert to numpy array for easier calculations
    X = data_filled.values
    
    # Estimate initial mean and covariance matrix from the data
    mean = np.nanmean(X, axis=0)
    cov = np.cov(X, rowvar=False)
    
    for iteration in range(max_iter):
        X_old = X.copy()
        
        # E-step: Estimate missing values based on current parameters
        for i in range(X.shape[0]):  # Iterate over each row
            missing_idx = np.isnan(data.iloc[i, :].values)
            observed_idx = ~missing_idx
            
            if np.any(missing_idx):
                observed_values = X[i, observed_idx]
                conditional_mean, _ = conditional_distribution(mean, cov, observed_idx, missing_idx, observed_values)
                
                # Fill in the missing values with conditional expectation
                X[i, missing_idx] = conditional_mean
        
        # M-step: Update the mean and covariance matrix based on the filled data
        mean = np.mean(X, axis=0)
        cov = np.cov(X, rowvar=False)
        
        # Check for convergence
        if np.linalg.norm(X - X_old) < tol:
            print(f"Converged after {iteration + 1} iterations.")
            break
    
    # Return the imputed data as a DataFrame
    imputed_data = pd.DataFrame(X, columns=data.columns)
    return imputed_data

# Load your smart bracelet dataset
file_path = 'prelucrate_2_mnar.csv'  # Replace with your dataset path
data = pd.read_csv(file_path)

# Drop categorical columns before applying EM (e.g., Activity, time)
numeric_columns = ['Accelerometerx', 'Accelerometery', 'Accelerometerz', 'Gyroscopex', 
                   'Gyroscopey', 'Gyroscopez', 'Orientationa', 'Orientationb', 
                   'Orientationc', 'Orientationd', 'AccMagnitude', 'GyroMagnitude', 
                   'Roll', 'Pitch', 'Yaw', 'hrm']
numeric_data = data[numeric_columns]

# Apply the EM algorithm to impute missing values
imputed_numeric_data = em_impute(numeric_data)

# Merge back the categorical columns (Activity, time)
imputed_data = pd.concat([imputed_numeric_data, data[['Activity', 'time']].reset_index(drop=True)], axis=1)

# Save the imputed dataset
imputed_data.to_csv('imputed_dataset_custom_em_mnar.csv', index=False)

print("Custom EM-based imputation completed.")







"""
Explanation of Code:
Initialization:
The missing values are initially imputed using simple mean imputation to provide a starting point.
E-Step:
For each row with missing values, the conditional expectation of the missing values is computed based on the observed values and the current estimates of the mean and covariance.
M-Step:
After imputing the missing values, the mean and covariance of the entire dataset are updated based on the filled data.
Convergence:
The algorithm checks for convergence by comparing the newly imputed data with the previous iteration. If the change is below a certain threshold (tol), the algorithm stops.
Practical Application:
For a dataset collected from a smart bracelet, which often involves multiple sensor readings such as accelerometer, gyroscope, heart rate, etc., this EM-based approach allows 
for a multivariate imputation that takes into account the relationships between the different sensor readings to produce more accurate imputations for missing data.
"""
