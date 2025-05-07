# Copyright (c) 2024 Gabriel-Vasilică Sasu
# All rights reserved.
#
# This script performs data imputation using matrix completion techniques, specifically IterativeSVD and SoftImpute from the fancyimpute library.
# The process is as follows:
# 1. Load the dataset that contains missing values.
# 2. Identify and select the numerical columns that require imputation. Categorical columns (like 'Activity', 'time') are typically excluded from direct matrix completion.
# 3. Normalize the selected numerical data (e.g., using MinMaxScaler to scale features to a [0, 1] range).
#    Normalization can improve the performance and stability of matrix completion algorithms.
# 4. Apply IterativeSVD to impute missing values in the normalized data.
#    IterativeSVD repeatedly performs Singular Value Decomposition (SVD) and reconstructs the matrix to fill missing entries.
# 5. Apply SoftImpute to impute missing values in the normalized data.
#    SoftImpute is another matrix completion algorithm that uses nuclear norm regularization.
# 6. Inverse transform the imputed data (from both SVD and SoftImpute methods) back to its original scale using the previously fitted scaler.
# 7. Convert the imputed NumPy arrays back into pandas DataFrames, preserving original column names.
# 8. Merge back any categorical or non-numeric columns that were initially excluded, ensuring proper alignment (e.g., by resetting index).
# 9. Save the two imputed datasets (one by IterativeSVD, one by SoftImpute) to separate CSV files.


import numpy as np
import pandas as pd
from fancyimpute import IterativeSVD, SoftImpute
from sklearn.preprocessing import MinMaxScaler

# Load your dataset
file_path = 'prelucrate_2_mnar.csv'  # Replace with the path to your dataset
data = pd.read_csv(file_path)

# Identify the columns to impute (e.g., accelerometer, gyroscope, etc.)
# Exclude categorical columns like 'Activity' and 'time'
numeric_columns = ['Accelerometerx', 'Accelerometery', 'Accelerometerz', 'Gyroscopex', 
                   'Gyroscopey', 'Gyroscopez', 'Orientationa', 'Orientationb', 
                   'Orientationc', 'Orientationd', 'AccMagnitude', 'GyroMagnitude', 
                   'Roll', 'Pitch', 'Yaw', 'hrm']
numeric_data = data[numeric_columns]

# Normalize the data before imputation
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(numeric_data)

# Apply matrix completion using IterativeSVD (similar to SVD matrix completion)
svd_imputer = IterativeSVD(max_iters=100)
imputed_data_svd = svd_imputer.fit_transform(normalized_data)

# Apply matrix completion using SoftImpute (another matrix completion technique)
soft_imputer = SoftImpute()
imputed_data_soft = soft_imputer.fit_transform(normalized_data)

# Inverse transform the data back to the original scale after imputation
imputed_data_svd = scaler.inverse_transform(imputed_data_svd)
imputed_data_soft = scaler.inverse_transform(imputed_data_soft)

# Convert back to DataFrame
imputed_svd_df = pd.DataFrame(imputed_data_svd, columns=numeric_columns)
imputed_soft_df = pd.DataFrame(imputed_data_soft, columns=numeric_columns)

# Merge back categorical columns (Activity, time)
imputed_svd_df = pd.concat([imputed_svd_df, data[['Activity', 'time']].reset_index(drop=True)], axis=1)
imputed_soft_df = pd.concat([imputed_soft_df, data[['Activity', 'time']].reset_index(drop=True)], axis=1)

# Save the imputed datasets
imputed_svd_df.to_csv('imputed_dataset_svd_mnar.csv', index=False)
imputed_soft_df.to_csv('imputed_dataset_soft_mnar.csv', index=False)

print("Matrix completion-based imputation completed using SVD and SoftImpute.")
