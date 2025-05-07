# Copyright (c) 2024 Gabriel-Vasilică Sasu
# All rights reserved.
#
# This script evaluates the performance of an imputation model, specifically GAIN (Generative Adversarial Imputation Nets),
# by comparing the imputed data with the original complete data. It calculates various metrics such as Root Mean Squared Error (RMSE)
# per feature and overall RMSE. It also visualizes the error distribution and compares GAIN with baseline imputation methods
# like mean and median imputation.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# **Assuming you have run the improved GAIN code and obtained the following variables:**
# - `original_data_array`: NumPy array of the original data without missing values.
# - `imputed_data`: NumPy array of the imputed data from the GAIN model.
# - `data_miss_array`: NumPy array of the data with missing values.
# - `mask`: NumPy array indicating observed (1) and missing (0) values.
# - `numeric_cols`: List of names of the numeric columns in your dataset.

# If these variables are not already defined, you can define them as follows:

# **Load the Original Data and Data with Missing Values**
# Replace 'original_data.csv' and 'missing_data.csv' with your actual file paths
original_data = pd.read_csv('../prelucrate_2.csv')
data_miss = pd.read_csv('../prelucrate_2_mar.csv')

# **Select Only Numeric Columns**
numeric_cols = original_data.select_dtypes(include=[np.number]).columns

# **Convert DataFrames to NumPy Arrays**
original_data_array = original_data[numeric_cols].values
data_miss_array = data_miss[numeric_cols].values

# **Create the Mask Matrix**
def create_mask(data):
    mask = ~np.isnan(data)
    mask = mask.astype(int)
    return mask

mask = create_mask(data_miss_array)

# **Imputed Data from GAIN**
# If you have saved the imputed data from the GAIN model to a CSV file:
imputed_full_df = pd.read_csv('imputed_data.csv')
imputed_data = imputed_full_df[numeric_cols].values

# **Evaluation Code Starts Here**

# **1. Function to calculate RMSE per feature**
def rmse_per_feature(original_data, imputed_data, mask):
    rmse_features = []
    for i in range(original_data.shape[1]):
        mask_feature = mask[:, i]
        original_feature = original_data[:, i]
        imputed_feature = imputed_data[:, i]
        missing_indices = np.where(mask_feature == 0)[0]
        if len(missing_indices) > 0:
            mse = np.mean((original_feature[missing_indices] - imputed_feature[missing_indices]) ** 2)
            rmse = np.sqrt(mse)
        else:
            rmse = np.nan  # No missing values in this feature
        rmse_features.append(rmse)
    return rmse_features

# **Calculate RMSE per feature**
rmse_features = rmse_per_feature(original_data_array, imputed_data, mask)
rmse_df = pd.DataFrame({
    'Feature': numeric_cols,
    'RMSE': rmse_features
})
print("RMSE per Feature:")
print(rmse_df)

# **Plot RMSE per feature**
plt.figure(figsize=(10, 6))
plt.bar(rmse_df['Feature'], rmse_df['RMSE'])
plt.xlabel('Features')
plt.ylabel('RMSE')
plt.title('RMSE per Feature')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# **2. Calculate Overall RMSE**
def rmse_loss(original_data, imputed_data, mask):
    numerator = np.sum(((1 - mask) * (original_data - imputed_data)) ** 2)
    denominator = np.sum(1 - mask)
    rmse = np.sqrt(numerator / (denominator + 1e-8))
    return rmse

overall_rmse = rmse_loss(original_data_array, imputed_data, mask)
print(f'Overall Imputation RMSE: {overall_rmse}')

# **3. Visualize Error Distribution**
errors = ((1 - mask) * (imputed_data - original_data_array)).flatten()
errors = errors[~np.isnan(errors)]  # Exclude NaNs
errors = errors[errors != 0]        # Exclude zeros (non-missing positions)

plt.figure(figsize=(10, 6))
plt.hist(errors, bins=50, edgecolor='k')
plt.xlabel('Imputation Error')
plt.ylabel('Frequency')
plt.title('Distribution of Imputation Errors')
plt.show()

# **4. Compare with Baseline Imputation Methods**

# **Mean Imputation**
mean_values = np.nanmean(data_miss_array, axis=0)
mean_imputed_data = np.where(np.isnan(data_miss_array), mean_values, data_miss_array)

# **Calculate RMSE for Mean Imputation**
mean_rmse = rmse_loss(original_data_array, mean_imputed_data, mask)
print(f'Mean Imputation RMSE: {mean_rmse}')

# **Median Imputation**
median_values = np.nanmedian(data_miss_array, axis=0)
median_imputed_data = np.where(np.isnan(data_miss_array), median_values, data_miss_array)

# **Calculate RMSE for Median Imputation**
median_rmse = rmse_loss(original_data_array, median_imputed_data, mask)
print(f'Median Imputation RMSE: {median_rmse}')

# **Summary of RMSEs**
rmse_summary = pd.DataFrame({
    'Method': ['GAIN', 'Mean Imputation', 'Median Imputation'],
    'RMSE': [overall_rmse, mean_rmse, median_rmse]
})
print("\nRMSE Summary:")
print(rmse_summary)

# **Plot RMSE Comparison**
plt.figure(figsize=(8, 5))
plt.bar(rmse_summary['Method'], rmse_summary['RMSE'], color=['blue', 'orange', 'green'])
plt.xlabel('Imputation Method')
plt.ylabel('RMSE')
plt.title('Comparison of Imputation Methods')
plt.show()

# **5. Detailed Error Analysis per Feature**

# **Function to calculate error statistics per feature**
def error_stats_per_feature(original_data, imputed_data, mask):
    stats = []
    for i in range(original_data.shape[1]):
        mask_feature = mask[:, i]
        original_feature = original_data[:, i]
        imputed_feature = imputed_data[:, i]
        missing_indices = np.where(mask_feature == 0)[0]
        if len(missing_indices) > 0:
            errors = original_feature[missing_indices] - imputed_feature[missing_indices]
            mean_error = np.mean(errors)
            std_error = np.std(errors)
            max_error = np.max(errors)
            min_error = np.min(errors)
        else:
            mean_error = std_error = max_error = min_error = np.nan
        stats.append({
            'Feature': numeric_cols[i],
            'Mean Error': mean_error,
            'Std Error': std_error,
            'Max Error': max_error,
            'Min Error': min_error
        })
    return pd.DataFrame(stats)

error_stats_df = error_stats_per_feature(original_data_array, imputed_data, mask)
print("\nError Statistics per Feature:")
print(error_stats_df)

# **Plot Mean Error per Feature**
plt.figure(figsize=(10, 6))
plt.bar(error_stats_df['Feature'], error_stats_df['Mean Error'])
plt.xlabel('Features')
plt.ylabel('Mean Error')
plt.title('Mean Imputation Error per Feature')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
