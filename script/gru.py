# Copyright (c) 2024 Gabriel-Vasilică Sasu
# All rights reserved.
#
# This script performs data imputation on a dataset with missing values using a pre-trained GRU (Gated Recurrent Unit) model.
# The process involves several steps:
# 1. Loading the dataset with missing values.
# 2. Defining numeric columns for scaling and imputation.
# 3. Scaling numeric data using MinMaxScaler.
# 4. One-hot encoding the 'Activity' categorical column, handling potential NaNs in it.
# 5. Defining context columns (scaled numeric features + one-hot encoded activity).
# 6. Identifying rows that contain missing values in the specified numeric columns.
# 7. Loading a pre-trained GRU model (e.g., 'gru_imputation_model_with_activity_weight.h5').
# 8. Iterating through rows with missing data:
#    a. Extracting a sequence of preceding data points as context.
#    b. Handling cases where insufficient preceding data is available by padding.
#    c. Using the GRU model to predict the missing values based on the context.
#    d. Filling the missing values in the dataset with the model's predictions.
# 9. Inverse transforming the scaled numeric columns back to their original scale.
# 10. Dropping the temporary one-hot encoded activity columns.
# 11. Saving the imputed dataset to a new CSV file.


import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from tensorflow.keras.models import load_model
import tensorflow as tf

# Load the dataset with missing values
file_path = '../prelucrate_2_mcar.csv'  # Replace with your dataset path
data = pd.read_csv(file_path)

# Define the columns to scale and encode
all_columns = ['Accelerometerx', 'Accelerometery', 'Accelerometerz',
               'Gyroscopex', 'Gyroscopey', 'Gyroscopez', 'hrm']

# Fit the scaler on the data
scaler = MinMaxScaler()
scaler.fit(data[all_columns])

# Normalize relevant columns
data[all_columns] = scaler.transform(data[all_columns])

# Fit the encoder on the data
encoder = OneHotEncoder(sparse_output=False)  # Updated parameter name

# Handle potential missing values in 'Activity' column
if data['Activity'].isnull().any():
    data['Activity'].fillna('Unknown', inplace=True)  # Replace NaNs with a placeholder

encoder.fit(data[['Activity']])

# Preprocess the activity (One-Hot Encoding)
activity_encoded = encoder.transform(data[['Activity']])

# Add the encoded activity to the dataset
activity_columns = encoder.get_feature_names_out(['Activity'])
activity_df = pd.DataFrame(activity_encoded, columns=activity_columns, index=data.index)
data = pd.concat([data, activity_df], axis=1)

# Define context columns
context_columns = all_columns + list(activity_columns)

# Define sequence length
sequence_length = 20

# Identify rows with missing data
missing_indices = data[data[all_columns].isna().any(axis=1)].index

# Load the trained GRU model without compiling
model = load_model('gru_imputation_model_with_activity_weight.h5', compile=False)

# Print model input and output shapes
print(f"Model input shape: {model.input_shape}")
print(f"Model output shape: {model.output_shape}")

# Loop over the missing data and impute values using the model
for idx in missing_indices:
    # Ensure there's enough context before the missing value
    if idx - sequence_length >= 0:
        # Extract context data
        context_data = data.loc[idx-sequence_length:idx-1, context_columns].values.reshape(1, sequence_length, len(context_columns))
    else:
        # Use available data and pad if necessary
        context_data = data.loc[:idx-1, context_columns].values
        context_data = context_data[-sequence_length:]  # Get the last available data
        context_data = context_data.reshape(1, context_data.shape[0], context_data.shape[1])

        # Pad sequences if necessary
        if context_data.shape[1] < sequence_length:
            padding = np.zeros((1, sequence_length - context_data.shape[1], context_data.shape[2]))
            context_data = np.concatenate((padding, context_data), axis=1)

    # Predict missing values
    predicted_values = model.predict(context_data)

    # Impute missing values
    for i, col in enumerate(all_columns):
        if pd.isna(data.at[idx, col]):
            # Adjust indexing based on predicted_values shape
            if len(predicted_values.shape) == 3:
                imputed_value = predicted_values[0][-1][i]
            elif len(predicted_values.shape) == 2:
                imputed_value = predicted_values[0][i]
            else:
                raise ValueError(f"Unexpected shape of predicted_values: {predicted_values.shape}")

            # Assign the imputed value
            data.at[idx, col] = imputed_value

            # Print the assigned value
            print(f"Imputed value for index {idx}, column '{col}': {data.at[idx, col]}")

# Check for missing values before inverse transforming
missing_before_inverse = data[all_columns].isna().sum()
print("Missing values before inverse transform:")
print(missing_before_inverse)

# Inverse transform the data back to the original scale
data[all_columns] = scaler.inverse_transform(data[all_columns])

# Check for missing values before saving
missing_before_saving = data[all_columns].isna().sum()
print("Missing values before saving:")
print(missing_before_saving)

# Drop the one-hot encoded activity columns before saving
data = data.drop(columns=activity_columns)

# Save the imputed dataset
data.to_csv('prelucrate_imputed_gru_mcar.csv', index=False)

print("Data imputation completed and saved as 'prelucrate_imputed_mnar.csv' without activity encoded columns.")
