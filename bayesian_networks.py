# Copyright (c) 2024 Gabriel-Vasilică Sasu
# All rights reserved.
#
# This script performs data imputation using a Bayesian Network.
# The key steps involved are:
# 1. Load the dataset with missing values.
# 2. Preprocess the data:
#    a. Exclude non-relevant columns (e.g., 'time') before imputation if they are not part of the model.
#    b. Encode categorical features (e.g., 'Activity') into numerical representations using LabelEncoder.
#    c. Discretize continuous numerical variables into a smaller number of bins using KBinsDiscretizer.
#       Bayesian Networks in pgmpy often work better with discrete data or require it for certain estimators.
#       Missing values in continuous columns are temporarily filled (e.g., with 0) before discretization.
#    d. Ensure all data fed to the Bayesian Network model is of integer type.
# 3. Define the structure of the Bayesian Network. This involves specifying the directed edges (dependencies) between variables (columns).
#    This structure is domain-specific and crucial for the model's performance.
# 4. Initialize the BayesianNetwork model with the defined structure.
# 5. Estimate the parameters (Conditional Probability Distributions - CPDs) of the network using the Expectation-Maximization (EM) algorithm.
#    EM is suitable here as it can handle missing data during parameter learning.
# 6. Update the model with the estimated CPDs and check its validity.
# 7. Perform inference to impute missing values:
#    a. Initialize an inference engine (e.g., VariableElimination).
#    b. For each row with missing values, use the observed values in that row as evidence.
#    c. Query the Bayesian Network for the most probable values (Maximum a Posteriori - MAP query) of the missing variables given the evidence.
#    d. Fill the missing cells with these inferred values.
# 8. Merge back any columns that were excluded initially (e.g., 'time').
# 9. Save the dataset with imputed values to a new CSV file.

import pandas as pd
import numpy as np # Not explicitly used in this version but often useful with pandas.
from pgmpy.models import BayesianNetwork
from pgmpy.estimators import ExpectationMaximization # For parameter learning with missing data.
from pgmpy.inference import VariableElimination # For performing inference to impute values.
from sklearn.preprocessing import LabelEncoder, KBinsDiscretizer # For data preprocessing.

# --- Configuration and Data Loading --- #
# Define the path to the dataset with missing values.
# IMPORTANT: Replace 'prelucrate_2_mnar.csv' with the actual path to your dataset.
file_path = "prelucrate_2_mnar.csv"  # Example: data with Missing Not At Random (MNAR) values.
data = pd.read_csv(file_path)

# --- Data Preprocessing --- #

# Step 1: Exclude 'time' column if it's not part of the Bayesian Network model or imputation target.
# This is done before other preprocessing steps that might alter column order or types.
data_for_bn = data.copy()
if "time" in data_for_bn.columns:
    data_without_time = data_for_bn.drop(columns=["time"])
else:
    data_without_time = data_for_bn

# Step 2: Encode the 'Activity' column to a numerical format using LabelEncoder.
# Bayesian Networks typically require discrete or numerical data.
if "Activity" in data_without_time.columns:
    le = LabelEncoder()
    data_without_time["Activity"] = le.fit_transform(data_without_time["Activity"].astype(str)) # astype(str) to handle potential mixed types or NaNs as strings

# Step 3: Discretize continuous variables.
# Bayesian Networks, especially with pgmpy's standard estimators, often work best with discrete data.
# KBinsDiscretizer divides continuous data into k bins.
continuous_columns = [
    "Accelerometerx", "Accelerometery", "Accelerometerz",
    "Gyroscopex", "Gyroscopey", "Gyroscopez",
    "Orientationa", "Orientationb", "Orientationc", "Orientationd",
    "AccMagnitude", "GyroMagnitude",
    "Roll", "Pitch", "Yaw", "hrm"
]

# Filter out columns not present in data_without_time to avoid errors
continuous_columns_present = [col for col in continuous_columns if col in data_without_time.columns]

if continuous_columns_present:
    # Initialize the discretizer: 5 bins, ordinal encoding, uniform strategy.
    discretizer = KBinsDiscretizer(n_bins=5, encode="ordinal", strategy="uniform", subsample=None)
    
    # Temporarily fill NaNs in continuous columns (e.g., with 0) before discretization,
    # as KBinsDiscretizer cannot handle NaNs directly during fit/transform.
    # The choice of fill value (0 here) should be considered carefully; median or mean might be alternatives.
    data_to_discretize = data_without_time[continuous_columns_present].fillna(0)
    discretized_values = discretizer.fit_transform(data_to_discretize)
    
    # Update the DataFrame with discretized values.
    data_without_time[continuous_columns_present] = discretized_values

# Ensure all data is of integer type, as required by some pgmpy functionalities or for consistency.
# This step assumes that after encoding and discretization, all relevant columns can be treated as integers.
# NaNs will be converted to a large negative number by astype(int) if not handled, so ensure NaNs are appropriately managed before this.
# For pgmpy, it's often better to work with data that has NaNs explicitly for EM.
# However, if discretization was performed on NaN-filled data, those fills are now part of the discrete values.
# Let's convert to integer type, but pgmpy's EM should still handle pd.NA or np.nan if present in the input to EM.
data_for_em = data_without_time.copy()
for col in data_for_em.columns:
    # Convert to float first to handle potential pd.NA, then to integer if possible.
    # This is a bit tricky; pgmpy's EM expects a DataFrame where NaNs are np.nan.
    if data_for_em[col].isnull().any():
         # If column still has NaNs, ensure they are np.nan for pgmpy
        data_for_em[col] = data_for_em[col].astype(float) # This will keep np.nan as is
    else:
        data_for_em[col] = data_for_em[col].astype(int)

# --- Bayesian Network Definition and Parameter Estimation --- #

# Step 4: Define the Bayesian Network structure (dependencies between variables).
# This is a crucial step and usually requires domain knowledge.
# The list contains tuples representing directed edges (Parent, Child).
# Example structure (needs to be adapted to the actual dataset and domain knowledge):
structure = [
    ('Accelerometerx', 'Accelerometery'),
    ('Accelerometery', 'Accelerometerz'),
    ('Gyroscopex', 'Gyroscopey'),
    ('Gyroscopey', 'Gyroscopez'),
    ('Accelerometerx', 'Gyroscopex'),
    ('Orientationa', 'Orientationb'),
    ('Orientationb', 'Orientationc'),
    ('Orientationc', 'Orientationd'),
    ('AccMagnitude', 'GyroMagnitude'),
    ('Roll', 'Pitch'),
    ('Pitch', 'Yaw'),
    ('Activity', 'Accelerometerx'),
    ('hrm', 'Accelerometerx'),
    # Add any additional dependencies based on your data
]
# Filter structure to only include nodes present in the data_for_em columns
valid_nodes = data_for_em.columns.tolist()
filtered_structure = [(u, v) for u, v in structure if u in valid_nodes and v in valid_nodes]

# Step 5: Initialize the Bayesian Network model with the defined structure.
model = BayesianNetwork(filtered_structure)

# Step 6: Initialize the Expectation-Maximization (EM) estimator for parameter learning.
# EM is used here because the data (`data_for_em`) may contain missing values (np.nan).
# The EM algorithm will estimate the CPDs from this incomplete data.
print("Estimating CPDs using Expectation-Maximization...")
em = ExpectationMaximization(model, data_for_em)

# Step 7: Estimate the parameters (CPDs) of the Bayesian Network.
# `n_jobs=-1` can be used to parallelize if computationally intensive and supported.
# `max_iter` might need adjustment.
estimated_cpds = em.get_parameters(latent_cardinalities={}) # Provide empty dict if no latent vars explicitly defined beyond EM's handling of missing data
print("CPD estimation completed.")

# Step 8: Update the model's CPDs with the parameters estimated by EM.
model.cpds = [] # Clear any existing CPDs
model.add_cpds(*estimated_cpds)

# Step 9: Check if the model structure and CPDs are valid.
assert model.check_model(), "The Bayesian Network model is invalid after parameter estimation."
print("Bayesian Network model is valid.")

# --- Imputation using Bayesian Network Inference --- #

# Step 10: Initialize the inference engine (Variable Elimination is a common choice).
inference_engine = VariableElimination(model)

# Function to impute missing data using the trained Bayesian Network.
def impute_missing_with_bn(data_to_impute_df, bn_model, inference_method):
    imputed_df = data_to_impute_df.copy()
    # Identify rows with any missing data.
    rows_with_missing = imputed_df[imputed_df.isnull().any(axis=1)]
    
    print(f"Found {len(rows_with_missing)} rows with missing values to impute.")
    for idx, row in rows_with_missing.iterrows():
        # For each row with missing data, prepare evidence (observed values).
        evidence_dict = row.dropna().to_dict()
        # Identify variables (columns) that are missing in this row.
        missing_vars_list = row[row.isnull()].index.tolist()
        
        # Ensure evidence keys and missing_vars are nodes in the model.
        evidence_dict = {k: v for k, v in evidence_dict.items() if k in bn_model.nodes()}
        missing_vars_list = [var for var in missing_vars_list if var in bn_model.nodes()]

        if not missing_vars_list: # Skip if no missing vars are part of the model
            continue
            
        try:
            # Perform MAP (Maximum a Posteriori) query to find the most probable values for missing variables.
            map_result = inference_method.map_query(variables=missing_vars_list, evidence=evidence_dict, show_progress=False)
            # Fill the missing values in the DataFrame with the inferred results.
            for var in missing_vars_list:
                if var in map_result:
                    imputed_df.at[idx, var] = map_result[var]
        except Exception as e:
            print(f"Error imputing row {idx} for variables {missing_vars_list} with evidence {evidence_dict}: {e}")
            # Optionally, handle the error, e.g., by leaving NaNs or using a fallback.
    return imputed_df

# Step 11: Impute missing data in the `data_for_em` DataFrame.
print("Starting imputation of missing values...")
imputed_data_discrete = impute_missing_with_bn(data_for_em, model, inference_engine)
print("Imputation completed.")

# --- Post-processing and Saving --- #
# Note: The imputed data (`imputed_data_discrete`) is in the discretized and label-encoded format.
# If the original scale/format is needed, inverse transformations for discretization and label encoding would be required.
# This script saves the data in its processed (discretized/encoded) state after imputation.

# Step 12: Merge back the 'time' column if it was originally present and dropped.
if "time" in data.columns:
    imputed_data_discrete["time"] = data["time"]

# Step 13: Save the final imputed dataset.
# IMPORTANT: Adjust the output filename as needed.
output_filename = "imputed_dataset_bayesian_network_mnar.csv"
imputed_data_discrete.to_csv(output_filename, index=False)

print("Bayesian network-based imputation completed and saved as 'imputed_dataset_bayesian_network_mcar.csv'.")
