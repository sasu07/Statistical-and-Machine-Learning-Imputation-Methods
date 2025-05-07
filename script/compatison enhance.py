# Copyright (c) 2024 Gabriel-Vasilică Sasu
# All rights reserved.
#
# This script is designed to compare the performance of various imputation techniques.
# It loads an original dataset, a dataset with missing values, and several datasets that have been imputed using different methods.
# Key performance metrics such as Mean Squared Error (MSE), Mean Absolute Error (MAE), and R-squared (R2) score are calculated
# for each imputation method by comparing the imputed values against the original, complete data.
# The script then normalizes these metrics and generates a comparative bar plot to visualize the performance of each technique,
# highlighting the best and weakest performers based on MSE and the method with the highest R2 score.

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# --- Data Loading --- #
# Load the original complete dataset, the dataset with artificially introduced missing values,
# and various datasets imputed by different techniques.
# IMPORTANT: Replace the placeholder file paths below with the actual paths to your data files.

# Example: Load the original dataset without any missing values.
original_data = pd.read_csv(
    # "/Users/admin/Desktop/Desktop - Admin’s MacBook Pro/Doctorat/mssing data/Articol jurnal/missingness type/technics/prelucrate_2.csv"
    "../prelucrate_2.csv"  # Placeholder - update this path
)
# Example: Load the dataset with missing values (e.g., MCAR - Missing Completely At Random).
data_with_missing = pd.read_csv(
    # "/Users/admin/Desktop/Desktop - Admin’s MacBook Pro/Doctorat/mssing data/Articol jurnal/missingness type/technics/prelucrate_2_mcar.csv"
    "../prelucrate_2_mcar.csv"  # Placeholder - update this path
)
# Example: Load data imputed by Expectation-Maximization.
em_imputed_data = pd.read_csv(
    # "/Users/admin/Desktop/Desktop - Admin’s MacBook Pro/Doctorat/mssing data/Articol jurnal/missingness type/technics/1.expectation_maximization/prelucrate_2_mcar_imp_hybrid_4.csv"
    "../1.expectation_maximization/prelucrate_2_mcar_imp_hybrid_4.csv"  # Placeholder
)
# Example: Load data imputed by Matrix Completion (SVD).
imputed_svd_df = pd.read_csv(
    # "/Users/admin/Desktop/Desktop - Admin’s MacBook Pro/Doctorat/mssing data/Articol jurnal/missingness type/technics/2.matrix_completion/imputed_dataset_svd_mcar.csv"
    "../2.matrix_completion/imputed_dataset_svd_mcar.csv"  # Placeholder
)
# Example: Load data imputed by Matrix Completion (SoftImpute).
imputed_soft_df = pd.read_csv(
    # "/Users/admin/Desktop/Desktop - Admin’s MacBook Pro/Doctorat/mssing data/Articol jurnal/missingness type/technics/2.matrix_completion/imputed_dataset_soft_mcar.csv"
    "../2.matrix_completion/imputed_dataset_soft_mcar.csv"  # Placeholder
)
# Example: Load data imputed by Bayesian Network.
bayesian_imputed_data = pd.read_csv(
    # "/Users/admin/Desktop/Desktop - Admin’s MacBook Pro/Doctorat/mssing data/Articol jurnal/missingness type/technics/3.bayesian/imputed_dataset_bayesian_network_mcar.csv"
    "../3.bayesian/imputed_dataset_bayesian_network_mcar.csv"  # Placeholder
)
# Example: Load data imputed by K-Nearest Neighbors (KNN).
knn_imputed_data = pd.read_csv(
    # "/Users/admin/Desktop/Desktop - Admin’s MacBook Pro/Doctorat/mssing data/Articol jurnal/missingness type/technics/4.knn/imputed_data_with_custom_metric_mcar.csv"
    "../4.knn/imputed_data_with_custom_metric_mcar.csv"  # Placeholder
)
# Example: Load data imputed by Support Vector Machine (SVM).
svm_imputed_data = pd.read_csv(
    # "/Users/admin/Desktop/Desktop - Admin’s MacBook Pro/Doctorat/mssing data/Articol jurnal/missingness type/technics/5.svm/imputed_data_with_trained_svm_mcar.csv"
    "../5.svm/imputed_data_with_trained_svm_mcar.csv"  # Placeholder
)
# Example: Load data imputed by Generative Adversarial Imputation Nets (GAIN).
gain_imputed_data = pd.read_csv(
    # "/Users/admin/Desktop/Desktop - Admin’s MacBook Pro/Doctorat/mssing data/Articol jurnal/missingness type/technics/6.gain/imputed_data_mcar.csv"
    "../6.gain/imputed_data_mcar.csv"  # Placeholder
)
# Example: Load data imputed by Variational Autoencoder (VAE).
vae_imputed_data = pd.read_csv(
    # "/Users/admin/Desktop/Desktop - Admin’s MacBook Pro/Doctorat/mssing data/Articol jurnal/missingness type/technics/7.VAE/data_imputed_simple_vae_mcar.csv"
    "../7.VAE/data_imputed_simple_vae_mcar.csv"  # Placeholder
)
# Example: Load data imputed by Gated Recurrent Unit (GRU) based model.
gru_imputed_data = pd.read_csv(
    # "/Users/admin/Desktop/Desktop - Admin’s MacBook Pro/Doctorat/mssing data/Articol jurnal/missingness type/technics/8.GRU/prelucrate_imputed_gru_mcar.csv"
    "../8.GRU/prelucrate_imputed_gru_mcar.csv"  # Placeholder
)

# --- Data Preprocessing --- #
# Select only the numeric columns for comparison, as imputation metrics are typically calculated on numerical data.
# Columns like 'Activity' or 'time' might be categorical or identifiers and are excluded here.
numeric_columns = original_data.select_dtypes(include=[np.number]).columns

# --- Metric Calculation --- #
# Function to compute imputation performance metrics.
def compute_imputation_metrics(original_data_values, imputed_data_values, missing_mask_values):
    """Computes MSE, MAE, and R2 score for imputed data.

    Args:
        original_data_values (np.ndarray): NumPy array of original data (ground truth) for numeric columns.
        imputed_data_values (np.ndarray): NumPy array of imputed data for numeric columns.
        missing_mask_values (np.ndarray): Boolean NumPy array indicating missing positions (True where missing).

    Returns:
        tuple: A tuple containing MSE, MAE, and R2 score.
    """
    # Calculate metrics only on the values that were originally missing.
    mse = mean_squared_error(original_data_values[missing_mask_values], imputed_data_values[missing_mask_values])
    mae = mean_absolute_error(original_data_values[missing_mask_values], imputed_data_values[missing_mask_values])
    r2 = r2_score(original_data_values[missing_mask_values], imputed_data_values[missing_mask_values])
    return mse, mae, r2

# Prepare a dictionary to store metrics for each imputation method.
imputation_metrics = {
    "Expectation_Maximization": {},
    "Matrix_Completion_SVD": {},
    "Matrix_Completion_SoftImpute": {},
    "Bayesian_Network": {},
    "KNN": {},
    "SVM": {},
    "GAIN": {},
    "VAE": {},
    "GRU": {}
}

# Dictionary mapping method names to their corresponding imputed datasets.
datasets = {
    "Expectation_Maximization": em_imputed_data,
    "Matrix_Completion_SVD": imputed_svd_df,
    "Matrix_Completion_SoftImpute": imputed_soft_df,
    "Bayesian_Network": bayesian_imputed_data,
    "KNN": knn_imputed_data,
    "SVM": svm_imputed_data,
    "GAIN": gain_imputed_data,
    "VAE": vae_imputed_data,
    "GRU": gru_imputed_data
}

# Create a mask that identifies the locations of missing values in the `data_with_missing` dataset.
# This mask is crucial for evaluating imputation performance only on the originally missing cells.
# The `.values` attribute converts the DataFrame selection to a NumPy array for direct use with `np.isnan`.
missing_mask = np.isnan(data_with_missing[numeric_columns].values)

# Loop through each imputation method and its corresponding dataset to compute and store performance metrics.
for method, imputed_data_df in datasets.items():
    # Ensure the imputed dataset has the same numeric columns as the original for fair comparison.
    # Also, convert to NumPy arrays for metric calculation.
    mse, mae, r2 = compute_imputation_metrics(
        original_data[numeric_columns].values, 
        imputed_data_df[numeric_columns].values, 
        missing_mask
    )
    
    # Store the computed metrics in the dictionary.
    imputation_metrics[method] = {
        "MSE": mse,
        "MAE": mae,
        "R2 Score": r2
    }

# Convert the metrics dictionary into a pandas DataFrame for easier handling and visualization.
# The .T transposes the DataFrame to have methods as rows and metrics as columns.
metrics_df = pd.DataFrame(imputation_metrics).T

# --- Metric Normalization and Plotting --- #
# Normalize MSE and MAE using Min-Max normalization to bring them to a comparable scale (0 to 1).
# R2 score is already on a relative scale, so it's often not normalized in the same way, but can be if desired.
metrics_df_normalized = metrics_df.copy()
# Normalized MSE = (MSE - min(MSE)) / (max(MSE) - min(MSE))
metrics_df_normalized["MSE"] = (metrics_df["MSE"] - metrics_df["MSE"].min()) / (metrics_df["MSE"].max() - metrics_df["MSE"].min())
# Normalized MAE = (MAE - min(MAE)) / (max(MAE) - min(MAE))
metrics_df_normalized["MAE"] = (metrics_df["MAE"] - metrics_df["MAE"].min()) / (metrics_df["MAE"].max() - metrics_df["MAE"].min())
# Rename columns for clarity in the plot.
metrics_df_normalized = metrics_df_normalized.rename(columns={"MSE": "Normalized MSE", "MAE": "Normalized MAE", "R2 Score": "R²"})

# Enhanced Plotting function with professional callouts and labels.
def enhanced_plot(plot_metrics_df, title, filename):
    """Generates and saves an enhanced bar plot comparing imputation methods.

    Args:
        plot_metrics_df (pd.DataFrame): DataFrame containing the (normalized) metrics for plotting.
        title (str): The title for the plot.
        filename (str): The filename to save the plot image.
    """
    sns.set(style="whitegrid") # Set a professional plot style.
    
    # Define a color palette for the bars.
    colors = sns.color_palette("pastel")
    
    # Create the bar plot.
    ax = plot_metrics_df.plot(kind="bar", figsize=(14, 9), color=colors, edgecolor="black") # Increased figure size
    
    # Set titles and labels with appropriate font sizes.
    plt.title(title, fontsize=18, weight="bold")
    plt.ylabel("Normalized Metric Value / R² Score", fontsize=15) # Clarified Y-axis label
    plt.xlabel("Imputation Methods", fontsize=15)
    plt.xticks(rotation=45, ha="right", fontsize=13) # Enhanced X-ticks readability
    plt.legend(loc="best", fancybox=True, shadow=True, ncol=1, fontsize=13) # Adjusted legend

    # Annotate the weakest performer based on Normalized MSE (higher is worse).
    weakest_method_mse = plot_metrics_df["Normalized MSE"].idxmax()
    weakest_value_mse = plot_metrics_df["Normalized MSE"].max()
    ax.text(plot_metrics_df.index.get_loc(weakest_method_mse), weakest_value_mse * 0.95, # Adjusted position
            f"Weakest MSE ({weakest_method_mse})", fontsize=11, color="red", ha="center", weight="bold", 
            bbox=dict(facecolor="white", edgecolor="red", alpha=0.7))

    # Annotate the best performer based on Normalized MSE (lower is better).
    best_method_mse = plot_metrics_df["Normalized MSE"].idxmin()
    best_value_mse = plot_metrics_df["Normalized MSE"].min()
    # Position annotation slightly above the bar for visibility.
    ax.text(plot_metrics_df.index.get_loc(best_method_mse), best_value_mse + 0.05, # Adjusted position
            f"Best MSE ({best_method_mse})", fontsize=11, color="blue", ha="center", weight="bold", 
            bbox=dict(facecolor="white", edgecolor="blue", alpha=0.7))

    # Annotate the method with the highest R² score (higher is better).
    highest_r2_method = plot_metrics_df["R²"].idxmax()
    highest_r2_value = plot_metrics_df["R²"].max()
    # Position annotation slightly above the bar for R2.
    ax.text(plot_metrics_df.index.get_loc(highest_r2_method), highest_r2_value + 0.05, # Adjusted position
            f"Highest R² ({highest_r2_method})", fontsize=11, color="green", ha="center", weight="bold", 
            bbox=dict(facecolor="white", edgecolor="green", alpha=0.7))

    # Adjust y-axis limits to provide some padding and prevent overlap with annotations.
    min_val = plot_metrics_df.values.min()
    max_val = plot_metrics_df.values.max()
    ax.set_ylim([min_val - 0.1*abs(min_val), max_val + 0.1*abs(max_val)]) # Dynamic padding

    # Add gridlines and improve layout.
    ax.grid(True, which="major", axis="y", linestyle="--", linewidth=0.7)
    plt.tight_layout() # Adjusts plot to ensure everything fits without overlapping.
    
    # Save the plot to a file.
    plt.savefig(filename)
    plt.show() # Display the plot.

# --- Plot Generation --- #
# Generate the enhanced comparison plot. 
# This example assumes the data used was for an MNAR (Missing Not At Random) scenario.
# The title and filename should be adjusted if comparing results for MCAR or MAR scenarios.
enhanced_plot(metrics_df_normalized, "Comparison of Imputation Methods (MNAR Scenario)", "imputation_comparison_enhanced_mnar.png")

# Note: To generate plots for MCAR and MAR scenarios, you would typically:
# 1. Load the corresponding `data_with_missing` files (e.g., `prelucrate_2_mcar.csv`, `prelucrate_2_mar.csv`).
# 2. Load the imputed datasets generated for those specific missingness patterns.
# 3. Recalculate the `missing_mask` based on the MCAR/MAR `data_with_missing` file.
# 4. Re-run the metric calculation loop.
# 5. Call `enhanced_plot` with appropriate titles and filenames (e.g., "mcar_comparison_enhanced.png").

print("Imputation comparison script completed. Plot saved.")

