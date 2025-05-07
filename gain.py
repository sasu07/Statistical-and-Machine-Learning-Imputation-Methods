# Copyright (c) 2024 Gabriel-Vasilică Sasu
# All rights reserved.
#
# This script implements the Generative Adversarial Imputation Nets (GAIN) model for imputing missing data in a tabular dataset.
# The core idea of GAIN is to train a Generator (G) to impute missing values and a Discriminator (D) to distinguish
# between observed and imputed values. The two models are trained adversarially.
#
# Key steps in the script:
# 1. Data Loading and Preprocessing:
#    - Loads an original complete dataset (for potential reference/evaluation, though not directly used in this GAIN training script for ground truth during training).
#    - Loads a dataset with missing values (`data_miss`).
#    - Selects only numeric columns for imputation.
#    - Creates a mask matrix (`mask`) indicating the positions of observed (1) and missing (0) values.
#    - Standardizes the numeric data (with missing values filled by `np.nan_to_num` after standardization which defaults to 0).
# 2. Model Architecture Definition:
#    - Defines a Generator model: Takes a data vector (with noise in missing parts) concatenated with a hint vector as input,
#      and outputs an imputed version of the data vector.
#    - Defines a Discriminator model: Takes a data vector (potentially imputed) concatenated with a hint vector as input,
#      and outputs a probability vector indicating, for each component, whether it is observed or imputed.
# 3. Training Setup:
#    - Defines hyperparameters (alpha for hint rate, batch size, epochs).
#    - Sets up Adam optimizers with learning rate schedules for both G and D.
#    - Prepares a TensorFlow `Dataset` for efficient batching and shuffling during training.
# 4. Core GAIN Logic:
#    - `sample_hint_tf`: Generates a hint matrix that provides partial information to the Discriminator about which components were originally observed.
#    - `impute_data`: A function to perform imputation using the trained generator (used for evaluation during/after training).
#    - `rmse_loss`: Calculates Root Mean Squared Error between original and imputed data for missing components (used for early stopping).
#    - `train_step` (decorated with `@tf.function` for performance):
#        - Generator (G) tries to generate realistic imputations to fool D.
#        - Discriminator (D) tries to correctly identify imputed values from observed ones, guided by the hint matrix.
#        - G_loss: Encourages G to produce imputations that D classifies as observed.
#        - D_loss: Standard binary cross-entropy loss for D.
#        - Gradients are computed and applied with clipping.
# 5. Training Loop:
#    - Iterates for a specified number of epochs.
#    - In each epoch, iterates over batches of data.
#    - Calls `train_step` to update G and D.
#    - Periodically evaluates imputation RMSE on the full dataset and saves the best generator model based on this RMSE (early stopping mechanism).
#    - Handles potential NaN values in losses to stop training if instability occurs.
# 6. Post-Training:
#    - Loads the best saved generator model.
#    - Uses the best generator to impute the missing data in the (normalized) dataset.
#    - Destandardizes the imputed data back to its original scale.
#    - Combines the imputed numeric data with any non-numeric columns from the original missing dataset.
#    - Saves the final imputed dataset to a CSV file.
#    - Plots the training losses for G and D.


import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import os

# Load the Original Data and Data with Missing Values
# Replace with your actual file paths
original_data = pd.read_csv('../prelucrate_2.csv')
data_miss = pd.read_csv('../prelucrate_2_mnar.csv')

# Select Only Numeric Columns
numeric_cols = original_data.select_dtypes(include=[np.number]).columns
non_numeric_cols = original_data.select_dtypes(exclude=[np.number]).columns

original_data_numeric = original_data[numeric_cols]
data_miss_numeric = data_miss[numeric_cols]

# Convert DataFrames to NumPy arrays
original_data_array = original_data_numeric.values
data_miss_array = data_miss_numeric.values

# Ensure both datasets have the same shape
assert original_data_array.shape == data_miss_array.shape, "The shapes of the datasets do not match."

# Create the Mask Matrix
def create_mask(data):
    mask = ~np.isnan(data)
    mask = mask.astype(float)
    return mask

mask = create_mask(data_miss_array)

# Standardize the Data
def standardize(data):
    mean_val = np.nanmean(data, axis=0)
    std_val = np.nanstd(data, axis=0)

    # Avoid division by zero
    std_val[std_val < 1e-8] = 1e-8

    # Standardize data
    std_data = (data - mean_val) / std_val
    std_data = np.nan_to_num(std_data)
    return std_data, mean_val, std_val

norm_data, mean_val, std_val = standardize(data_miss_array)

# Ensure input data is valid
print("Number of NaNs in norm_data:", np.isnan(norm_data).sum())
print("Number of Infs in norm_data:", np.isinf(norm_data).sum())
print("Number of NaNs in mask:", np.isnan(mask).sum())
print("Number of Infs in mask:", np.isinf(mask).sum())

# Convert data and mask to tensors
norm_data_tensor = tf.convert_to_tensor(norm_data, dtype=tf.float32)
mask_tensor = tf.convert_to_tensor(mask, dtype=tf.float32)

# Define Model Dimensions
dim = norm_data.shape[1]
h_dim = dim * 2  # Number of hidden units

# Build the Generator Model
def build_generator():
    inputs = layers.Input(shape=(dim * 2,))
    x = layers.Dense(h_dim)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dense(h_dim)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dense(h_dim // 2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    outputs = layers.Dense(dim, activation='sigmoid')(x)
    model = models.Model(inputs, outputs)
    return model

# Build the Discriminator Model
def build_discriminator():
    inputs = layers.Input(shape=(dim * 2,))
    x = layers.Dense(h_dim)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dense(h_dim)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    x = layers.Dense(h_dim // 2)(x)
    x = layers.BatchNormalization()(x)
    x = layers.LeakyReLU()(x)
    outputs = layers.Dense(dim, activation='sigmoid')(x)
    model = models.Model(inputs, outputs)
    return model

generator = build_generator()
discriminator = build_discriminator()

# Define Loss Functions and Optimizers with Learning Rate Scheduling
alpha = 0.7  # Hint rate
batch_size = 64  # Batch size
epochs = 1000  # Number of epochs for training

# Adjusted learning rate schedulers
lr_schedule_gen = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.0005,  # Reduced learning rate
    decay_steps=1000,
    decay_rate=0.96,
    staircase=True
)

lr_schedule_disc = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.0005,  # Reduced learning rate
    decay_steps=1000,
    decay_rate=0.96,
    staircase=True
)

generator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule_gen)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule_disc)

# Prepare the Dataset for Training
dataset_size = norm_data_tensor.shape[0]
dataset = tf.data.Dataset.from_tensor_slices((norm_data_tensor, mask_tensor))
dataset = dataset.shuffle(buffer_size=dataset_size).batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

# Function to sample hint vectors using TensorFlow
def sample_hint_tf(mask, alpha):
    random_prob = tf.random.uniform(shape=tf.shape(mask), minval=0, maxval=1)
    hint_matrix = tf.cast(random_prob < alpha, tf.float32)
    hint_matrix = hint_matrix * mask
    return hint_matrix

# Function to impute missing data
@tf.function
def impute_data(generator, data, mask, alpha):
    Z = tf.random.uniform(shape=tf.shape(data), minval=0., maxval=0.01)
    X_hat = data * mask + Z * (1 - mask)
    H = sample_hint_tf(mask, alpha)
    G_input = tf.concat([X_hat, H], axis=1)
    G_sample = generator(G_input, training=False)
    imputed_data = data * mask + G_sample * (1 - mask)
    return imputed_data

# Function to destandardize the data
def destandardize(imputed_data, mean_val, std_val):
    imputed_data = imputed_data * std_val + mean_val
    return imputed_data

# Training Loop with Early Stopping
generator_losses = []
discriminator_losses = []
patience = 100  # Early stopping patience
best_rmse = np.inf
patience_counter = 0

# Function to compute RMSE
def rmse_loss(original_data, imputed_data, mask):
    numerator = tf.reduce_sum(((1 - mask) * (original_data - imputed_data)) ** 2)
    denominator = tf.reduce_sum(1 - mask)
    rmse = tf.sqrt(numerator / (denominator + 1e-8))
    return rmse

# Compile the training step into a TensorFlow graph
@tf.function
def train_step(batch_data, batch_mask):
    batch_size_actual = tf.shape(batch_data)[0]

    # Sample random noise Z
    Z = tf.random.uniform(shape=[batch_size_actual, dim], minval=0., maxval=0.01)

    # Masked data with noise
    X_hat = batch_data * batch_mask + Z * (1 - batch_mask)

    # Sample hint vectors
    H = sample_hint_tf(batch_mask, alpha)

    # Concatenate Mask and Hint to Generator input
    G_input = tf.concat([X_hat, H], axis=1)

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        # Generator forward pass
        G_sample = generator(G_input, training=True)
        tf.debugging.assert_all_finite(G_sample, message='G_sample contains NaNs or Infs')

        # Combine real and generated data
        X_hat2 = batch_data * batch_mask + G_sample * (1 - batch_mask)

        # Concatenate Mask and Hint to Discriminator input
        D_input = tf.concat([X_hat2, H], axis=1)

        # Discriminator forward pass
        D_prob = discriminator(D_input, training=True)
        epsilon = 1e-7
        D_prob = tf.clip_by_value(D_prob, epsilon, 1. - epsilon)
        tf.debugging.assert_all_finite(D_prob, message='D_prob contains NaNs or Infs')

        # Generator loss
        G_loss = -tf.reduce_mean((1 - batch_mask) * tf.math.log(D_prob))
        tf.debugging.assert_all_finite(G_loss, message='G_loss contains NaNs or Infs')

        # Discriminator loss
        D_loss = -tf.reduce_mean(
            batch_mask * tf.math.log(D_prob) +
            (1 - batch_mask) * tf.math.log(1. - D_prob)
        )
        tf.debugging.assert_all_finite(D_loss, message='D_loss contains NaNs or Infs')

    # Compute gradients
    gen_gradients = gen_tape.gradient(G_loss, generator.trainable_variables)
    disc_gradients = disc_tape.gradient(D_loss, discriminator.trainable_variables)

    # Apply gradient clipping
    gen_gradients = [tf.clip_by_norm(g, 5.0) for g in gen_gradients]
    disc_gradients = [tf.clip_by_norm(g, 5.0) for g in disc_gradients]

    # Apply gradients
    generator_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))

    return G_loss, D_loss

# Training Loop
for epoch in range(epochs):
    nan_in_loss = False  # Flag to check for NaNs in loss
    for batch_data, batch_mask in dataset:
        G_loss, D_loss = train_step(batch_data, batch_mask)

        # Check for NaNs in losses
        if np.isnan(G_loss.numpy()) or np.isnan(D_loss.numpy()):
            print(f"NaN detected in losses at epoch {epoch + 1}. Stopping training.")
            nan_in_loss = True
            break

    if nan_in_loss:
        break

    # Record losses
    generator_losses.append(G_loss.numpy())
    discriminator_losses.append(D_loss.numpy())

    # Early stopping and logging every 100 epochs
    if (epoch + 1) % 100 == 0:
        print(f'Epoch {epoch + 1}, Generator Loss: {G_loss.numpy()}, Discriminator Loss: {D_loss.numpy()}')

        # Impute missing data
        imputed_data_norm = impute_data(generator, norm_data_tensor, mask_tensor, alpha)
        imputed_data_norm_np = imputed_data_norm.numpy()
        imputed_data = destandardize(imputed_data_norm_np, mean_val, std_val)

        # Calculate RMSE
        rmse = rmse_loss(original_data_array, imputed_data, mask).numpy()
        print(f'Epoch {epoch + 1}, Imputation RMSE: {rmse}')

        # Check if RMSE is valid number
        if np.isnan(rmse):
            print(f"RMSE is NaN at epoch {epoch + 1}, skipping model saving.")
        elif rmse < best_rmse:
            best_rmse = rmse
            patience_counter = 0
            # Save the best model
            try:
                generator.save('best_generator.h5')
                print(f"Model saved at epoch {epoch + 1} with RMSE: {rmse}")
            except Exception as e:
                print(f"Error saving model: {e}")
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch + 1}')
            break

# After training, check if the model file exists
if os.path.exists('best_generator.h5'):
    print("Model file 'best_generator.h5' exists.")
else:
    print("Model file 'best_generator.h5' was not found.")

# Load the best model
if os.path.exists('best_generator.h5'):
    try:
        generator = tf.keras.models.load_model('best_generator.h5', compile=False)
        print("Best generator model loaded successfully.")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print("Cannot load 'best_generator.h5' because it does not exist.")
    # Optionally, proceed with the current generator or handle the error appropriately.

# Impute missing data with the best model
imputed_data_norm = impute_data(generator, norm_data_tensor, mask_tensor, alpha)
imputed_data_norm_np = imputed_data_norm.numpy()
imputed_data = destandardize(imputed_data_norm_np, mean_val, std_val)

# Debugging statements
print("imputed_data shape:", imputed_data.shape)
print("Number of NaNs in imputed_data:", np.isnan(imputed_data).sum())
print("imputed_data min/max:", np.min(imputed_data), np.max(imputed_data))
print("imputed_data first few rows:\n", imputed_data[:5])

# Combine Imputed Numeric Data with Non-Numeric Data
imputed_numeric_df = pd.DataFrame(imputed_data, columns=numeric_cols)

# Debugging statements
print("imputed_numeric_df shape:", imputed_numeric_df.shape)
print("imputed_numeric_df columns:", imputed_numeric_df.columns)
print("imputed_numeric_df head:\n", imputed_numeric_df.head())

# Include non-numeric columns in the final DataFrame
if len(non_numeric_cols) > 0:
    non_numeric_data = data_miss[non_numeric_cols].reset_index(drop=True)
    imputed_full_df = pd.concat([imputed_numeric_df, non_numeric_data], axis=1)
else:
    imputed_full_df = imputed_numeric_df

# Debugging statements
print("imputed_full_df shape:", imputed_full_df.shape)
print("imputed_full_df columns:", imputed_full_df.columns)
print("imputed_full_df head:\n", imputed_full_df.head())

# Save the imputed data to a CSV file
imputed_full_df.to_csv('imputed_data_mcar.csv', index=False)
print("Imputed data saved to 'imputed_data_mnar.csv'.")

# Plot the Generator and Discriminator Losses
plt.figure(figsize=(10, 6))
plt.plot(generator_losses, label='Generator Loss')
plt.plot(discriminator_losses, label='Discriminator Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training Losses')
plt.show()
