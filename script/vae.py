# Copyright (c) 2024 Gabriel-Vasilică Sasu
# All rights reserved.
#
# This script implements a Variational Autoencoder (VAE) for imputing missing data in a dataset.
# The process includes:
# 1. Loading the dataset (e.g., sensor data from smart bracelets).
# 2. Preprocessing: Selecting numeric columns, performing initial mean imputation for missing values,
#    and standardizing the data using StandardScaler.
# 3. Defining the VAE architecture:
#    - Encoder: Maps input data to a latent space, outputting mean (z_mean) and log-variance (z_log_var) of the latent distribution.
#    - Sampling Layer: Samples from the latent distribution using z_mean and z_log_var (reparameterization trick).
#    - Decoder: Reconstructs the data from the sampled latent vector.
# 4. Implementing the VAE model as a Keras Model subclass, including a custom `train_step` method
#    that defines the VAE loss (reconstruction loss + KL divergence).
# 5. Compiling and training the VAE model on the preprocessed (scaled and initially imputed) data.
# 6. Using the trained VAE to reconstruct the input data. Since the VAE learns to reconstruct the overall data distribution,
#    the reconstructed output can be seen as an imputed version of the input, especially for the initially mean-imputed values.
# 7. Inverse transforming the reconstructed (and thus imputed) data back to its original scale.
# 8. Saving the VAE-imputed dataset to a new CSV file.

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# Load and preprocess data
data_file = 'prelucrate_2_mnar.csv'  # Update the file path to your data file
data = pd.read_csv(data_file)

# Remove non-numeric columns (like 'time' and 'Activity') if present
numeric_data = data.select_dtypes(include=[np.number])

# Handle missing values using initial imputation (mean imputation)
imputer = SimpleImputer(strategy='mean')
data_imputed = imputer.fit_transform(numeric_data)

# Standardize the data (scaling to zero mean and unit variance)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_imputed)

# Define VAE parameters
input_dim = data_scaled.shape[1]
intermediate_dim = 64
latent_dim = 2

# Build the encoder model
encoder_inputs = keras.Input(shape=(input_dim,))
x = layers.Dense(intermediate_dim, activation='relu')(encoder_inputs)
z_mean = layers.Dense(latent_dim, name='z_mean')(x)
z_log_var = layers.Dense(latent_dim, name='z_log_var')(x)

# Sampling layer using a custom layer subclass
class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

# Apply the sampling layer
z = Sampling()([z_mean, z_log_var])

# Instantiate the encoder model
encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name='encoder')

# Build the decoder model
latent_inputs = keras.Input(shape=(latent_dim,))
x = layers.Dense(intermediate_dim, activation='relu')(latent_inputs)
decoder_outputs = layers.Dense(input_dim)(x)
decoder = keras.Model(latent_inputs, decoder_outputs, name='decoder')

# Define the VAE as a Model subclass
class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
    
    def train_step(self, data):
        if isinstance(data, tuple):
            data = data[0]
        with tf.GradientTape() as tape:
            # Forward pass
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            # Compute reconstruction loss
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.square(data - reconstruction), axis=1
                )
            )
            # Compute KL divergence
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            )
            # Total loss
            total_loss = reconstruction_loss + kl_loss
        # Compute gradients
        grads = tape.gradient(total_loss, self.trainable_weights)
        # Apply gradients
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        # Return loss values
        return {'loss': total_loss, 'reconstruction_loss': reconstruction_loss, 'kl_loss': kl_loss}

# Instantiate the VAE model
vae = VAE(encoder, decoder)
vae.compile(optimizer=keras.optimizers.Adam())

# Train the VAE
vae.fit(data_scaled, epochs=50, batch_size=16)

# Use the VAE to reconstruct the data (impute missing values)
z_mean_pred, z_log_var_pred, z_pred = encoder.predict(data_scaled)
data_reconstructed = decoder.predict(z_pred)

# Inverse transform the scaled data to the original scale
data_imputed_final = scaler.inverse_transform(data_reconstructed)

# Convert the imputed data back into a DataFrame
data_imputed_final_df = pd.DataFrame(data_imputed_final, columns=numeric_data.columns)

# Save the imputed dataset to a CSV file
data_imputed_final_df.to_csv('data_imputed_simple_vae_mnar.csv', index=False)
print("Imputed data saved to 'data_imputed_simple_vae.csv'.")
