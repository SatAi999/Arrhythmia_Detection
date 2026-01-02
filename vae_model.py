"""
Advanced Variational Autoencoder (VAE) Architecture for ECG Anomaly Detection

Features:
- Deep convolutional encoder-decoder
- Reparameterization trick with epsilon sampling
- Custom VAE loss (reconstruction + KL divergence)
- Batch normalization and dropout for regularization
- Support for different loss functions (MSE, MAE)
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np
import yaml
from typing import Tuple, Optional


class Sampling(layers.Layer):
    """
    Reparameterization trick: z = mu + exp(0.5 * log_var) * epsilon
    where epsilon ~ N(0, I)
    """
    
    def call(self, inputs):
        """
        Args:
            inputs: [z_mean, z_log_var]
        
        Returns:
            Sampled latent vector z
        """
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAEEncoder(Model):
    """
    Convolutional Encoder for ECG signals
    
    Architecture:
    Input -> Conv1D blocks (Conv+BN+ReLU+Dropout) -> Flatten -> Dense -> [z_mean, z_log_var]
    """
    
    def __init__(self, latent_dim: int, encoder_config: dict):
        super(VAEEncoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.config = encoder_config
        
        # Convolutional layers
        self.conv_layers = []
        filters_list = self.config['conv_filters']
        kernel_sizes = self.config['kernel_sizes']
        strides = self.config['strides']
        dropout_rate = self.config['dropout_rate']
        use_batch_norm = self.config['batch_norm']
        
        for i, (filters, kernel, stride) in enumerate(zip(filters_list, kernel_sizes, strides)):
            # Conv1D
            conv = layers.Conv1D(
                filters=filters,
                kernel_size=kernel,
                strides=stride,
                padding='same',
                activation=None,
                name=f'encoder_conv_{i+1}'
            )
            self.conv_layers.append(conv)
            
            # Batch Normalization
            if use_batch_norm:
                bn = layers.BatchNormalization(name=f'encoder_bn_{i+1}')
                self.conv_layers.append(bn)
            
            # Activation
            act = layers.ReLU(name=f'encoder_relu_{i+1}')
            self.conv_layers.append(act)
            
            # Dropout
            dropout = layers.Dropout(dropout_rate, name=f'encoder_dropout_{i+1}')
            self.conv_layers.append(dropout)
        
        # Flatten and Dense
        self.flatten = layers.Flatten(name='encoder_flatten')
        self.dense = layers.Dense(256, activation='relu', name='encoder_dense')
        
        # Latent space
        self.z_mean = layers.Dense(latent_dim, name='z_mean')
        self.z_log_var = layers.Dense(latent_dim, name='z_log_var')
        self.sampling = Sampling()
    
    def call(self, inputs, training=None):
        """
        Forward pass
        
        Args:
            inputs: ECG windows (batch_size, window_length, 1)
            training: Training mode flag
        
        Returns:
            z: Sampled latent vector
            z_mean: Mean of latent distribution
            z_log_var: Log variance of latent distribution
        """
        x = inputs
        
        # Convolutional blocks
        for layer in self.conv_layers:
            x = layer(x, training=training)
        
        # Flatten and dense
        x = self.flatten(x)
        x = self.dense(x, training=training)
        
        # Latent distribution parameters
        z_mean = self.z_mean(x)
        z_log_var = self.z_log_var(x)
        
        # Sample from latent space
        z = self.sampling([z_mean, z_log_var])
        
        return z, z_mean, z_log_var


class VAEDecoder(Model):
    """
    Convolutional Decoder for ECG reconstruction
    
    Architecture:
    z -> Dense -> Reshape -> Conv1DTranspose blocks -> Output
    """
    
    def __init__(self, window_length: int, decoder_config: dict):
        super(VAEDecoder, self).__init__()
        
        self.window_length = window_length
        self.config = decoder_config
        
        # Calculate dimensions after encoder
        # Assuming 4 Conv1D layers with strides [2, 2, 2, 1]
        self.encoded_length = window_length // 8  # After 3 stride-2 convolutions
        self.encoded_filters = decoder_config['conv_transpose_filters'][0]
        
        # Dense and reshape
        self.dense = layers.Dense(
            self.encoded_length * self.encoded_filters,
            activation='relu',
            name='decoder_dense'
        )
        self.reshape = layers.Reshape(
            (self.encoded_length, self.encoded_filters),
            name='decoder_reshape'
        )
        
        # Transposed convolutional layers
        self.deconv_layers = []
        filters_list = decoder_config['conv_transpose_filters']
        kernel_sizes = decoder_config['kernel_sizes']
        strides = decoder_config['strides']
        dropout_rate = decoder_config['dropout_rate']
        use_batch_norm = decoder_config['batch_norm']
        
        for i, (filters, kernel, stride) in enumerate(zip(filters_list, kernel_sizes, strides)):
            # Conv1DTranspose
            deconv = layers.Conv1DTranspose(
                filters=filters,
                kernel_size=kernel,
                strides=stride,
                padding='same',
                activation=None,
                name=f'decoder_deconv_{i+1}'
            )
            self.deconv_layers.append(deconv)
            
            # Batch Normalization
            if use_batch_norm and i < len(filters_list) - 1:  # No BN on last layer
                bn = layers.BatchNormalization(name=f'decoder_bn_{i+1}')
                self.deconv_layers.append(bn)
            
            # Activation (ReLU for hidden, Linear for output)
            if i < len(filters_list) - 1:
                act = layers.ReLU(name=f'decoder_relu_{i+1}')
                self.deconv_layers.append(act)
                
                # Dropout
                dropout = layers.Dropout(dropout_rate, name=f'decoder_dropout_{i+1}')
                self.deconv_layers.append(dropout)
        
        # Final output layer
        self.output_layer = layers.Conv1D(
            filters=1,
            kernel_size=7,
            padding='same',
            activation='linear',
            name='decoder_output'
        )
    
    def call(self, inputs, training=None):
        """
        Forward pass
        
        Args:
            inputs: Latent vector z (batch_size, latent_dim)
            training: Training mode flag
        
        Returns:
            Reconstructed ECG (batch_size, window_length, 1)
        """
        x = self.dense(inputs, training=training)
        x = self.reshape(x)
        
        # Transposed convolutional blocks
        for layer in self.deconv_layers:
            x = layer(x, training=training)
        
        # Final output
        x = self.output_layer(x)
        
        # Ensure correct output length
        if x.shape[1] != self.window_length:
            x = tf.image.resize(
                tf.expand_dims(x, -1),
                [self.window_length, 1]
            )
            x = tf.squeeze(x, -1)
        
        return x


class VAE(Model):
    """
    Complete Variational Autoencoder for ECG Anomaly Detection
    
    Combines Encoder and Decoder with custom VAE loss
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        super(VAE, self).__init__()
        
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        model_config = self.config['model']
        self.latent_dim = model_config['latent_dim']
        self.beta = model_config['beta']  # KL divergence weight
        self.reconstruction_loss_type = model_config['reconstruction_loss']
        
        window_length = self.config['dataset']['window_length']
        
        # Build encoder and decoder
        self.encoder = VAEEncoder(
            latent_dim=self.latent_dim,
            encoder_config=model_config['encoder']
        )
        self.decoder = VAEDecoder(
            window_length=window_length,
            decoder_config=model_config['decoder']
        )
        
        # Metrics
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="reconstruction_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")
    
    @property
    def metrics(self):
        """Return list of tracked metrics"""
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]
    
    def call(self, inputs, training=None):
        """
        Forward pass through VAE
        
        Args:
            inputs: ECG windows (batch_size, window_length, 1)
            training: Training mode flag
        
        Returns:
            Reconstructed ECG
        """
        z, z_mean, z_log_var = self.encoder(inputs, training=training)
        reconstruction = self.decoder(z, training=training)
        return reconstruction
    
    def compute_loss(self, x, reconstruction, z_mean, z_log_var):
        """
        Compute VAE loss = Reconstruction Loss + Beta * KL Divergence
        
        Args:
            x: Original ECG
            reconstruction: Reconstructed ECG
            z_mean: Mean of latent distribution
            z_log_var: Log variance of latent distribution
        
        Returns:
            Total loss, reconstruction loss, KL loss
        """
        # Reconstruction loss
        if self.reconstruction_loss_type == 'mse':
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.square(x - reconstruction),
                    axis=[1, 2]
                )
            )
        elif self.reconstruction_loss_type == 'mae':
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    tf.abs(x - reconstruction),
                    axis=[1, 2]
                )
            )
        else:
            raise ValueError(f"Unknown reconstruction loss: {self.reconstruction_loss_type}")
        
        # KL divergence loss
        kl_loss = -0.5 * tf.reduce_mean(
            tf.reduce_sum(
                1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
                axis=1
            )
        )
        
        # Total loss
        total_loss = reconstruction_loss + self.beta * kl_loss
        
        return total_loss, reconstruction_loss, kl_loss
    
    def train_step(self, data):
        """
        Custom training step
        
        Args:
            data: Batch of ECG windows
        
        Returns:
            Dictionary of losses
        """
        with tf.GradientTape() as tape:
            # Forward pass
            z, z_mean, z_log_var = self.encoder(data, training=True)
            reconstruction = self.decoder(z, training=True)
            
            # Compute loss
            total_loss, reconstruction_loss, kl_loss = self.compute_loss(
                data, reconstruction, z_mean, z_log_var
            )
        
        # Backward pass
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        
        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
    def test_step(self, data):
        """
        Custom test step
        
        Args:
            data: Batch of ECG windows
        
        Returns:
            Dictionary of losses
        """
        # Forward pass
        z, z_mean, z_log_var = self.encoder(data, training=False)
        reconstruction = self.decoder(z, training=False)
        
        # Compute loss
        total_loss, reconstruction_loss, kl_loss = self.compute_loss(
            data, reconstruction, z_mean, z_log_var
        )
        
        # Update metrics
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        
        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }
    
    def encode(self, x):
        """
        Encode ECG to latent space
        
        Args:
            x: ECG windows
        
        Returns:
            z_mean: Latent representation (deterministic)
        """
        _, z_mean, _ = self.encoder(x, training=False)
        return z_mean
    
    def decode(self, z):
        """
        Decode latent vector to ECG
        
        Args:
            z: Latent vectors
        
        Returns:
            Reconstructed ECG
        """
        return self.decoder(z, training=False)
    
    def reconstruct(self, x):
        """
        Reconstruct ECG (encode -> decode)
        
        Args:
            x: Original ECG windows
        
        Returns:
            Reconstructed ECG windows
        """
        z, _, _ = self.encoder(x, training=False)
        return self.decoder(z, training=False)
    
    def get_config(self):
        """Return configuration for serialization"""
        return {
            'config_path': 'config.yaml'
        }
    
    @classmethod
    def from_config(cls, config):
        """Create VAE from configuration"""
        return cls(config_path=config.get('config_path', 'config.yaml'))


if __name__ == "__main__":
    # Test VAE architecture
    print("Building VAE model...")
    
    vae = VAE()
    
    # Build model with dummy input
    dummy_input = tf.random.normal((32, 187, 1))
    _ = vae(dummy_input)
    
    # Print model summary
    print("\n" + "="*60)
    print("ENCODER")
    print("="*60)
    vae.encoder.summary()
    
    print("\n" + "="*60)
    print("DECODER")
    print("="*60)
    vae.decoder.summary()
    
    print("\n" + "="*60)
    print(f"Total parameters: {vae.count_params():,}")
    print("="*60)
