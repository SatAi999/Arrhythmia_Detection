"""
Training Pipeline for ECG VAE with Advanced Features

Features:
- Custom callbacks (EarlyStopping, ModelCheckpoint, LearningRateScheduler)
- Training history visualization
- Latent space visualization during training
- Model checkpointing with best model saving
- TensorBoard logging
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
import yaml

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import (
    EarlyStopping, 
    ModelCheckpoint, 
    ReduceLROnPlateau,
    TensorBoard,
    Callback
)

from vae_model import VAE
from preprocessing import ECGPreprocessor


class LatentSpaceVisualizer(Callback):
    """
    Custom callback to visualize latent space during training
    """
    
    def __init__(self, X_val, y_val, save_dir, frequency=10):
        super().__init__()
        self.X_val = X_val
        self.y_val = y_val
        self.save_dir = save_dir
        self.frequency = frequency
        os.makedirs(save_dir, exist_ok=True)
    
    def on_epoch_end(self, epoch, logs=None):
        """Visualize latent space at end of epoch"""
        if (epoch + 1) % self.frequency == 0:
            # Encode validation data
            z_mean = self.model.encode(self.X_val[:1000])
            
            # Plot first 2 latent dimensions
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(
                z_mean[:, 0], 
                z_mean[:, 1],
                c=self.y_val[:1000],
                cmap='RdYlBu',
                alpha=0.6,
                s=10
            )
            plt.colorbar(scatter, label='Label (0=Normal, 1=Anomaly)')
            plt.xlabel('Latent Dimension 1')
            plt.ylabel('Latent Dimension 2')
            plt.title(f'Latent Space Visualization - Epoch {epoch+1}')
            plt.grid(True, alpha=0.3)
            
            filepath = os.path.join(self.save_dir, f'latent_epoch_{epoch+1}.png')
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            plt.close()


class VAETrainer:
    """
    Advanced training pipeline for VAE
    
    Features:
    - Automated data loading
    - Configurable training parameters
    - Multiple callbacks
    - Training history tracking
    - Model checkpointing
    - Visualization
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize trainer with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.training_config = self.config['training']
        self.paths_config = self.config['paths']
        
        # Create directories
        self.models_dir = self.paths_config['models_dir']
        self.results_dir = self.paths_config['results_dir']
        self.logs_dir = self.paths_config['logs_dir']
        
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Training parameters
        self.batch_size = self.training_config['batch_size']
        self.epochs = self.training_config['epochs']
        self.learning_rate = self.training_config['learning_rate']
        
        # Initialize model
        self.vae = None
        self.history = None
    
    def load_data(self):
        """Load preprocessed data"""
        print("Loading preprocessed data...")
        preprocessor = ECGPreprocessor()
        
        try:
            data = preprocessor.load_processed_data()
        except FileNotFoundError:
            print("Preprocessed data not found. Running preprocessing pipeline...")
            from data_loader import WFDBDataLoader
            
            loader = WFDBDataLoader()
            try:
                X, y = loader.load_from_hdf5()
            except FileNotFoundError:
                print("Processing raw WFDB data...")
                X, y = loader.process_all_records(save_hdf5=True)
            
            data = preprocessor.process_pipeline(X, y, augment_train=True, save_processed=True)
        
        self.X_train = data['X_train']
        self.y_train = data['y_train']
        self.X_val = data['X_val']
        self.y_val = data['y_val']
        self.X_test = data['X_test']
        self.y_test = data['y_test']
        
        print(f"\nData loaded successfully:")
        print(f"  Train: {self.X_train.shape}")
        print(f"  Val: {self.X_val.shape}")
        print(f"  Test: {self.X_test.shape}")
        
        return data
    
    def build_model(self):
        """Build and compile VAE model"""
        print("\nBuilding VAE model...")
        self.vae = VAE()
        
        # Build model
        dummy_input = tf.random.normal((1, self.X_train.shape[1], 1))
        _ = self.vae(dummy_input)
        
        # Compile
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.vae.compile(optimizer=optimizer)
        
        print(f"Model built with {self.vae.count_params():,} parameters")
        
        return self.vae
    
    def get_callbacks(self):
        """Create training callbacks"""
        callbacks = []
        
        # Early Stopping
        if self.training_config['early_stopping']['enabled']:
            early_stop = EarlyStopping(
                monitor='val_total_loss',
                patience=self.training_config['early_stopping']['patience'],
                min_delta=self.training_config['early_stopping']['min_delta'],
                restore_best_weights=True,
                mode='min',
                verbose=1
            )
            callbacks.append(early_stop)
        
        # Model Checkpoint
        checkpoint_path = os.path.join(self.models_dir, 'vae_best_model.h5')
        checkpoint = ModelCheckpoint(
            filepath=checkpoint_path,
            monitor='val_total_loss',
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            verbose=1
        )
        callbacks.append(checkpoint)
        
        # Learning Rate Scheduler
        if self.training_config['lr_schedule']['enabled']:
            lr_scheduler = ReduceLROnPlateau(
                monitor='val_total_loss',
                factor=self.training_config['lr_schedule']['factor'],
                patience=self.training_config['lr_schedule']['patience'],
                min_lr=self.training_config['lr_schedule']['min_lr'],
                mode='min',
                verbose=1
            )
            callbacks.append(lr_scheduler)
        
        # TensorBoard
        log_dir = os.path.join(
            self.logs_dir, 
            f"vae_{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        )
        tensorboard = TensorBoard(
            log_dir=log_dir,
            histogram_freq=1,
            write_graph=True
        )
        callbacks.append(tensorboard)
        
        # Latent Space Visualizer
        latent_vis_dir = os.path.join(self.results_dir, 'latent_space')
        latent_visualizer = LatentSpaceVisualizer(
            self.X_val, 
            self.y_val,
            latent_vis_dir,
            frequency=10
        )
        callbacks.append(latent_visualizer)
        
        return callbacks
    
    def train(self):
        """
        Train VAE model
        
        Returns:
            Training history
        """
        print("\n" + "="*60)
        print("Starting VAE Training")
        print("="*60)
        
        # Get callbacks
        callbacks = self.get_callbacks()
        
        # Train model
        self.history = self.vae.fit(
            self.X_train,
            batch_size=self.batch_size,
            epochs=self.epochs,
            validation_data=self.X_val,
            callbacks=callbacks,
            verbose=1
        )
        
        print("\n" + "="*60)
        print("Training Complete!")
        print("="*60)
        
        # Save final model
        final_model_path = os.path.join(self.models_dir, 'vae_final_model.h5')
        self.vae.save(final_model_path)
        print(f"Final model saved to {final_model_path}")
        
        # Save training history
        self.save_training_history()
        
        # Plot training curves
        self.plot_training_history()
        
        return self.history
    
    def save_training_history(self):
        """Save training history to JSON"""
        history_path = os.path.join(self.results_dir, 'training_history.json')
        
        history_dict = {}
        for key, values in self.history.history.items():
            history_dict[key] = [float(v) for v in values]
        
        with open(history_path, 'w') as f:
            json.dump(history_dict, f, indent=4)
        
        print(f"Training history saved to {history_path}")
    
    def plot_training_history(self):
        """Plot and save training curves"""
        history = self.history.history
        
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # Total Loss
        axes[0].plot(history['total_loss'], label='Train', linewidth=2)
        axes[0].plot(history['val_total_loss'], label='Validation', linewidth=2)
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Total Loss', fontsize=12)
        axes[0].set_title('VAE Total Loss', fontsize=14, fontweight='bold')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Reconstruction Loss
        axes[1].plot(history['reconstruction_loss'], label='Train', linewidth=2)
        axes[1].plot(history['val_reconstruction_loss'], label='Validation', linewidth=2)
        axes[1].set_xlabel('Epoch', fontsize=12)
        axes[1].set_ylabel('Reconstruction Loss', fontsize=12)
        axes[1].set_title('Reconstruction Loss', fontsize=14, fontweight='bold')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # KL Divergence
        axes[2].plot(history['kl_loss'], label='Train', linewidth=2)
        axes[2].plot(history['val_kl_loss'], label='Validation', linewidth=2)
        axes[2].set_xlabel('Epoch', fontsize=12)
        axes[2].set_ylabel('KL Divergence', fontsize=12)
        axes[2].set_title('KL Divergence Loss', fontsize=14, fontweight='bold')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = os.path.join(self.results_dir, 'training_curves.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {save_path}")
        
        plt.close()
    
    def visualize_reconstructions(self, num_samples=10):
        """
        Visualize original vs reconstructed ECG signals
        
        Args:
            num_samples: Number of samples to visualize
        """
        # Get random samples from validation set
        indices = np.random.choice(len(self.X_val), num_samples, replace=False)
        samples = self.X_val[indices]
        labels = self.y_val[indices]
        
        # Reconstruct
        reconstructions = self.vae.reconstruct(samples)
        
        # Plot
        fig, axes = plt.subplots(num_samples, 1, figsize=(15, 2.5*num_samples))
        
        for i in range(num_samples):
            ax = axes[i] if num_samples > 1 else axes
            
            # Original
            ax.plot(samples[i, :, 0], label='Original', linewidth=1.5, alpha=0.8)
            # Reconstructed
            ax.plot(reconstructions[i, :, 0], label='Reconstructed', 
                   linewidth=1.5, linestyle='--', alpha=0.8)
            
            label_text = "Normal" if labels[i] == 0 else "Anomaly"
            ax.set_title(f'Sample {i+1} - {label_text}', fontsize=12, fontweight='bold')
            ax.set_xlabel('Time (samples)')
            ax.set_ylabel('Amplitude')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        save_path = os.path.join(self.results_dir, 'reconstructions.png')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Reconstruction visualizations saved to {save_path}")
        
        plt.close()


def main():
    """Main training script"""
    # Initialize trainer
    trainer = VAETrainer()
    
    # Load data
    trainer.load_data()
    
    # Build model
    trainer.build_model()
    
    # Train
    trainer.train()
    
    # Visualize reconstructions
    trainer.visualize_reconstructions(num_samples=10)
    
    print("\n" + "="*60)
    print("Training pipeline completed successfully!")
    print("="*60)


if __name__ == "__main__":
    main()
