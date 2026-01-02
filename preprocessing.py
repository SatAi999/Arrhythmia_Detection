"""
ECG Signal Preprocessing Pipeline
Includes normalization, denoising, augmentation, and train/val/test splitting
"""

import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from scipy.signal import savgol_filter
import yaml
from typing import Tuple, Optional, Dict
import os
import h5py


class ECGPreprocessor:
    """
    Advanced ECG preprocessing pipeline
    
    Features:
    - Multiple normalization strategies (Standard, MinMax)
    - Savitzky-Golay denoising
    - Data augmentation (noise, time shift, amplitude scaling)
    - Stratified train/val/test splitting
    - Preserves only normal beats for training (semi-supervised)
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize preprocessor with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.preproc_config = self.config['preprocessing']
        self.dataset_config = self.config['dataset']
        
        # Normalization method
        self.normalization = self.preproc_config['normalization']
        if self.normalization == 'standard':
            self.scaler = StandardScaler()
        elif self.normalization == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown normalization: {self.normalization}")
        
        # Denoising
        self.denoising = self.preproc_config['denoising']
        self.savgol_window = self.preproc_config['savgol_window']
        self.savgol_polyorder = self.preproc_config['savgol_polyorder']
        
        # Augmentation
        self.augmentation_config = self.preproc_config['augmentation']
        
        # Splits
        self.train_ratio = self.dataset_config['train_ratio']
        self.val_ratio = self.dataset_config['val_ratio']
        self.test_ratio = self.dataset_config['test_ratio']
    
    def denoise(self, X: np.ndarray) -> np.ndarray:
        """
        Apply Savitzky-Golay filter for denoising
        
        Args:
            X: ECG segments (num_samples, window_length)
        
        Returns:
            Denoised ECG segments
        """
        if not self.denoising:
            return X
        
        X_denoised = np.zeros_like(X)
        for i in range(len(X)):
            X_denoised[i] = savgol_filter(
                X[i], 
                window_length=self.savgol_window,
                polyorder=self.savgol_polyorder
            )
        
        return X_denoised
    
    def normalize(self, X: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Normalize ECG segments
        
        Args:
            X: ECG segments (num_samples, window_length)
            fit: Whether to fit scaler (True for train, False for val/test)
        
        Returns:
            Normalized ECG segments
        """
        original_shape = X.shape
        X_reshaped = X.reshape(-1, 1)
        
        if fit:
            X_normalized = self.scaler.fit_transform(X_reshaped)
        else:
            X_normalized = self.scaler.transform(X_reshaped)
        
        return X_normalized.reshape(original_shape)
    
    def augment_data(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply data augmentation to ECG segments
        
        Augmentation techniques:
        1. Gaussian noise addition
        2. Time shifting
        3. Amplitude scaling
        
        Args:
            X: ECG segments (num_samples, window_length)
            y: Labels
        
        Returns:
            Augmented X and y (doubled in size)
        """
        if not self.augmentation_config['enabled']:
            return X, y
        
        aug_config = self.augmentation_config
        X_aug_list = [X]
        y_aug_list = [y]
        
        # 1. Gaussian noise
        noise_std = aug_config['gaussian_noise_std']
        X_noise = X + np.random.normal(0, noise_std, X.shape)
        X_aug_list.append(X_noise)
        y_aug_list.append(y)
        
        # 2. Time shifting
        max_shift = aug_config['time_shift_max']
        X_shifted = np.zeros_like(X)
        for i in range(len(X)):
            shift = np.random.randint(-max_shift, max_shift + 1)
            X_shifted[i] = np.roll(X[i], shift)
        X_aug_list.append(X_shifted)
        y_aug_list.append(y)
        
        # 3. Amplitude scaling
        scale_range = aug_config['amplitude_scale_range']
        scales = np.random.uniform(scale_range[0], scale_range[1], len(X))
        X_scaled = X * scales[:, np.newaxis]
        X_aug_list.append(X_scaled)
        y_aug_list.append(y)
        
        # Concatenate all augmentations
        X_augmented = np.concatenate(X_aug_list, axis=0)
        y_augmented = np.concatenate(y_aug_list, axis=0)
        
        # Shuffle
        indices = np.random.permutation(len(X_augmented))
        X_augmented = X_augmented[indices]
        y_augmented = y_augmented[indices]
        
        return X_augmented, y_augmented
    
    def split_data(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        train_normal_only: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into train/val/test sets
        
        For semi-supervised anomaly detection:
        - Training: Only normal beats (label=0)
        - Validation: Normal + anomalies (for threshold tuning)
        - Testing: Normal + anomalies (for evaluation)
        
        Args:
            X: ECG segments
            y: Labels (0=normal, 1=anomaly)
            train_normal_only: Use only normal beats for training
        
        Returns:
            X_train, y_train, X_val, y_val, X_test, y_test
        """
        # Separate normal and anomaly samples
        normal_indices = np.where(y == 0)[0]
        anomaly_indices = np.where(y == 1)[0]
        
        X_normal = X[normal_indices]
        y_normal = y[normal_indices]
        
        X_anomaly = X[anomaly_indices]
        y_anomaly = y[anomaly_indices]
        
        # Split normal samples
        val_test_ratio = self.val_ratio + self.test_ratio
        X_train_normal, X_temp_normal, y_train_normal, y_temp_normal = train_test_split(
            X_normal, y_normal, 
            test_size=val_test_ratio, 
            random_state=42
        )
        
        val_ratio_adjusted = self.val_ratio / val_test_ratio
        X_val_normal, X_test_normal, y_val_normal, y_test_normal = train_test_split(
            X_temp_normal, y_temp_normal,
            test_size=(1 - val_ratio_adjusted),
            random_state=42
        )
        
        # Split anomaly samples (only for val and test)
        if len(X_anomaly) > 0:
            val_ratio_anomaly = self.val_ratio / val_test_ratio
            X_val_anomaly, X_test_anomaly, y_val_anomaly, y_test_anomaly = train_test_split(
                X_anomaly, y_anomaly,
                test_size=(1 - val_ratio_anomaly),
                random_state=42
            )
            
            # Combine normal and anomaly for val/test
            X_val = np.concatenate([X_val_normal, X_val_anomaly], axis=0)
            y_val = np.concatenate([y_val_normal, y_val_anomaly], axis=0)
            
            X_test = np.concatenate([X_test_normal, X_test_anomaly], axis=0)
            y_test = np.concatenate([y_test_normal, y_test_anomaly], axis=0)
        else:
            X_val = X_val_normal
            y_val = y_val_normal
            X_test = X_test_normal
            y_test = y_test_normal
        
        # Training set (normal only for semi-supervised learning)
        if train_normal_only:
            X_train = X_train_normal
            y_train = y_train_normal
        else:
            # Include anomalies if doing supervised learning
            X_train = X_train_normal
            y_train = y_train_normal
        
        # Shuffle val and test sets
        val_indices = np.random.permutation(len(X_val))
        X_val = X_val[val_indices]
        y_val = y_val[val_indices]
        
        test_indices = np.random.permutation(len(X_test))
        X_test = X_test[test_indices]
        y_test = y_test[test_indices]
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def prepare_for_model(self, X: np.ndarray) -> np.ndarray:
        """
        Reshape data for model input (add channel dimension)
        
        Args:
            X: ECG segments (num_samples, window_length)
        
        Returns:
            Reshaped data (num_samples, window_length, 1)
        """
        return X.reshape(X.shape[0], X.shape[1], 1)
    
    def process_pipeline(
        self, 
        X: np.ndarray, 
        y: np.ndarray,
        augment_train: bool = True,
        save_processed: bool = True
    ) -> Dict[str, np.ndarray]:
        """
        Complete preprocessing pipeline
        
        Steps:
        1. Denoise
        2. Split into train/val/test
        3. Normalize (fit on train only)
        4. Augment training data
        5. Reshape for model
        
        Args:
            X: Raw ECG segments
            y: Labels
            augment_train: Apply augmentation to training set
            save_processed: Save processed data to disk
        
        Returns:
            Dictionary with train/val/test splits
        """
        print("Starting preprocessing pipeline...")
        
        # Step 1: Denoise
        print("Denoising...")
        X_denoised = self.denoise(X)
        
        # Step 2: Split data
        print("Splitting data...")
        X_train, y_train, X_val, y_val, X_test, y_test = self.split_data(
            X_denoised, y, train_normal_only=True
        )
        
        print(f"Train: {len(X_train)} samples (normal only)")
        print(f"Val: {len(X_val)} samples ({np.sum(y_val==0)} normal, {np.sum(y_val==1)} anomaly)")
        print(f"Test: {len(X_test)} samples ({np.sum(y_test==0)} normal, {np.sum(y_test==1)} anomaly)")
        
        # Step 3: Normalize (fit on train)
        print("Normalizing...")
        X_train = self.normalize(X_train, fit=True)
        X_val = self.normalize(X_val, fit=False)
        X_test = self.normalize(X_test, fit=False)
        
        # Step 4: Augment training data
        if augment_train:
            print("Augmenting training data...")
            X_train, y_train = self.augment_data(X_train, y_train)
            print(f"After augmentation: {len(X_train)} training samples")
        
        # Step 5: Reshape for model
        print("Reshaping for model...")
        X_train = self.prepare_for_model(X_train)
        X_val = self.prepare_for_model(X_val)
        X_test = self.prepare_for_model(X_test)
        
        data = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test
        }
        
        # Save processed data
        if save_processed:
            self.save_processed_data(data)
        
        print("Preprocessing complete!")
        return data
    
    def save_processed_data(self, data: Dict[str, np.ndarray]):
        """Save preprocessed data to HDF5"""
        processed_dir = self.dataset_config['processed_data_dir']
        os.makedirs(processed_dir, exist_ok=True)
        
        filepath = os.path.join(processed_dir, 'preprocessed_data.h5')
        
        with h5py.File(filepath, 'w') as f:
            for key, value in data.items():
                f.create_dataset(key, data=value, compression='gzip')
        
        print(f"Saved preprocessed data to {filepath}")
    
    def load_processed_data(self) -> Dict[str, np.ndarray]:
        """Load preprocessed data from HDF5"""
        processed_dir = self.dataset_config['processed_data_dir']
        filepath = os.path.join(processed_dir, 'preprocessed_data.h5')
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Preprocessed data not found at {filepath}. "
                "Run process_pipeline() first."
            )
        
        data = {}
        with h5py.File(filepath, 'r') as f:
            for key in f.keys():
                data[key] = f[key][:]
        
        print(f"Loaded preprocessed data from {filepath}")
        return data


if __name__ == "__main__":
    from data_loader import WFDBDataLoader
    
    # Load raw data
    loader = WFDBDataLoader()
    
    try:
        X, y = loader.load_from_hdf5()
    except FileNotFoundError:
        print("Processing raw data...")
        X, y = loader.process_all_records(save_hdf5=True)
    
    # Preprocess
    preprocessor = ECGPreprocessor()
    data = preprocessor.process_pipeline(X, y, augment_train=True, save_processed=True)
    
    print("\nFinal shapes:")
    for key, value in data.items():
        print(f"{key}: {value.shape}")
