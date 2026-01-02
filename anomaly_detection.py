"""
Multi-Strategy Anomaly Detection System

Implements multiple anomaly detection strategies:
1. Reconstruction Error
2. Latent Space Distance (Mahalanobis)
3. KL Divergence
4. Ensemble Voting

Features:
- Automatic threshold tuning on validation set
- Multiple threshold methods (statistical, percentile, fixed)
- Ensemble decision making
- Detailed predictions with confidence scores
"""

import numpy as np
import pandas as pd
import yaml
from typing import Tuple, Dict, List
import os
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2
import tensorflow as tf


class AnomalyDetector:
    """
    Advanced multi-strategy anomaly detection system
    """
    
    def __init__(self, vae_model, config_path: str = "config.yaml"):
        """
        Initialize anomaly detector
        
        Args:
            vae_model: Trained VAE model
            config_path: Path to configuration file
        """
        self.vae = vae_model
        
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.anomaly_config = self.config['anomaly_detection']
        self.threshold_method = self.anomaly_config['threshold_method']
        self.use_ensemble = self.anomaly_config['use_ensemble']
        
        # Thresholds for each method (to be computed)
        self.thresholds = {
            'reconstruction_error': None,
            'latent_distance': None,
            'kl_divergence': None
        }
        
        # Normal data statistics (for Mahalanobis distance)
        self.latent_mean = None
        self.latent_cov = None
        self.latent_cov_inv = None
    
    def compute_reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        """
        Compute reconstruction error for each sample
        
        Args:
            X: ECG windows (num_samples, window_length, 1)
        
        Returns:
            Reconstruction errors (num_samples,)
        """
        reconstructions = self.vae.reconstruct(X)
        
        # Mean Squared Error per sample
        errors = np.mean(np.square(X - reconstructions), axis=(1, 2))
        
        return errors
    
    def compute_latent_statistics(self, X_normal: np.ndarray):
        """
        Compute latent space statistics from normal training data
        
        Args:
            X_normal: Normal ECG windows for computing statistics
        """
        # Encode to latent space
        z_normal = self.vae.encode(X_normal)
        
        # Compute mean and covariance
        self.latent_mean = np.mean(z_normal, axis=0)
        self.latent_cov = np.cov(z_normal, rowvar=False)
        
        # Add regularization to avoid singular matrix
        self.latent_cov += np.eye(self.latent_cov.shape[0]) * 1e-6
        
        # Compute inverse covariance for Mahalanobis distance
        self.latent_cov_inv = np.linalg.inv(self.latent_cov)
    
    def compute_latent_distance(self, X: np.ndarray) -> np.ndarray:
        """
        Compute Mahalanobis distance in latent space
        
        Args:
            X: ECG windows
        
        Returns:
            Mahalanobis distances (num_samples,)
        """
        if self.latent_mean is None or self.latent_cov_inv is None:
            raise ValueError("Latent statistics not computed. Call compute_latent_statistics first.")
        
        # Encode to latent space
        z = self.vae.encode(X)
        
        # Compute Mahalanobis distance
        distances = np.array([
            mahalanobis(z_i, self.latent_mean, self.latent_cov_inv)
            for z_i in z
        ])
        
        return distances
    
    def compute_kl_divergence(self, X: np.ndarray) -> np.ndarray:
        """
        Compute KL divergence for each sample
        
        Args:
            X: ECG windows
        
        Returns:
            KL divergences (num_samples,)
        """
        # Get latent distribution parameters
        _, z_mean, z_log_var = self.vae.encoder(X, training=False)
        
        # Compute KL divergence per sample
        kl_divergence = -0.5 * tf.reduce_sum(
            1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var),
            axis=1
        )
        
        return kl_divergence.numpy()
    
    def compute_threshold(
        self, 
        scores: np.ndarray,
        method: str = None,
        k: float = 3.0
    ) -> float:
        """
        Compute anomaly threshold
        
        Args:
            scores: Anomaly scores from normal validation data
            method: Threshold method ('statistical', 'percentile', 'fixed')
            k: Multiplier for standard deviation (for statistical method)
        
        Returns:
            Threshold value
        """
        if method is None:
            method = self.threshold_method
        
        if method == 'statistical':
            # threshold = mean + k * std
            mean = np.mean(scores)
            std = np.std(scores)
            threshold = mean + k * std
        
        elif method == 'percentile':
            # Use percentile
            percentile = self.anomaly_config['threshold_percentile']
            threshold = np.percentile(scores, percentile)
        
        elif method == 'fixed':
            # Use fixed threshold
            threshold = self.anomaly_config['threshold_fixed']
        
        else:
            raise ValueError(f"Unknown threshold method: {method}")
        
        return threshold
    
    def fit_thresholds(
        self, 
        X_val: np.ndarray, 
        y_val: np.ndarray,
        k: float = None
    ):
        """
        Fit thresholds on validation set
        
        Args:
            X_val: Validation ECG windows
            y_val: Validation labels
            k: Threshold multiplier (overrides config if provided)
        """
        if k is None:
            k = self.anomaly_config['threshold_k']
        
        print("Fitting anomaly detection thresholds...")
        
        # Get only normal samples from validation set
        normal_indices = np.where(y_val == 0)[0]
        X_normal = X_val[normal_indices]
        
        # Compute latent statistics
        self.compute_latent_statistics(X_normal)
        
        # Compute scores for each method
        methods_to_use = self.anomaly_config['ensemble_methods'] if self.use_ensemble else ['reconstruction_error']
        
        for method in methods_to_use:
            if method == 'reconstruction_error':
                scores = self.compute_reconstruction_error(X_normal)
            elif method == 'latent_distance':
                scores = self.compute_latent_distance(X_normal)
            elif method == 'kl_divergence':
                scores = self.compute_kl_divergence(X_normal)
            else:
                continue
            
            # Compute threshold
            threshold = self.compute_threshold(scores, k=k)
            self.thresholds[method] = threshold
            
            print(f"  {method}: threshold = {threshold:.6f}")
        
        print("Thresholds fitted successfully!")
    
    def predict_single_method(
        self, 
        X: np.ndarray,
        method: str
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict anomalies using a single method
        
        Args:
            X: ECG windows
            method: Detection method
        
        Returns:
            predictions: Binary predictions (0=normal, 1=anomaly)
            scores: Anomaly scores
        """
        # Compute scores
        if method == 'reconstruction_error':
            scores = self.compute_reconstruction_error(X)
        elif method == 'latent_distance':
            scores = self.compute_latent_distance(X)
        elif method == 'kl_divergence':
            scores = self.compute_kl_divergence(X)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Get threshold
        threshold = self.thresholds[method]
        if threshold is None:
            raise ValueError(f"Threshold for {method} not set. Call fit_thresholds first.")
        
        # Predict
        predictions = (scores > threshold).astype(int)
        
        return predictions, scores
    
    def predict_ensemble(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Predict anomalies using ensemble of methods
        
        Args:
            X: ECG windows
        
        Returns:
            predictions: Binary predictions (0=normal, 1=anomaly)
            all_scores: Dictionary of scores from each method
        """
        methods = self.anomaly_config['ensemble_methods']
        
        predictions_list = []
        all_scores = {}
        
        for method in methods:
            preds, scores = self.predict_single_method(X, method)
            predictions_list.append(preds)
            all_scores[method] = scores
        
        # Majority voting
        predictions_array = np.array(predictions_list)
        ensemble_predictions = (np.mean(predictions_array, axis=0) >= 0.5).astype(int)
        
        return ensemble_predictions, all_scores
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Predict anomalies (ensemble or single method)
        
        Args:
            X: ECG windows
        
        Returns:
            predictions: Binary predictions
            scores: Dictionary of anomaly scores
        """
        if self.use_ensemble:
            return self.predict_ensemble(X)
        else:
            preds, scores = self.predict_single_method(X, 'reconstruction_error')
            return preds, {'reconstruction_error': scores}
    
    def save_predictions(
        self, 
        predictions: np.ndarray,
        y_true: np.ndarray,
        scores: Dict[str, np.ndarray],
        output_path: str
    ):
        """
        Save predictions to CSV
        
        Args:
            predictions: Predicted labels
            y_true: True labels
            scores: Dictionary of anomaly scores
            output_path: Path to save CSV
        """
        # Create dataframe
        df = pd.DataFrame({
            'sample_index': np.arange(len(predictions)),
            'true_label': y_true,
            'predicted_label': predictions
        })
        
        # Add scores
        for method, method_scores in scores.items():
            df[f'{method}_score'] = method_scores
        
        # Save
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        
        print(f"Predictions saved to {output_path}")
    
    def compute_confidence_scores(
        self, 
        scores: Dict[str, np.ndarray]
    ) -> np.ndarray:
        """
        Compute confidence scores for predictions
        
        Args:
            scores: Dictionary of anomaly scores
        
        Returns:
            Confidence scores (0-1)
        """
        # Normalize scores to [0, 1]
        normalized_scores = []
        
        for method, method_scores in scores.items():
            threshold = self.thresholds[method]
            # Distance from threshold (normalized)
            norm_scores = np.abs(method_scores - threshold) / (threshold + 1e-8)
            normalized_scores.append(norm_scores)
        
        # Average across methods
        confidence = np.mean(normalized_scores, axis=0)
        
        # Clip to [0, 1]
        confidence = np.clip(confidence, 0, 1)
        
        return confidence


if __name__ == "__main__":
    from preprocessing import ECGPreprocessor
    from tensorflow import keras
    
    # Load preprocessed data
    preprocessor = ECGPreprocessor()
    data = preprocessor.load_processed_data()
    
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']
    
    # Load trained model
    model_path = "saved_models/vae_best_model.h5"
    vae = keras.models.load_model(model_path, compile=False)
    
    # Initialize detector
    detector = AnomalyDetector(vae)
    
    # Fit thresholds
    detector.fit_thresholds(X_val, y_val, k=3.0)
    
    # Predict on test set
    predictions, scores = detector.predict(X_test)
    
    # Save predictions
    output_path = "results/predictions/test_predictions.csv"
    detector.save_predictions(predictions, y_test, scores, output_path)
    
    # Print summary
    print("\nPrediction Summary:")
    print(f"Total samples: {len(predictions)}")
    print(f"Predicted normal: {np.sum(predictions == 0)}")
    print(f"Predicted anomaly: {np.sum(predictions == 1)}")
