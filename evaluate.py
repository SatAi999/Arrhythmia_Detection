"""
Comprehensive Evaluation Module for ECG Anomaly Detection

Features:
- Multiple classification metrics (Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC)
- Confusion matrix visualization
- ROC and Precision-Recall curves
- Reconstruction error distribution plots
- Latent space visualization (t-SNE, PCA, UMAP)
- Per-class performance analysis
- Detailed evaluation report generation
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, roc_curve,
    precision_recall_curve, confusion_matrix, classification_report
)
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import yaml
import os
import json
from typing import Dict, Tuple

# Optional: UMAP
try:
    from umap import UMAP
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


class ModelEvaluator:
    """
    Comprehensive evaluation system for anomaly detection models
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize evaluator with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.eval_config = self.config['evaluation']
        self.results_dir = self.config['paths']['results_dir']
        
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Set plotting style
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 300
    
    def compute_metrics(
        self, 
        y_true: np.ndarray,
        y_pred: np.ndarray,
        scores: np.ndarray
    ) -> Dict[str, float]:
        """
        Compute all classification metrics
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            scores: Anomaly scores
        
        Returns:
            Dictionary of metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred, zero_division=0)
        
        # ROC-AUC
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, scores)
        except ValueError:
            metrics['roc_auc'] = 0.0
        
        # PR-AUC
        try:
            metrics['pr_auc'] = average_precision_score(y_true, scores)
        except ValueError:
            metrics['pr_auc'] = 0.0
        
        # Specificity (True Negative Rate)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        
        # False Positive Rate
        metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        
        # False Negative Rate
        metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        return metrics
    
    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: str = None
    ):
        """
        Plot confusion matrix
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save figure
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        
        # Plot heatmap
        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            square=True,
            cbar_kws={'label': 'Count'},
            annot_kws={'size': 16}
        )
        
        plt.xlabel('Predicted Label', fontsize=14, fontweight='bold')
        plt.ylabel('True Label', fontsize=14, fontweight='bold')
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xticks([0.5, 1.5], ['Normal (0)', 'Anomaly (1)'], fontsize=12)
        plt.yticks([0.5, 1.5], ['Normal (0)', 'Anomaly (1)'], fontsize=12, rotation=0)
        
        # Add percentage annotations
        for i in range(2):
            for j in range(2):
                percentage = cm[i, j] / cm.sum() * 100
                plt.text(
                    j + 0.5, i + 0.7,
                    f'({percentage:.1f}%)',
                    ha='center', va='center',
                    fontsize=10, color='gray'
                )
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.results_dir, 'confusion_matrix.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Confusion matrix saved to {save_path}")
        plt.close()
    
    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        scores: np.ndarray,
        save_path: str = None
    ):
        """
        Plot ROC curve
        
        Args:
            y_true: True labels
            scores: Anomaly scores
            save_path: Path to save figure
        """
        fpr, tpr, thresholds = roc_curve(y_true, scores)
        roc_auc = roc_auc_score(y_true, scores)
        
        plt.figure(figsize=(10, 8))
        
        # Plot ROC curve
        plt.plot(
            fpr, tpr,
            color='darkorange',
            lw=3,
            label=f'ROC Curve (AUC = {roc_auc:.3f})'
        )
        
        # Plot diagonal (random classifier)
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=14, fontweight='bold')
        plt.ylabel('True Positive Rate', fontsize=14, fontweight='bold')
        plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16, fontweight='bold')
        plt.legend(loc="lower right", fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.results_dir, 'roc_curve.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
        plt.close()
    
    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        scores: np.ndarray,
        save_path: str = None
    ):
        """
        Plot Precision-Recall curve
        
        Args:
            y_true: True labels
            scores: Anomaly scores
            save_path: Path to save figure
        """
        precision, recall, thresholds = precision_recall_curve(y_true, scores)
        pr_auc = average_precision_score(y_true, scores)
        
        plt.figure(figsize=(10, 8))
        
        # Plot PR curve
        plt.plot(
            recall, precision,
            color='green',
            lw=3,
            label=f'PR Curve (AUC = {pr_auc:.3f})'
        )
        
        # Plot baseline (proportion of positive class)
        baseline = np.sum(y_true) / len(y_true)
        plt.plot([0, 1], [baseline, baseline], color='gray', lw=2, linestyle='--', label=f'Baseline ({baseline:.3f})')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('Recall', fontsize=14, fontweight='bold')
        plt.ylabel('Precision', fontsize=14, fontweight='bold')
        plt.title('Precision-Recall Curve', fontsize=16, fontweight='bold')
        plt.legend(loc="best", fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.results_dir, 'precision_recall_curve.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Precision-Recall curve saved to {save_path}")
        plt.close()
    
    def plot_error_distribution(
        self,
        y_true: np.ndarray,
        scores: np.ndarray,
        threshold: float = None,
        save_path: str = None
    ):
        """
        Plot reconstruction error distribution
        
        Args:
            y_true: True labels
            scores: Reconstruction errors
            threshold: Anomaly threshold
            save_path: Path to save figure
        """
        normal_scores = scores[y_true == 0]
        anomaly_scores = scores[y_true == 1]
        
        plt.figure(figsize=(12, 6))
        
        # Plot histograms
        plt.hist(
            normal_scores,
            bins=50,
            alpha=0.6,
            color='blue',
            label=f'Normal (n={len(normal_scores)})',
            density=True
        )
        plt.hist(
            anomaly_scores,
            bins=50,
            alpha=0.6,
            color='red',
            label=f'Anomaly (n={len(anomaly_scores)})',
            density=True
        )
        
        # Plot threshold
        if threshold is not None:
            plt.axvline(
                threshold,
                color='green',
                linestyle='--',
                linewidth=2,
                label=f'Threshold = {threshold:.4f}'
            )
        
        plt.xlabel('Reconstruction Error', fontsize=14, fontweight='bold')
        plt.ylabel('Density', fontsize=14, fontweight='bold')
        plt.title('Reconstruction Error Distribution', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.results_dir, 'error_distribution.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Error distribution saved to {save_path}")
        plt.close()
    
    def plot_latent_space(
        self,
        z: np.ndarray,
        y: np.ndarray,
        method: str = 'tsne',
        save_path: str = None
    ):
        """
        Visualize latent space using dimensionality reduction
        
        Args:
            z: Latent representations
            y: Labels
            method: Visualization method ('tsne', 'pca', 'umap')
            save_path: Path to save figure
        """
        print(f"Computing {method.upper()} for latent space visualization...")
        
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
            z_2d = reducer.fit_transform(z)
        elif method == 'pca':
            reducer = PCA(n_components=2, random_state=42)
            z_2d = reducer.fit_transform(z)
        elif method == 'umap':
            if not UMAP_AVAILABLE:
                print("UMAP not available. Install with: pip install umap-learn")
                print("Falling back to t-SNE...")
                method = 'tsne'
                reducer = TSNE(n_components=2, random_state=42, perplexity=30)
                z_2d = reducer.fit_transform(z)
            else:
                reducer = UMAP(n_components=2, random_state=42)
                z_2d = reducer.fit_transform(z)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Plot
        plt.figure(figsize=(12, 10))
        
        # Plot normal samples
        normal_idx = y == 0
        plt.scatter(
            z_2d[normal_idx, 0],
            z_2d[normal_idx, 1],
            c='blue',
            alpha=0.5,
            s=20,
            label=f'Normal (n={np.sum(normal_idx)})',
            edgecolors='none'
        )
        
        # Plot anomaly samples
        anomaly_idx = y == 1
        plt.scatter(
            z_2d[anomaly_idx, 0],
            z_2d[anomaly_idx, 1],
            c='red',
            alpha=0.7,
            s=20,
            label=f'Anomaly (n={np.sum(anomaly_idx)})',
            edgecolors='none'
        )
        
        plt.xlabel(f'{method.upper()} Dimension 1', fontsize=14, fontweight='bold')
        plt.ylabel(f'{method.upper()} Dimension 2', fontsize=14, fontweight='bold')
        plt.title(f'Latent Space Visualization ({method.upper()})', fontsize=16, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = os.path.join(self.results_dir, f'latent_space_{method}.png')
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Latent space visualization saved to {save_path}")
        plt.close()
    
    def generate_report(
        self,
        metrics: Dict[str, float],
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: str = None
    ):
        """
        Generate comprehensive evaluation report
        
        Args:
            metrics: Dictionary of metrics
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save report
        """
        if save_path is None:
            save_path = os.path.join(self.results_dir, 'evaluation_report.txt')
        
        with open(save_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("ECG ANOMALY DETECTION - EVALUATION REPORT\n")
            f.write("="*70 + "\n\n")
            
            # Dataset statistics
            f.write("Dataset Statistics:\n")
            f.write("-" * 70 + "\n")
            f.write(f"Total samples: {len(y_true)}\n")
            f.write(f"Normal samples: {np.sum(y_true == 0)} ({100*np.sum(y_true == 0)/len(y_true):.1f}%)\n")
            f.write(f"Anomaly samples: {np.sum(y_true == 1)} ({100*np.sum(y_true == 1)/len(y_true):.1f}%)\n\n")
            
            # Classification metrics
            f.write("Classification Metrics:\n")
            f.write("-" * 70 + "\n")
            for metric_name, value in metrics.items():
                f.write(f"{metric_name.upper():20s}: {value:.4f}\n")
            f.write("\n")
            
            # Confusion matrix
            cm = confusion_matrix(y_true, y_pred)
            f.write("Confusion Matrix:\n")
            f.write("-" * 70 + "\n")
            f.write(f"                Predicted Normal    Predicted Anomaly\n")
            f.write(f"True Normal     {cm[0, 0]:8d}            {cm[0, 1]:8d}\n")
            f.write(f"True Anomaly    {cm[1, 0]:8d}            {cm[1, 1]:8d}\n\n")
            
            # Classification report
            f.write("Detailed Classification Report:\n")
            f.write("-" * 70 + "\n")
            f.write(classification_report(
                y_true, y_pred,
                target_names=['Normal', 'Anomaly'],
                digits=4
            ))
            
            f.write("\n" + "="*70 + "\n")
        
        print(f"Evaluation report saved to {save_path}")
        
        # Also save as JSON
        json_path = save_path.replace('.txt', '.json')
        with open(json_path, 'w') as f:
            json.dump(metrics, f, indent=4)
        print(f"Metrics saved to {json_path}")
    
    def evaluate_full(
        self,
        vae_model,
        X_test: np.ndarray,
        y_test: np.ndarray,
        predictions: np.ndarray,
        scores: np.ndarray,
        threshold: float = None
    ):
        """
        Perform full evaluation with all visualizations
        
        Args:
            vae_model: Trained VAE model
            X_test: Test ECG windows
            y_test: True labels
            predictions: Predicted labels
            scores: Anomaly scores
            threshold: Detection threshold
        """
        print("\n" + "="*70)
        print("RUNNING COMPREHENSIVE EVALUATION")
        print("="*70 + "\n")
        
        # Compute metrics
        print("Computing metrics...")
        metrics = self.compute_metrics(y_test, predictions, scores)
        
        # Print metrics
        print("\nClassification Metrics:")
        print("-" * 70)
        for metric_name, value in metrics.items():
            print(f"{metric_name.upper():20s}: {value:.4f}")
        
        # Plot confusion matrix
        if self.eval_config['visualization']['plot_confusion_matrix']:
            print("\nGenerating confusion matrix...")
            self.plot_confusion_matrix(y_test, predictions)
        
        # Plot ROC curve
        if self.eval_config['visualization']['plot_roc_curve']:
            print("Generating ROC curve...")
            self.plot_roc_curve(y_test, scores)
        
        # Plot PR curve
        if self.eval_config['visualization']['plot_pr_curve']:
            print("Generating Precision-Recall curve...")
            self.plot_precision_recall_curve(y_test, scores)
        
        # Plot error distribution
        if self.eval_config['visualization']['plot_error_distribution']:
            print("Generating error distribution...")
            self.plot_error_distribution(y_test, scores, threshold)
        
        # Plot latent space
        if self.eval_config['visualization']['plot_latent_space']:
            print("Generating latent space visualization...")
            z_test = vae_model.encode(X_test)
            vis_method = self.eval_config['visualization']['latent_visualization']
            self.plot_latent_space(z_test, y_test, method=vis_method)
        
        # Generate report
        print("Generating evaluation report...")
        self.generate_report(metrics, y_test, predictions)
        
        print("\n" + "="*70)
        print("EVALUATION COMPLETE!")
        print("="*70 + "\n")
        
        return metrics


if __name__ == "__main__":
    from preprocessing import ECGPreprocessor
    from anomaly_detection import AnomalyDetector
    from tensorflow import keras
    
    # Load data
    preprocessor = ECGPreprocessor()
    data = preprocessor.load_processed_data()
    
    X_test = data['X_test']
    y_test = data['y_test']
    X_val = data['X_val']
    y_val = data['y_val']
    
    # Load model
    model_path = "saved_models/vae_best_model.h5"
    vae = keras.models.load_model(model_path, compile=False)
    
    # Detect anomalies
    detector = AnomalyDetector(vae)
    detector.fit_thresholds(X_val, y_val)
    predictions, scores_dict = detector.predict(X_test)
    
    # Get primary scores (reconstruction error)
    scores = scores_dict['reconstruction_error']
    threshold = detector.thresholds['reconstruction_error']
    
    # Evaluate
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_full(
        vae, X_test, y_test, predictions, scores, threshold
    )
