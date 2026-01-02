"""
Utility Functions for ECG Analysis and Visualization

Provides helper functions for:
- Signal visualization
- Error computation
- Data export
- Performance profiling
- Model comparison
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from typing import List, Dict, Tuple
import time
from functools import wraps


def plot_ecg_signal(
    signal: np.ndarray,
    sampling_rate: int = 360,
    title: str = "ECG Signal",
    annotations: Dict = None,
    save_path: str = None
):
    """
    Plot ECG signal with optional annotations
    
    Args:
        signal: ECG signal array
        sampling_rate: Sampling rate in Hz
        title: Plot title
        annotations: Dictionary of annotations {sample_idx: label}
        save_path: Path to save figure
    """
    time_axis = np.arange(len(signal)) / sampling_rate
    
    plt.figure(figsize=(15, 5))
    plt.plot(time_axis, signal, linewidth=1, color='blue')
    
    # Add annotations if provided
    if annotations:
        for idx, label in annotations.items():
            plt.axvline(x=idx/sampling_rate, color='red', linestyle='--', alpha=0.5)
            plt.text(idx/sampling_rate, max(signal), label, rotation=90)
    
    plt.xlabel('Time (seconds)', fontsize=12)
    plt.ylabel('Amplitude', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def compute_signal_quality_index(signal: np.ndarray) -> float:
    """
    Compute signal quality index (SQI)
    
    Args:
        signal: ECG signal
    
    Returns:
        Quality score (0-1, higher is better)
    """
    # Simple SQI based on variance and flatline detection
    variance = np.var(signal)
    diff = np.diff(signal)
    flatline_ratio = np.sum(np.abs(diff) < 1e-6) / len(diff)
    
    # Normalize variance (typical ECG variance range)
    normalized_variance = np.clip(variance / 0.1, 0, 1)
    
    # Quality score
    sqi = normalized_variance * (1 - flatline_ratio)
    
    return sqi


def compare_models_performance(
    models_metrics: Dict[str, Dict[str, float]],
    save_path: str = None
) -> pd.DataFrame:
    """
    Compare performance of multiple models
    
    Args:
        models_metrics: Dict of {model_name: {metric: value}}
        save_path: Path to save comparison plot
    
    Returns:
        Comparison DataFrame
    """
    # Create DataFrame
    df = pd.DataFrame(models_metrics).T
    
    # Plot comparison
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc', 'pr_auc']
    
    for idx, metric in enumerate(metrics_to_plot):
        if metric in df.columns:
            ax = axes[idx]
            df[metric].plot(kind='bar', ax=ax, color='steelblue')
            ax.set_title(metric.upper(), fontsize=12, fontweight='bold')
            ax.set_ylabel('Score')
            ax.set_ylim([0, 1])
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return df


def export_latent_representations(
    vae_model,
    X: np.ndarray,
    y: np.ndarray,
    output_path: str
):
    """
    Export latent representations to CSV
    
    Args:
        vae_model: Trained VAE model
        X: ECG windows
        y: Labels
        output_path: Path to save CSV
    """
    # Encode
    z = vae_model.encode(X)
    
    # Create DataFrame
    latent_cols = [f'latent_{i}' for i in range(z.shape[1])]
    df = pd.DataFrame(z, columns=latent_cols)
    df['label'] = y
    
    # Save
    df.to_csv(output_path, index=False)
    print(f"Latent representations exported to {output_path}")


def profile_inference_time(
    model,
    X_sample: np.ndarray,
    num_runs: int = 100
) -> Dict[str, float]:
    """
    Profile model inference time
    
    Args:
        model: Model to profile
        X_sample: Sample input
        num_runs: Number of runs for averaging
    
    Returns:
        Timing statistics
    """
    times = []
    
    # Warmup
    _ = model(X_sample)
    
    # Profile
    for _ in range(num_runs):
        start = time.time()
        _ = model(X_sample)
        end = time.time()
        times.append(end - start)
    
    times = np.array(times)
    
    stats = {
        'mean_ms': np.mean(times) * 1000,
        'std_ms': np.std(times) * 1000,
        'min_ms': np.min(times) * 1000,
        'max_ms': np.max(times) * 1000,
        'throughput_samples_per_sec': len(X_sample) / np.mean(times)
    }
    
    return stats


def timer(func):
    """
    Decorator to measure function execution time
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f"{func.__name__} took {end - start:.2f} seconds")
        return result
    return wrapper


def save_model_summary(model, save_path: str):
    """
    Save model architecture summary to text file
    
    Args:
        model: Keras model
        save_path: Path to save summary
    """
    with open(save_path, 'w') as f:
        model.summary(print_fn=lambda x: f.write(x + '\n'))
    
    print(f"Model summary saved to {save_path}")


def create_training_animation(
    history_path: str,
    output_path: str = "results/training_animation.gif"
):
    """
    Create animated GIF of training progress
    
    Args:
        history_path: Path to training history JSON
        output_path: Path to save GIF
    """
    import json
    from matplotlib.animation import FuncAnimation, PillowWriter
    
    # Load history
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    epochs = len(history['total_loss'])
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    def animate(frame):
        ax.clear()
        ax.plot(history['total_loss'][:frame+1], label='Train Loss', linewidth=2)
        ax.plot(history['val_total_loss'][:frame+1], label='Val Loss', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Total Loss')
        ax.set_title(f'Training Progress - Epoch {frame+1}/{epochs}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    anim = FuncAnimation(fig, animate, frames=epochs, interval=200)
    
    writer = PillowWriter(fps=5)
    anim.save(output_path, writer=writer)
    
    print(f"Training animation saved to {output_path}")
    plt.close()


def plot_beat_types_distribution(
    annotations: List[str],
    save_path: str = None
):
    """
    Plot distribution of ECG beat types
    
    Args:
        annotations: List of beat type annotations
        save_path: Path to save figure
    """
    from collections import Counter
    
    # Count beat types
    counts = Counter(annotations)
    
    # Sort by frequency
    sorted_counts = sorted(counts.items(), key=lambda x: x[1], reverse=True)
    labels, values = zip(*sorted_counts)
    
    # Plot
    plt.figure(figsize=(14, 6))
    bars = plt.bar(range(len(labels)), values, color='steelblue', edgecolor='black')
    plt.xticks(range(len(labels)), labels, fontsize=12)
    plt.xlabel('Beat Type', fontsize=14, fontweight='bold')
    plt.ylabel('Count', fontsize=14, fontweight='bold')
    plt.title('ECG Beat Types Distribution', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (bar, value) in enumerate(zip(bars, values)):
        plt.text(
            bar.get_x() + bar.get_width()/2,
            bar.get_height(),
            f'{value}',
            ha='center',
            va='bottom',
            fontsize=10
        )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def generate_synthetic_ecg(
    length: int = 1000,
    heart_rate: int = 75,
    sampling_rate: int = 360,
    noise_level: float = 0.05
) -> np.ndarray:
    """
    Generate synthetic ECG signal for testing
    
    Args:
        length: Signal length in samples
        heart_rate: Heart rate in BPM
        sampling_rate: Sampling rate in Hz
        noise_level: Gaussian noise standard deviation
    
    Returns:
        Synthetic ECG signal
    """
    t = np.arange(length) / sampling_rate
    
    # R-R interval
    rr_interval = 60.0 / heart_rate
    
    # Generate P, QRS, T waves
    ecg = np.zeros(length)
    
    num_beats = int(length / (rr_interval * sampling_rate))
    
    for i in range(num_beats):
        beat_start = int(i * rr_interval * sampling_rate)
        
        if beat_start + 100 < length:
            # P wave
            p_idx = beat_start + 20
            ecg[p_idx:p_idx+20] += 0.1 * np.sin(np.linspace(0, np.pi, 20))
            
            # QRS complex
            qrs_idx = beat_start + 50
            ecg[qrs_idx:qrs_idx+20] += 1.0 * np.sin(np.linspace(0, np.pi, 20))
            
            # T wave
            t_idx = beat_start + 80
            ecg[t_idx:t_idx+30] += 0.3 * np.sin(np.linspace(0, np.pi, 30))
    
    # Add noise
    ecg += np.random.normal(0, noise_level, length)
    
    return ecg


def calculate_heart_rate_variability(r_peaks: np.ndarray, sampling_rate: int = 360) -> Dict[str, float]:
    """
    Calculate HRV metrics from R-peaks
    
    Args:
        r_peaks: Array of R-peak indices
        sampling_rate: Sampling rate in Hz
    
    Returns:
        Dictionary of HRV metrics
    """
    # R-R intervals in milliseconds
    rr_intervals = np.diff(r_peaks) / sampling_rate * 1000
    
    hrv_metrics = {
        'mean_rr_ms': np.mean(rr_intervals),
        'std_rr_ms': np.std(rr_intervals),
        'rmssd_ms': np.sqrt(np.mean(np.diff(rr_intervals)**2)),
        'nn50': np.sum(np.abs(np.diff(rr_intervals)) > 50),
        'pnn50': np.sum(np.abs(np.diff(rr_intervals)) > 50) / len(rr_intervals) * 100,
        'mean_hr_bpm': 60000 / np.mean(rr_intervals),
        'std_hr_bpm': 60000 / np.mean(rr_intervals) * (np.std(rr_intervals) / np.mean(rr_intervals))
    }
    
    return hrv_metrics


if __name__ == "__main__":
    # Test utilities
    
    # Generate synthetic ECG
    print("Generating synthetic ECG...")
    ecg = generate_synthetic_ecg(length=2000, heart_rate=75)
    
    # Plot
    plot_ecg_signal(ecg, title="Synthetic ECG Signal")
    
    # Compute quality
    sqi = compute_signal_quality_index(ecg)
    print(f"Signal Quality Index: {sqi:.3f}")
    
    print("\nUtilities module ready!")
