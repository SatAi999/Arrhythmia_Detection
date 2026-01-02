"""
Demo Script - Quick Demonstration of ECG Anomaly Detection

This script provides a quick demo of the system's capabilities without
requiring full training. Uses synthetic data for demonstration.
"""

import numpy as np
import matplotlib.pyplot as plt
from utils import generate_synthetic_ecg, plot_ecg_signal


def demo_synthetic_ecg():
    """Demonstrate synthetic ECG generation"""
    print("\n" + "="*70)
    print("DEMO 1: Synthetic ECG Generation")
    print("="*70 + "\n")
    
    # Generate normal ECG
    normal_ecg = generate_synthetic_ecg(
        length=2000,
        heart_rate=75,
        sampling_rate=360,
        noise_level=0.02
    )
    
    # Generate abnormal ECG (higher noise, irregular rhythm)
    abnormal_ecg = generate_synthetic_ecg(
        length=2000,
        heart_rate=120,  # Tachycardia
        sampling_rate=360,
        noise_level=0.1
    )
    
    # Plot comparison
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))
    
    time_axis = np.arange(len(normal_ecg)) / 360
    
    axes[0].plot(time_axis, normal_ecg, color='blue', linewidth=1)
    axes[0].set_title('Normal ECG (75 BPM)', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Time (seconds)')
    axes[0].set_ylabel('Amplitude')
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(time_axis, abnormal_ecg, color='red', linewidth=1)
    axes[1].set_title('Abnormal ECG (Tachycardia, 120 BPM)', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Time (seconds)')
    axes[1].set_ylabel('Amplitude')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('demo_ecg_comparison.png', dpi=300, bbox_inches='tight')
    print("✓ Saved comparison plot to 'demo_ecg_comparison.png'")
    plt.show()


def demo_data_stats():
    """Show dataset statistics"""
    print("\n" + "="*70)
    print("DEMO 2: Dataset Statistics")
    print("="*70 + "\n")
    
    try:
        from data_loader import WFDBDataLoader
        
        loader = WFDBDataLoader()
        
        # Try to load existing data
        try:
            X, y = loader.load_from_hdf5()
            loader.print_statistics()
            
            print("\nData Shape:")
            print(f"  X: {X.shape} (samples, window_length)")
            print(f"  y: {y.shape} (samples,)")
            
            print("\nClass Distribution:")
            normal_count = np.sum(y == 0)
            anomaly_count = np.sum(y == 1)
            print(f"  Normal: {normal_count:,} ({100*normal_count/len(y):.1f}%)")
            print(f"  Anomaly: {anomaly_count:,} ({100*anomaly_count/len(y):.1f}%)")
            
        except FileNotFoundError:
            print("⚠ Processed data not found.")
            print("  Run 'python preprocessing.py' to process the dataset first.")
    
    except ImportError as e:
        print(f"⚠ Error: {e}")
        print("  Ensure all dependencies are installed: pip install -r requirements.txt")


def demo_model_architecture():
    """Show model architecture"""
    print("\n" + "="*70)
    print("DEMO 3: VAE Model Architecture")
    print("="*70 + "\n")
    
    try:
        from vae_model import VAE
        import tensorflow as tf
        
        # Build model
        vae = VAE()
        
        # Create dummy input to build the model
        dummy_input = tf.random.normal((1, 187, 1))
        _ = vae(dummy_input)
        
        print("Encoder Architecture:")
        print("-" * 70)
        vae.encoder.summary()
        
        print("\n\nDecoder Architecture:")
        print("-" * 70)
        vae.decoder.summary()
        
        print("\n\nModel Statistics:")
        print("-" * 70)
        print(f"Total Parameters: {vae.count_params():,}")
        print(f"Latent Dimension: {vae.latent_dim}")
        print(f"Input Shape: (batch_size, 187, 1)")
        print(f"Output Shape: (batch_size, 187, 1)")
        
    except Exception as e:
        print(f"⚠ Error: {e}")
        print("  Ensure TensorFlow is installed: pip install tensorflow==2.14.0")


def demo_anomaly_detection_concept():
    """Demonstrate anomaly detection concept"""
    print("\n" + "="*70)
    print("DEMO 4: Anomaly Detection Concept")
    print("="*70 + "\n")
    
    print("How VAE Detects Anomalies:")
    print("-" * 70)
    print("""
    1. TRAINING PHASE (Normal ECG Only):
       - VAE learns to compress normal ECG into latent space
       - VAE learns to reconstruct normal ECG from latent space
       - Reconstruction error is LOW for normal ECG
    
    2. TESTING PHASE (Normal + Anomalous ECG):
       - Normal ECG: Reconstructed well → LOW error → Classified as NORMAL
       - Anomalous ECG: Reconstructed poorly → HIGH error → Classified as ANOMALY
    
    3. THRESHOLD:
       - threshold = mean(normal_errors) + k × std(normal_errors)
       - Default k = 3.0 (adjustable in Streamlit dashboard)
       - If reconstruction_error > threshold → ANOMALY
    
    4. ENSEMBLE DETECTION (Advanced):
       - Method 1: Reconstruction Error
       - Method 2: Latent Space Distance (Mahalanobis)
       - Method 3: KL Divergence
       - Final Decision: Majority voting
    """)
    
    # Visualize threshold concept
    np.random.seed(42)
    normal_errors = np.random.normal(0.02, 0.01, 1000)
    anomaly_errors = np.random.normal(0.08, 0.02, 200)
    
    threshold = np.mean(normal_errors) + 3 * np.std(normal_errors)
    
    plt.figure(figsize=(12, 6))
    
    plt.hist(normal_errors, bins=50, alpha=0.6, color='blue', label='Normal ECG', density=True)
    plt.hist(anomaly_errors, bins=50, alpha=0.6, color='red', label='Anomaly ECG', density=True)
    plt.axvline(threshold, color='green', linestyle='--', linewidth=2, 
                label=f'Threshold = {threshold:.4f}')
    
    plt.xlabel('Reconstruction Error', fontsize=12)
    plt.ylabel('Density', fontsize=12)
    plt.title('Anomaly Detection Using Reconstruction Error', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('demo_threshold_concept.png', dpi=300, bbox_inches='tight')
    print("✓ Saved threshold concept plot to 'demo_threshold_concept.png'")
    plt.show()


def demo_project_structure():
    """Show project structure"""
    print("\n" + "="*70)
    print("DEMO 5: Project Structure")
    print("="*70 + "\n")
    
    structure = """
    ECG-VAE-Anomaly-Detection/
    │
    ├── 📋 Configuration
    │   ├── config.yaml              # All hyperparameters and settings
    │   └── requirements.txt         # Python dependencies
    │
    ├── 🔬 Core Modules
    │   ├── data_loader.py          # WFDB data loading and segmentation
    │   ├── preprocessing.py        # Signal preprocessing and augmentation
    │   ├── vae_model.py           # VAE architecture (Encoder, Decoder)
    │   ├── train.py               # Training pipeline with callbacks
    │   ├── anomaly_detection.py   # Multi-strategy anomaly detection
    │   ├── evaluate.py            # Comprehensive evaluation
    │   └── utils.py               # Helper functions
    │
    ├── 🖥️ User Interfaces
    │   ├── streamlit_app.py       # Interactive web dashboard
    │   └── run_pipeline.py        # Automated pipeline execution
    │
    ├── 📊 Data
    │   ├── mit-bih-arrhythmia-database-1.0.0/  # Raw MIT-BIH data
    │   └── data/processed/        # Preprocessed train/val/test splits
    │
    ├── 💾 Outputs
    │   ├── saved_models/          # Trained VAE models
    │   ├── results/               # Metrics, plots, predictions
    │   └── logs/                  # TensorBoard training logs
    │
    └── 📚 Documentation
        ├── README.md              # Comprehensive documentation
        ├── QUICKSTART.md          # Quick start guide
        └── LICENSE                # MIT License
    """
    
    print(structure)


def main():
    """Run all demos"""
    print("\n" + "="*70)
    print("🫀 ECG ANOMALY DETECTION - SYSTEM DEMO")
    print("="*70)
    print("\nThis demo showcases the capabilities of the ECG anomaly detection system.")
    print("No training required - just a quick overview!\n")
    
    # Run demos
    demo_project_structure()
    demo_model_architecture()
    demo_anomaly_detection_concept()
    demo_synthetic_ecg()
    demo_data_stats()
    
    # Final message
    print("\n" + "="*70)
    print("✅ DEMO COMPLETE!")
    print("="*70)
    print("\nNext Steps:")
    print("  1. Check generated demo plots in current directory")
    print("  2. Read QUICKSTART.md for setup instructions")
    print("  3. Run full pipeline: python run_pipeline.py")
    print("  4. Launch dashboard: streamlit run streamlit_app.py")
    print("\n" + "="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nDemo interrupted by user.")
    except Exception as e:
        print(f"\n\nError during demo: {str(e)}")
        import traceback
        traceback.print_exc()
