# 📑 Project File Index

Complete reference guide to all files in the ECG Anomaly Detection project.

---

## 📋 Configuration Files

### `config.yaml` (Main Configuration)
**Purpose**: Central configuration for all hyperparameters and settings
**Key Sections**:
- Dataset settings (window length, sampling rate, beat annotations)
- Preprocessing (normalization, denoising, augmentation)
- Model architecture (encoder, decoder, latent dimensions)
- Training parameters (batch size, epochs, learning rate)
- Anomaly detection (threshold methods, ensemble settings)
- Evaluation (metrics, visualization options)
- File paths (models, results, logs)

**When to Edit**: To change hyperparameters, adjust thresholds, modify architecture

---

### `requirements.txt` (Dependencies)
**Purpose**: Lists all Python package dependencies
**Key Packages**:
- TensorFlow 2.14.0 (deep learning framework)
- WFDB 4.1.2 (ECG signal processing)
- Streamlit 1.26.0 (web dashboard)
- scikit-learn 1.3.0 (preprocessing and metrics)
- Matplotlib, Seaborn, Plotly (visualizations)

**When to Use**: During installation with `pip install -r requirements.txt`

---

## 🔬 Core Python Modules

### `data_loader.py` (WFDB Data Loading)
**Lines**: ~300
**Purpose**: Load and segment ECG signals from MIT-BIH database
**Key Classes**:
- `WFDBDataLoader`: Main data loading class

**Key Methods**:
- `get_record_list()`: Get list of available ECG records
- `load_record()`: Load single ECG record with annotations
- `segment_beats()`: Segment signal into fixed-length windows
- `quality_check()`: Filter low-quality segments
- `process_all_records()`: Process entire database
- `save_to_hdf5()`: Save processed data
- `load_from_hdf5()`: Load processed data

**Usage**: Run standalone to process MIT-BIH database
```bash
python data_loader.py
```

---

### `preprocessing.py` (Signal Preprocessing)
**Lines**: ~400
**Purpose**: Preprocess ECG signals and create train/val/test splits
**Key Classes**:
- `ECGPreprocessor`: Complete preprocessing pipeline

**Key Methods**:
- `denoise()`: Apply Savitzky-Golay filter
- `normalize()`: Normalize signals (Standard or MinMax)
- `augment_data()`: Apply data augmentation
- `split_data()`: Create train/val/test splits (semi-supervised)
- `process_pipeline()`: Run complete preprocessing
- `save_processed_data()`: Save to HDF5
- `load_processed_data()`: Load from HDF5

**Preprocessing Steps**:
1. Denoising (Savitzky-Golay filter)
2. Train/val/test split (70/15/15)
3. Normalization (fit on train only)
4. Data augmentation (Gaussian noise, time shift, amplitude scaling)
5. Reshape for model input (add channel dimension)

**Usage**: Run standalone to preprocess data
```bash
python preprocessing.py
```

---

### `vae_model.py` (VAE Architecture)
**Lines**: ~500
**Purpose**: Define Variational Autoencoder architecture
**Key Classes**:
- `Sampling`: Reparameterization trick layer
- `VAEEncoder`: Convolutional encoder
- `VAEDecoder`: Convolutional decoder
- `VAE`: Complete VAE model

**Architecture**:
```
Input (187, 1)
  ↓
Encoder (4× Conv1D + BN + ReLU + Dropout)
  ↓
Latent Space (16D): z_mean, z_log_var
  ↓
Sampling (reparameterization trick)
  ↓
Decoder (4× Conv1DTranspose + BN + ReLU + Dropout)
  ↓
Output (187, 1)
```

**Loss Function**: `total_loss = reconstruction_loss + beta * kl_loss`

**Key Methods**:
- `encode()`: Encode ECG to latent space
- `decode()`: Decode latent vector to ECG
- `reconstruct()`: Full encode-decode cycle
- `train_step()`: Custom training step
- `test_step()`: Custom evaluation step

**Usage**: Run standalone to test architecture
```bash
python vae_model.py
```

---

### `train.py` (Training Pipeline)
**Lines**: ~400
**Purpose**: Train VAE model with callbacks and monitoring
**Key Classes**:
- `LatentSpaceVisualizer`: Custom callback for latent space visualization
- `VAETrainer`: Complete training pipeline

**Training Features**:
- Early stopping (patience=20)
- Model checkpointing (saves best model)
- Learning rate scheduling
- TensorBoard logging
- Latent space visualization every 10 epochs
- Training curve plotting
- Reconstruction visualization

**Key Methods**:
- `load_data()`: Load preprocessed data
- `build_model()`: Build and compile VAE
- `get_callbacks()`: Create training callbacks
- `train()`: Execute training loop
- `visualize_reconstructions()`: Plot original vs reconstructed ECG

**Usage**: Run standalone to train model
```bash
python train.py
```

**Expected Time**: 30-60 minutes (GPU), 2-4 hours (CPU)

---

### `anomaly_detection.py` (Multi-Strategy Detection)
**Lines**: ~450
**Purpose**: Detect anomalies using multiple strategies
**Key Classes**:
- `AnomalyDetector`: Multi-strategy anomaly detection system

**Detection Methods**:
1. **Reconstruction Error**: MSE between original and reconstructed
2. **Latent Distance**: Mahalanobis distance from normal distribution
3. **KL Divergence**: Distribution-based anomaly scoring
4. **Ensemble**: Majority voting across methods

**Key Methods**:
- `compute_reconstruction_error()`: Calculate MSE
- `compute_latent_statistics()`: Fit normal distribution
- `compute_latent_distance()`: Mahalanobis distance
- `compute_kl_divergence()`: KL divergence per sample
- `fit_thresholds()`: Tune thresholds on validation set
- `predict()`: Classify as normal or anomaly
- `save_predictions()`: Export results to CSV

**Threshold Methods**:
- **Statistical**: `threshold = mean + k * std` (default k=3)
- **Percentile**: Use Nth percentile (default 95%)
- **Fixed**: User-defined threshold

**Usage**: Run standalone to detect anomalies
```bash
python anomaly_detection.py
```

---

### `evaluate.py` (Comprehensive Evaluation)
**Lines**: ~500
**Purpose**: Evaluate model performance with metrics and visualizations
**Key Classes**:
- `ModelEvaluator`: Complete evaluation system

**Metrics Computed**:
- Accuracy, Precision, Recall, F1-Score
- ROC-AUC, PR-AUC
- Specificity, False Positive Rate, False Negative Rate

**Visualizations Generated**:
- Confusion matrix
- ROC curve
- Precision-Recall curve
- Reconstruction error distribution
- Latent space (t-SNE, PCA, UMAP)

**Key Methods**:
- `compute_metrics()`: Calculate all classification metrics
- `plot_confusion_matrix()`: Visualize confusion matrix
- `plot_roc_curve()`: Plot ROC curve
- `plot_precision_recall_curve()`: Plot PR curve
- `plot_error_distribution()`: Histogram of errors
- `plot_latent_space()`: Dimensionality reduction visualization
- `generate_report()`: Create detailed text report
- `evaluate_full()`: Run complete evaluation

**Usage**: Run standalone for full evaluation
```bash
python evaluate.py
```

---

### `utils.py` (Utility Functions)
**Lines**: ~300
**Purpose**: Helper functions for visualization and analysis
**Key Functions**:
- `plot_ecg_signal()`: Plot ECG with annotations
- `compute_signal_quality_index()`: Calculate SQI
- `compare_models_performance()`: Compare multiple models
- `export_latent_representations()`: Export latent vectors to CSV
- `profile_inference_time()`: Measure inference speed
- `generate_synthetic_ecg()`: Create synthetic ECG for testing
- `calculate_heart_rate_variability()`: Compute HRV metrics
- `timer`: Decorator for timing functions

**Usage**: Import utilities or run standalone for tests
```bash
python utils.py
```

---

## 🖥️ User Interfaces

### `streamlit_app.py` (Interactive Dashboard)
**Lines**: ~600
**Purpose**: Web-based interface for anomaly detection
**Pages**:
1. **Upload & Predict**: Upload ECG files and detect anomalies
2. **Model Evaluation**: View comprehensive performance metrics
3. **About**: Technical documentation and model info

**Features**:
- File upload (WFDB, CSV, or test dataset)
- Real-time anomaly detection
- Adjustable threshold slider
- ECG comparison plots (original vs reconstructed)
- Error distribution histogram
- Sample visualizations
- Downloadable predictions (CSV)
- Performance metrics dashboard

**Usage**: Launch dashboard
```bash
streamlit run streamlit_app.py
```
**Access**: http://localhost:8501

---

### `run_pipeline.py` (Automated Pipeline)
**Lines**: ~250
**Purpose**: Execute complete pipeline automatically
**Pipeline Steps**:
1. Data loading and preprocessing
2. Model training
3. Anomaly detection
4. Comprehensive evaluation

**Command-line Arguments**:
- `--skip-preprocessing`: Skip data preprocessing
- `--skip-training`: Skip model training
- `--config`: Specify custom config file

**Usage**:
```bash
# Full pipeline
python run_pipeline.py

# Skip preprocessing
python run_pipeline.py --skip-preprocessing

# Skip both preprocessing and training
python run_pipeline.py --skip-preprocessing --skip-training
```

---

### `demo.py` (Quick Demonstration)
**Lines**: ~250
**Purpose**: Quick demo without training
**Demo Sections**:
1. Project structure overview
2. Model architecture display
3. Anomaly detection concept explanation
4. Synthetic ECG generation
5. Dataset statistics (if available)

**Usage**: Run quick demo
```bash
python demo.py
```
**Output**: Demo plots saved to current directory

---

## 📚 Documentation Files

### `README.md` (Main Documentation)
**Sections**: 65+
**Purpose**: Comprehensive project documentation
**Contents**:
- Project overview and features
- Installation instructions
- Quick start guide
- Usage examples
- Configuration details
- Performance benchmarks
- Technical deep dive
- Troubleshooting
- Academic citations
- Contributing guidelines

**Length**: ~1,500 lines

---

### `QUICKSTART.md` (Quick Setup Guide)
**Purpose**: 5-minute setup guide
**Contents**:
- Step-by-step installation
- Quick pipeline execution
- Expected outputs
- Common issues
- Performance tips
- Verification checklist

**Target Audience**: First-time users wanting quick setup

---

### `INSTALLATION.md` (Detailed Installation)
**Purpose**: Comprehensive installation and troubleshooting
**Contents**:
- System requirements
- Detailed installation steps
- 10+ common issues with solutions
- GPU setup guide
- Performance optimization
- Verification tests
- Additional resources

**Target Audience**: Users encountering installation problems

---

### `PROJECT_SUMMARY.md` (Project Overview)
**Purpose**: High-level project summary for recruiters
**Contents**:
- Key highlights and achievements
- Complete file structure
- Usage options
- What makes it special
- Performance benchmarks
- How to present in interviews
- Learning outcomes
- Enhancement ideas

**Target Audience**: Recruiters, interviewers, portfolio viewers

---

### `LICENSE` (MIT License)
**Purpose**: Software license
**Type**: MIT License (permissive open source)
**Includes**: Dataset attribution (MIT-BIH)

---

### `.gitignore` (Git Ignore Rules)
**Purpose**: Specify files to exclude from version control
**Excludes**:
- Python cache files (`__pycache__`, `*.pyc`)
- Virtual environments (`venv/`, `env/`)
- Large data files (`.h5`, `.hdf5`)
- Trained models (`.h5` in `saved_models/`)
- Generated results (plots, logs)
- System files (`.DS_Store`, `Thumbs.db`)

---

## 📊 Data Files (Generated)

### `data/processed/ecg_data.h5`
**Generated by**: `data_loader.py`
**Contents**: Segmented ECG windows and labels
**Size**: ~500 MB
**Format**: HDF5

**Datasets**:
- `X`: ECG segments (num_samples, 187)
- `y`: Labels (num_samples,)

**Metadata**:
- window_length, sampling_rate
- total_beats, normal_beats, anomaly_beats

---

### `data/processed/preprocessed_data.h5`
**Generated by**: `preprocessing.py`
**Contents**: Train/val/test splits (preprocessed and augmented)
**Size**: ~800 MB
**Format**: HDF5

**Datasets**:
- `X_train`, `y_train`: Training set (normal only, augmented)
- `X_val`, `y_val`: Validation set (normal + anomalies)
- `X_test`, `y_test`: Test set (normal + anomalies)

---

## 💾 Model Files (Generated)

### `saved_models/vae_best_model.h5`
**Generated by**: `train.py` (ModelCheckpoint callback)
**Purpose**: Best model (lowest validation loss)
**Size**: ~25 MB
**Format**: Keras HDF5

**Use**: Load for inference and evaluation

---

### `saved_models/vae_final_model.h5`
**Generated by**: `train.py` (at end of training)
**Purpose**: Final model after all epochs
**Size**: ~25 MB
**Format**: Keras HDF5

---

## 📈 Results Files (Generated)

### `results/training_curves.png`
**Generated by**: `train.py`
**Contents**: 3 subplots (total loss, reconstruction loss, KL loss)

### `results/confusion_matrix.png`
**Generated by**: `evaluate.py`
**Contents**: 2×2 confusion matrix with counts and percentages

### `results/roc_curve.png`
**Generated by**: `evaluate.py`
**Contents**: ROC curve with AUC score

### `results/precision_recall_curve.png`
**Generated by**: `evaluate.py`
**Contents**: PR curve with AUC score

### `results/error_distribution.png`
**Generated by**: `evaluate.py`
**Contents**: Histogram of reconstruction errors (normal vs anomaly)

### `results/latent_space_tsne.png`
**Generated by**: `evaluate.py`
**Contents**: t-SNE visualization of latent space

### `results/evaluation_report.txt`
**Generated by**: `evaluate.py`
**Contents**: Detailed text report with all metrics

### `results/evaluation_report.json`
**Generated by**: `evaluate.py`
**Contents**: Metrics in JSON format

### `results/training_history.json`
**Generated by**: `train.py`
**Contents**: Training history (loss curves data)

### `results/predictions/test_predictions.csv`
**Generated by**: `anomaly_detection.py` or `run_pipeline.py`
**Contents**: Predictions on test set

**Columns**:
- sample_index
- true_label
- predicted_label
- reconstruction_error (and other scores if ensemble)

---

## 📝 Logs (Generated)

### `logs/vae_YYYYMMDD-HHMMSS/`
**Generated by**: `train.py` (TensorBoard callback)
**Contents**: TensorBoard training logs

**View with**:
```bash
tensorboard --logdir logs/
```

---

## 🗂️ Directory Structure Summary

```
ECG-VAE-Anomaly-Detection/
├── 📋 Config & Dependencies
│   ├── config.yaml
│   └── requirements.txt
│
├── 🔬 Core Modules (Python)
│   ├── data_loader.py
│   ├── preprocessing.py
│   ├── vae_model.py
│   ├── train.py
│   ├── anomaly_detection.py
│   ├── evaluate.py
│   └── utils.py
│
├── 🖥️ User Interfaces
│   ├── streamlit_app.py
│   ├── run_pipeline.py
│   └── demo.py
│
├── 📚 Documentation
│   ├── README.md (1,500 lines)
│   ├── QUICKSTART.md
│   ├── INSTALLATION.md
│   ├── PROJECT_SUMMARY.md
│   ├── FILE_INDEX.md (this file)
│   ├── LICENSE
│   └── .gitignore
│
├── 📊 Data (Input & Generated)
│   ├── mit-bih-arrhythmia-database-1.0.0/ (48 records)
│   └── data/processed/ (HDF5 files)
│
├── 💾 Models (Generated)
│   └── saved_models/ (.h5 files)
│
├── 📈 Results (Generated)
│   ├── results/ (plots, reports)
│   └── results/predictions/ (CSV files)
│
└── 📝 Logs (Generated)
    └── logs/ (TensorBoard logs)
```

---

## 🔍 Quick Reference

### To Process Data:
```bash
python data_loader.py        # Load WFDB files
python preprocessing.py      # Preprocess and split
```

### To Train Model:
```bash
python train.py              # Train VAE
```

### To Evaluate:
```bash
python evaluate.py           # Full evaluation
```

### To Run Pipeline:
```bash
python run_pipeline.py       # Automated pipeline
python demo.py              # Quick demo
```

### To Launch Dashboard:
```bash
streamlit run streamlit_app.py
```

---

**Total Project Size**: 4,000+ lines of code across 10 core Python modules + 6 documentation files

**Key Achievement**: Complete, production-ready ECG anomaly detection system with end-to-end pipeline!
