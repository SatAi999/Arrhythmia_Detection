# Quick Start Guide

## 🚀 Getting Started in 5 Minutes

### Prerequisites
- Python 3.8 or higher
- pip package manager
- 4GB+ RAM recommended
- GPU (optional, but recommended for faster training)

---

## Step-by-Step Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

This installs all required packages including TensorFlow, WFDB, Streamlit, and visualization libraries.

---

### 2. Quick Data Check

Verify that the MIT-BIH database is present:

```bash
# Check if data directory exists
ls mit-bih-arrhythmia-database-1.0.0/
```

You should see files like `100.dat`, `100.hea`, `100.atr`, etc.

---

### 3. Run Complete Pipeline (Automated)

**Option A: Full Pipeline** (Recommended for first run)
```bash
python run_pipeline.py
```

This runs:
1. Data loading and preprocessing (~5-10 minutes)
2. Model training (~30-60 minutes on GPU, 2-4 hours on CPU)
3. Anomaly detection (~2 minutes)
4. Comprehensive evaluation (~5 minutes)

**Option B: Skip Steps** (If you've run before)
```bash
# Skip preprocessing (use existing processed data)
python run_pipeline.py --skip-preprocessing

# Skip both preprocessing and training (use existing model)
python run_pipeline.py --skip-preprocessing --skip-training
```

---

### 4. Manual Step-by-Step Execution

If you prefer manual control:

#### Step 1: Data Preprocessing
```bash
python preprocessing.py
```
Output: Creates `data/processed/` with segmented ECG windows

#### Step 2: Train Model
```bash
python train.py
```
Output: Creates `saved_models/vae_best_model.h5` and training visualizations

#### Step 3: Evaluate Model
```bash
python evaluate.py
```
Output: Creates evaluation metrics and visualizations in `results/`

---

### 5. Launch Interactive Dashboard

```bash
streamlit run streamlit_app.py
```

Then open your browser at: **http://localhost:8501**

The dashboard allows you to:
- Upload new ECG files for prediction
- Visualize anomaly detection results
- Explore model performance metrics
- Download predictions as CSV

---

## 📊 Expected Outputs

After running the pipeline, you'll have:

### Trained Models
- `saved_models/vae_best_model.h5` - Best model (lowest validation loss)
- `saved_models/vae_final_model.h5` - Final model after all epochs

### Processed Data
- `data/processed/ecg_data.h5` - Segmented ECG windows
- `data/processed/preprocessed_data.h5` - Train/val/test splits

### Results
- `results/training_curves.png` - Loss curves during training
- `results/confusion_matrix.png` - Model performance visualization
- `results/roc_curve.png` - ROC curve
- `results/precision_recall_curve.png` - PR curve
- `results/error_distribution.png` - Reconstruction error histogram
- `results/latent_space_tsne.png` - Latent space visualization
- `results/evaluation_report.txt` - Detailed metrics report
- `results/predictions/test_predictions.csv` - Predictions on test set

### Logs
- `logs/vae_YYYYMMDD-HHMMSS/` - TensorBoard logs

---

## 🔍 Viewing Results

### TensorBoard
```bash
tensorboard --logdir logs/
```
Open browser at: http://localhost:6006

### Check Metrics
```bash
cat results/evaluation_report.txt
```

### View Training History
```python
import json
with open('results/training_history.json', 'r') as f:
    history = json.load(f)
print(history.keys())
```

---

## 🎯 Testing Individual Components

### Test Data Loader
```bash
python data_loader.py
```

### Test VAE Model
```bash
python vae_model.py
```

### Test Utilities
```bash
python utils.py
```

---

## 🐛 Common Issues & Solutions

### Issue: Import errors
```bash
# Solution: Ensure you're in the correct directory
cd ECG-VAE-Anomaly-Detection
python run_pipeline.py
```

### Issue: GPU not detected
```bash
# Check if TensorFlow sees GPU
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"

# If empty, install GPU version
pip install tensorflow-gpu==2.14.0
```

### Issue: Out of memory during training
```bash
# Edit config.yaml and reduce batch size
# Change: batch_size: 256  →  batch_size: 128
```

### Issue: Streamlit port already in use
```bash
# Use different port
streamlit run streamlit_app.py --server.port 8502
```

---

## ⚡ Performance Tips

### Speed Up Training
1. **Use GPU**: Install CUDA-enabled TensorFlow
2. **Reduce Epochs**: Edit `config.yaml`, set `epochs: 100`
3. **Use Smaller Dataset**: Process fewer records in `data_loader.py`

### Reduce Memory Usage
1. **Lower Batch Size**: `batch_size: 128` in `config.yaml`
2. **Disable Augmentation**: Set `augmentation.enabled: False`
3. **Process in Batches**: Modify evaluation to use smaller batches

---

## 📚 Next Steps

After completing the quick start:

1. **Explore the Dashboard**
   - Upload your own ECG files
   - Adjust detection thresholds
   - Download predictions

2. **Experiment with Hyperparameters**
   - Edit `config.yaml`
   - Try different latent dimensions
   - Tune detection thresholds

3. **Advanced Features**
   - Implement custom anomaly detection methods
   - Add new visualization techniques
   - Integrate with real-time data streams

4. **Deploy**
   - Containerize with Docker
   - Deploy Streamlit to cloud
   - Create REST API for predictions

---

## 💡 Tips for Best Results

1. **Quality Data**: Ensure MIT-BIH files are complete and uncorrupted
2. **Sufficient Training**: Let the model train for at least 100 epochs
3. **Threshold Tuning**: Use the Streamlit dashboard to find optimal threshold
4. **Ensemble Methods**: Enable ensemble detection for better accuracy
5. **Monitor Training**: Use TensorBoard to watch training progress

---

## 🎓 Learning Resources

- **VAE Theory**: Read the docstrings in `vae_model.py`
- **ECG Basics**: Check MIT-BIH database documentation
- **TensorFlow**: Official TensorFlow tutorials
- **Streamlit**: Streamlit documentation and examples

---

## ✅ Checklist

Before considering setup complete, verify:

- [ ] All dependencies installed without errors
- [ ] MIT-BIH database accessible
- [ ] Data preprocessing completed successfully
- [ ] Model training finished (validation loss converged)
- [ ] Evaluation metrics generated
- [ ] Streamlit dashboard launches without errors
- [ ] Can upload and predict on test ECG files

---

**Ready to impress recruiters? You now have a world-class ECG anomaly detection system! 🚀**

For detailed documentation, see [README.md](README.md)
