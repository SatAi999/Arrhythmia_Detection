# 🔧 Installation & Troubleshooting Guide

## 📋 System Requirements

### Minimum Requirements
- **OS**: Windows 10/11, macOS 10.14+, or Linux (Ubuntu 18.04+)
- **Python**: 3.8 or higher
- **RAM**: 8 GB (16 GB recommended)
- **Storage**: 5 GB free space
- **CPU**: Multi-core processor (GPU recommended but optional)

### Recommended Requirements
- **GPU**: NVIDIA GPU with CUDA support (for faster training)
- **CUDA**: 11.2 or higher (if using GPU)
- **cuDNN**: 8.1 or higher (if using GPU)

---

## 🚀 Installation Steps

### Step 1: Verify Python Installation

```bash
# Check Python version (should be 3.8+)
python --version

# If not installed, download from: https://www.python.org/downloads/
```

### Step 2: Create Virtual Environment (Recommended)

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Upgrade pip first
pip install --upgrade pip

# Install all dependencies
pip install -r requirements.txt
```

**Note**: This may take 5-10 minutes depending on your internet speed.

### Step 4: Verify Installation

```bash
# Test Python imports
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
python -c "import wfdb; print('WFDB installed successfully')"
python -c "import streamlit as st; print('Streamlit version:', st.__version__)"
```

### Step 5: Check GPU Availability (Optional)

```bash
python -c "import tensorflow as tf; print('GPUs Available:', tf.config.list_physical_devices('GPU'))"
```

---

## 🐛 Common Issues & Solutions

### Issue 1: TensorFlow Installation Failed

**Error:**
```
ERROR: Could not find a version that satisfies the requirement tensorflow==2.14.0
```

**Solution:**
```bash
# Try installing without version constraint
pip install tensorflow

# Or use CPU-only version
pip install tensorflow-cpu

# For macOS with M1/M2 chips
pip install tensorflow-macos tensorflow-metal
```

---

### Issue 2: WFDB Import Error

**Error:**
```
ImportError: No module named 'wfdb'
```

**Solution:**
```bash
# Install WFDB separately
pip install wfdb

# If still fails, try upgrading
pip install --upgrade wfdb
```

---

### Issue 3: NumPy/SciPy Version Conflicts

**Error:**
```
ERROR: Cannot install numpy==1.24.3 because these package versions have conflicting dependencies.
```

**Solution:**
```bash
# Let pip resolve dependencies automatically
pip install --upgrade numpy scipy scikit-learn

# Then install other packages
pip install -r requirements.txt
```

---

### Issue 4: Streamlit Not Starting

**Error:**
```
streamlit: command not found
```

**Solution:**
```bash
# Reinstall Streamlit
pip install --upgrade streamlit

# Or run directly with Python
python -m streamlit run streamlit_app.py

# On Windows, you may need to add Scripts folder to PATH
```

---

### Issue 5: GPU Not Detected

**Error:**
```
GPUs Available: []
```

**Solution:**

1. **Check NVIDIA Driver:**
```bash
nvidia-smi
```

2. **Install CUDA Toolkit:**
   - Download from: https://developer.nvidia.com/cuda-downloads
   - Install cuDNN: https://developer.nvidia.com/cudnn

3. **Install GPU-enabled TensorFlow:**
```bash
pip uninstall tensorflow
pip install tensorflow-gpu==2.14.0
```

4. **Verify GPU:**
```bash
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```

---

### Issue 6: Out of Memory During Training

**Error:**
```
ResourceExhaustedError: OOM when allocating tensor
```

**Solution:**

**Option A: Reduce Batch Size**
Edit `config.yaml`:
```yaml
training:
  batch_size: 128  # Reduce from 256
```

**Option B: Enable Memory Growth (GPU)**
Add to `train.py`:
```python
import tensorflow as tf
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
```

**Option C: Use CPU**
```bash
export CUDA_VISIBLE_DEVICES=""  # Linux/Mac
set CUDA_VISIBLE_DEVICES=  # Windows
```

---

### Issue 7: WFDB Files Not Found

**Error:**
```
FileNotFoundError: mit-bih-arrhythmia-database-1.0.0/100.dat not found
```

**Solution:**

1. **Verify Directory Structure:**
```bash
ls mit-bih-arrhythmia-database-1.0.0/
```

2. **Check File Extensions:**
   - Ensure files have `.dat`, `.hea`, `.atr` extensions
   - Not `.dat.txt` or other variations

3. **Update Path in config.yaml:**
```yaml
dataset:
  raw_data_dir: "path/to/your/mit-bih-arrhythmia-database-1.0.0"
```

---

### Issue 8: ModuleNotFoundError

**Error:**
```
ModuleNotFoundError: No module named 'preprocessing'
```

**Solution:**

1. **Ensure you're in the correct directory:**
```bash
pwd  # Should show project root
ls   # Should show data_loader.py, preprocessing.py, etc.
```

2. **Add project root to PYTHONPATH:**
```bash
# Linux/Mac
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Windows
set PYTHONPATH=%PYTHONPATH%;%CD%
```

---

### Issue 9: Permission Denied

**Error:**
```
PermissionError: [Errno 13] Permission denied: 'saved_models/vae_best_model.h5'
```

**Solution:**

**Option A: Run as Administrator (Windows)**
```bash
# Right-click Command Prompt → "Run as administrator"
```

**Option B: Change Permissions (Linux/Mac)**
```bash
chmod -R 755 saved_models/
chmod -R 755 data/
chmod -R 755 results/
```

---

### Issue 10: Streamlit Port Already in Use

**Error:**
```
OSError: [Errno 48] Address already in use
```

**Solution:**
```bash
# Use different port
streamlit run streamlit_app.py --server.port 8502

# Or kill process on port 8501
# Windows
netstat -ano | findstr :8501
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:8501 | xargs kill -9
```

---

## 🔍 Verification Checklist

Before running the pipeline, verify:

- [ ] Python 3.8+ installed
- [ ] Virtual environment activated (optional but recommended)
- [ ] All dependencies installed (`pip list`)
- [ ] TensorFlow imports successfully
- [ ] WFDB imports successfully
- [ ] Streamlit imports successfully
- [ ] MIT-BIH database files accessible
- [ ] Directory structure correct
- [ ] Sufficient disk space (5+ GB)
- [ ] Sufficient RAM (8+ GB)

---

## 📊 Performance Optimization

### For Faster Training:

1. **Use GPU**: Install CUDA and GPU-enabled TensorFlow
2. **Increase Batch Size**: If you have enough memory
   ```yaml
   training:
     batch_size: 512  # Default is 256
   ```
3. **Use Mixed Precision**: Add to `train.py`
   ```python
   from tensorflow.keras import mixed_precision
   policy = mixed_precision.Policy('mixed_float16')
   mixed_precision.set_global_policy(policy)
   ```
4. **Reduce Epochs**: For quick testing
   ```yaml
   training:
     epochs: 50  # Default is 200
   ```

### For Lower Memory Usage:

1. **Reduce Batch Size**:
   ```yaml
   training:
     batch_size: 64
   ```
2. **Reduce Model Size**:
   ```yaml
   model:
     encoder:
       conv_filters: [16, 32, 64, 128]  # Half the size
   ```
3. **Disable Augmentation**:
   ```yaml
   preprocessing:
     augmentation:
       enabled: False
   ```

---

## 🧪 Testing Your Setup

Run these tests to ensure everything works:

### Test 1: Demo Script
```bash
python demo.py
```
**Expected**: Should show project structure and generate demo plots

### Test 2: Data Loader
```bash
python data_loader.py
```
**Expected**: Should process MIT-BIH files (may take 5-10 minutes)

### Test 3: Model Architecture
```bash
python vae_model.py
```
**Expected**: Should display encoder and decoder summaries

### Test 4: Utilities
```bash
python utils.py
```
**Expected**: Should generate and plot synthetic ECG

---

## 📚 Additional Resources

### Python Package Documentation:
- TensorFlow: https://www.tensorflow.org/api_docs
- WFDB: https://wfdb.readthedocs.io/
- Streamlit: https://docs.streamlit.io/
- NumPy: https://numpy.org/doc/
- Pandas: https://pandas.pydata.org/docs/

### MIT-BIH Database:
- Homepage: https://physionet.org/content/mitdb/1.0.0/
- Documentation: https://archive.physionet.org/physiobank/database/html/mitdbdir/mitdbdir.htm

### Deep Learning Resources:
- VAE Tutorial: https://www.tensorflow.org/tutorials/generative/cvae
- Anomaly Detection: https://www.tensorflow.org/tutorials/generative/autoencoder

---

## 🆘 Getting Help

If you encounter issues not covered here:

1. **Check Error Message**: Read the full error traceback
2. **Google the Error**: Often solutions exist on StackOverflow
3. **Check Dependencies**: Run `pip list` and compare versions
4. **Update Packages**: Try `pip install --upgrade <package>`
5. **Restart**: Sometimes restarting Python/terminal helps
6. **Clean Install**: Delete `venv/` and start fresh

---

## 📞 Support Channels

For additional help:
- **GitHub Issues**: Open an issue in the repository
- **Stack Overflow**: Tag with `tensorflow`, `python`, `ecg`
- **TensorFlow Forum**: https://discuss.tensorflow.org/

---

## ✅ Success Indicators

You're ready to proceed when:

✅ All imports work without errors
✅ `python demo.py` runs successfully
✅ Streamlit launches without errors
✅ GPU detected (if applicable)
✅ MIT-BIH files accessible
✅ No permission errors when creating directories

---

**Happy Coding! You've got this! 🚀**
