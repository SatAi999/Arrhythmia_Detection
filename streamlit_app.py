"""
Advanced Streamlit Dashboard for ECG Anomaly Detection

Features:
- Interactive file upload for new ECG recordings
- Real-time anomaly detection with adjustable threshold
- Visual comparison: Original vs Reconstructed ECG
- Reconstruction error histogram and statistics
- Latent space visualization
- Model evaluation metrics display
- Batch processing for multiple files
- Download predictions as CSV
- Beautiful UI with custom styling
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import wfdb
import io
import os
import yaml
from tensorflow import keras
from typing import Tuple, Dict

# Import custom modules
from preprocessing import ECGPreprocessor
from anomaly_detection import AnomalyDetector
from evaluate import ModelEvaluator


# Page configuration
st.set_page_config(
    page_title="ECG Anomaly Detection - VAE System",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #FF6B6B;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    h1 {
        color: #FF4B4B;
    }
    h2 {
        color: #0E1117;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model_and_config():
    """Load trained VAE model and configuration"""
    from vae_model import VAE
    
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    model_path = os.path.join(config['paths']['models_dir'], 'vae_best_model.h5')
    
    if not os.path.exists(model_path):
        st.error(f"Model not found at {model_path}. Please train the model first.")
        st.stop()
    
    # Build the model with correct architecture
    vae = VAE("config.yaml")
    
    # Build model by calling it with dummy data
    dummy_input = np.zeros((1, config['dataset']['window_length'], 1), dtype=np.float32)
    _ = vae(dummy_input)
    
    # Load weights
    vae.load_weights(model_path)
    
    return vae, config


@st.cache_data
def load_preprocessed_data():
    """Load preprocessed data for evaluation"""
    try:
        preprocessor = ECGPreprocessor()
        data = preprocessor.load_processed_data()
        return data
    except FileNotFoundError:
        return None


def preprocess_uploaded_ecg(ecg_data: np.ndarray, _preprocessor: ECGPreprocessor) -> np.ndarray:
    """
    Preprocess uploaded ECG data
    
    Args:
        ecg_data: Raw ECG signal
        _preprocessor: ECGPreprocessor instance
    
    Returns:
        Preprocessed ECG windows
    """
    # Denoise
    if _preprocessor.denoising:
        from scipy.signal import savgol_filter
        ecg_data = savgol_filter(
            ecg_data,
            window_length=_preprocessor.savgol_window,
            polyorder=_preprocessor.savgol_polyorder
        )
    
    # Segment into windows
    window_length = _preprocessor.dataset_config['window_length']
    segments = []
    
    for i in range(0, len(ecg_data) - window_length, window_length // 2):
        segment = ecg_data[i:i + window_length]
        if len(segment) == window_length:
            segments.append(segment)
    
    if len(segments) == 0:
        return None
    
    X = np.array(segments)
    
    # Normalize
    X = _preprocessor.normalize(X, fit=False)
    
    # Reshape for model
    X = _preprocessor.prepare_for_model(X)
    
    return X


def plot_ecg_comparison(original: np.ndarray, reconstructed: np.ndarray, title: str = "ECG Comparison"):
    """Plot original vs reconstructed ECG using Plotly"""
    import tensorflow as tf
    
    # Convert TensorFlow tensors to numpy if needed
    if isinstance(original, tf.Tensor):
        original = original.numpy()
    if isinstance(reconstructed, tf.Tensor):
        reconstructed = reconstructed.numpy()
    
    fig = go.Figure()
    
    # Original
    fig.add_trace(go.Scatter(
        y=original.flatten(),
        mode='lines',
        name='Original',
        line=dict(color='blue', width=2)
    ))
    
    # Reconstructed
    fig.add_trace(go.Scatter(
        y=reconstructed.flatten(),
        mode='lines',
        name='Reconstructed',
        line=dict(color='red', width=2, dash='dash')
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title='Time (samples)',
        yaxis_title='Amplitude',
        hovermode='x unified',
        height=400,
        template='plotly_white'
    )
    
    return fig


def plot_error_histogram(errors: np.ndarray, threshold: float):
    """Plot reconstruction error histogram using Plotly"""
    fig = go.Figure()
    
    fig.add_trace(go.Histogram(
        x=errors,
        nbinsx=50,
        name='Reconstruction Error',
        marker_color='steelblue'
    ))
    
    # Add threshold line
    fig.add_vline(
        x=threshold,
        line_dash="dash",
        line_color="red",
        line_width=3,
        annotation_text=f"Threshold: {threshold:.4f}",
        annotation_position="top right"
    )
    
    fig.update_layout(
        title='Reconstruction Error Distribution',
        xaxis_title='Reconstruction Error',
        yaxis_title='Count',
        showlegend=True,
        height=400,
        template='plotly_white'
    )
    
    return fig


def main():
    """Main Streamlit application"""
    
    # Header
    st.title("🫀 ECG Anomaly Detection System")
    st.markdown("### Powered by Variational Autoencoder (VAE)")
    st.markdown("---")
    
    # Load model and config
    with st.spinner("Loading model..."):
        vae, config = load_model_and_config()
        preprocessor = ECGPreprocessor()
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    
    # Navigation
    page = st.sidebar.radio(
        "Navigation",
        ["📁 Upload & Predict", "📊 Model Evaluation", "ℹ️ About"]
    )
    
    # Threshold slider
    st.sidebar.markdown("---")
    st.sidebar.subheader("Detection Threshold")
    threshold_k = st.sidebar.slider(
        "Threshold Multiplier (k)",
        min_value=1.0,
        max_value=5.0,
        value=config['anomaly_detection']['threshold_k'],
        step=0.1,
        help="Threshold = mean + k × std"
    )
    
    # ==================== PAGE 1: Upload & Predict ====================
    if page == "📁 Upload & Predict":
        st.header("Upload ECG Data for Anomaly Detection")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("Upload Options")
            
            upload_option = st.radio(
                "Choose input format:",
                ["WFDB Record (.dat + .hea)", "CSV File", "Use Test Dataset"]
            )
            
            X_data = None
            labels = None
            
            if upload_option == "WFDB Record (.dat + .hea)":
                st.info("Upload both .dat and .hea files from MIT-BIH database")
                
                dat_file = st.file_uploader("Upload .dat file", type=['dat'])
                hea_file = st.file_uploader("Upload .hea file", type=['hea'])
                
                if dat_file and hea_file:
                    # Save temporary files
                    with open("temp_record.dat", "wb") as f:
                        f.write(dat_file.read())
                    with open("temp_record.hea", "wb") as f:
                        f.write(hea_file.read())
                    
                    try:
                        # Read WFDB record
                        record = wfdb.rdrecord("temp_record")
                        ecg_signal = record.p_signal[:, 0]  # First channel
                        
                        st.success(f"Loaded {len(ecg_signal)} samples")
                        
                        # Preprocess
                        X_data = preprocess_uploaded_ecg(ecg_signal, preprocessor)
                        
                        if X_data is not None:
                            st.success(f"Segmented into {len(X_data)} windows")
                        
                        # Cleanup
                        os.remove("temp_record.dat")
                        os.remove("temp_record.hea")
                        
                    except Exception as e:
                        st.error(f"Error reading WFDB files: {str(e)}")
            
            elif upload_option == "CSV File":
                st.info("Upload CSV file with ECG signal (single column)")
                
                csv_file = st.file_uploader("Upload CSV file", type=['csv'])
                
                if csv_file:
                    try:
                        df = pd.read_csv(csv_file)
                        
                        # Get first column
                        ecg_signal = df.iloc[:, 0].values
                        
                        st.success(f"Loaded {len(ecg_signal)} samples")
                        
                        # Preprocess
                        X_data = preprocess_uploaded_ecg(ecg_signal, preprocessor)
                        
                        if X_data is not None:
                            st.success(f"Segmented into {len(X_data)} windows")
                    
                    except Exception as e:
                        st.error(f"Error reading CSV: {str(e)}")
            
            else:  # Use Test Dataset
                st.info("Using preprocessed test dataset")
                
                data = load_preprocessed_data()
                
                if data is not None:
                    X_data = data['X_test']
                    labels = data['y_test']
                    st.success(f"Loaded {len(X_data)} test samples")
                else:
                    st.error("Test dataset not found. Please run preprocessing first.")
        
        with col2:
            st.subheader("Detection Status")
            
            if X_data is not None:
                st.metric("Total Windows", len(X_data))
                
                # Run detection
                if st.button("🔍 Run Anomaly Detection", type="primary"):
                    with st.spinner("Running detection..."):
                        # Initialize detector
                        detector = AnomalyDetector(vae)
                        
                        # Load validation data to fit thresholds
                        data = load_preprocessed_data()
                        if data:
                            detector.fit_thresholds(data['X_val'], data['y_val'], k=threshold_k)
                        else:
                            st.warning("Validation data not available. Using default threshold.")
                            # Use mock thresholds
                            detector.thresholds['reconstruction_error'] = 0.05
                        
                        # Predict
                        predictions, scores_dict = detector.predict(X_data)
                        scores = scores_dict['reconstruction_error']
                        threshold = detector.thresholds['reconstruction_error']
                        
                        # Store in session state
                        st.session_state['predictions'] = predictions
                        st.session_state['scores'] = scores
                        st.session_state['threshold'] = threshold
                        st.session_state['X_data'] = X_data
                        st.session_state['labels'] = labels
                        
                        st.success("Detection complete!")
            else:
                st.warning("Please upload data first")
        
        # Display results
        if 'predictions' in st.session_state:
            st.markdown("---")
            st.header("Detection Results")
            
            predictions = st.session_state['predictions']
            scores = st.session_state['scores']
            threshold = st.session_state['threshold']
            X_data = st.session_state['X_data']
            labels = st.session_state['labels']
            
            # Summary metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Samples", len(predictions))
            
            with col2:
                normal_count = np.sum(predictions == 0)
                st.metric("Normal", normal_count, 
                         delta=f"{100*normal_count/len(predictions):.1f}%")
            
            with col3:
                anomaly_count = np.sum(predictions == 1)
                st.metric("Anomaly", anomaly_count,
                         delta=f"{100*anomaly_count/len(predictions):.1f}%",
                         delta_color="inverse")
            
            with col4:
                if labels is not None:
                    accuracy = np.mean(predictions == labels)
                    st.metric("Accuracy", f"{100*accuracy:.1f}%")
                else:
                    st.metric("Threshold", f"{threshold:.4f}")
            
            # Error histogram
            st.subheader("Reconstruction Error Distribution")
            fig_hist = plot_error_histogram(scores, threshold)
            st.plotly_chart(fig_hist, use_container_width=True)
            
            # Sample visualizations
            st.subheader("Sample ECG Comparisons")
            
            # Select samples to visualize
            num_samples = min(5, len(X_data))
            sample_indices = st.multiselect(
                "Select samples to visualize (by index):",
                options=list(range(len(X_data))),
                default=list(range(num_samples))
            )
            
            if sample_indices:
                for idx in sample_indices:
                    original = X_data[idx]
                    reconstructed = vae.reconstruct(original[np.newaxis, ...])[0]
                    
                    pred_label = "🔴 ANOMALY" if predictions[idx] == 1 else "🟢 NORMAL"
                    true_label = ""
                    if labels is not None:
                        true_label = f" (True: {'Anomaly' if labels[idx] == 1 else 'Normal'})"
                    
                    title = f"Sample {idx} - {pred_label}{true_label} | Error: {scores[idx]:.4f}"
                    
                    fig = plot_ecg_comparison(original, reconstructed, title)
                    st.plotly_chart(fig, use_container_width=True)
            
            # Download predictions
            st.subheader("Download Results")
            
            # Create DataFrame
            results_df = pd.DataFrame({
                'sample_index': np.arange(len(predictions)),
                'predicted_label': predictions,
                'reconstruction_error': scores,
                'is_anomaly': predictions == 1
            })
            
            if labels is not None:
                results_df['true_label'] = labels
                results_df['correct'] = predictions == labels
            
            # Convert to CSV
            csv_buffer = io.StringIO()
            results_df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            st.download_button(
                label="📥 Download Predictions (CSV)",
                data=csv_data,
                file_name="anomaly_predictions.csv",
                mime="text/csv"
            )
    
    # ==================== PAGE 2: Model Evaluation ====================
    elif page == "📊 Model Evaluation":
        st.header("Model Performance Evaluation")
        
        data = load_preprocessed_data()
        
        if data is None:
            st.error("Test data not available. Please run preprocessing first.")
            st.stop()
        
        X_test = data['X_test']
        y_test = data['y_test']
        
        if st.button("🚀 Run Full Evaluation", type="primary"):
            with st.spinner("Running comprehensive evaluation..."):
                # Initialize detector and evaluator
                detector = AnomalyDetector(vae)
                evaluator = ModelEvaluator()
                
                # Fit thresholds and predict
                detector.fit_thresholds(data['X_val'], data['y_val'], k=threshold_k)
                predictions, scores_dict = detector.predict(X_test)
                scores = scores_dict['reconstruction_error']
                threshold = detector.thresholds['reconstruction_error']
                
                # Compute metrics
                metrics = evaluator.compute_metrics(y_test, predictions, scores)
                
                # Display metrics
                st.subheader("Classification Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Accuracy", f"{metrics['accuracy']:.3f}")
                    st.metric("Precision", f"{metrics['precision']:.3f}")
                
                with col2:
                    st.metric("Recall", f"{metrics['recall']:.3f}")
                    st.metric("F1 Score", f"{metrics['f1_score']:.3f}")
                
                with col3:
                    st.metric("ROC-AUC", f"{metrics['roc_auc']:.3f}")
                    st.metric("PR-AUC", f"{metrics['pr_auc']:.3f}")
                
                with col4:
                    st.metric("Specificity", f"{metrics['specificity']:.3f}")
                    st.metric("FPR", f"{metrics['fpr']:.3f}")
                
                # Visualizations
                st.markdown("---")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Confusion Matrix")
                    evaluator.plot_confusion_matrix(y_test, predictions)
                    st.image("results/confusion_matrix.png")
                
                with col2:
                    st.subheader("ROC Curve")
                    evaluator.plot_roc_curve(y_test, scores)
                    st.image("results/roc_curve.png")
                
                col3, col4 = st.columns(2)
                
                with col3:
                    st.subheader("Precision-Recall Curve")
                    evaluator.plot_precision_recall_curve(y_test, scores)
                    st.image("results/precision_recall_curve.png")
                
                with col4:
                    st.subheader("Error Distribution")
                    evaluator.plot_error_distribution(y_test, scores, threshold)
                    st.image("results/error_distribution.png")
                
                # Latent space
                st.markdown("---")
                st.subheader("Latent Space Visualization")
                
                z_test = vae.encode(X_test[:2000])  # Limit for performance
                evaluator.plot_latent_space(z_test, y_test[:2000], method='tsne')
                st.image("results/latent_space_tsne.png")
                
                st.success("Evaluation complete!")
    
    # ==================== PAGE 3: About ====================
    else:
        st.header("About This System")
        
        st.markdown("""
        ### 🫀 ECG Anomaly Detection using Variational Autoencoder
        
        This advanced system uses deep learning to detect arrhythmias in ECG signals.
        
        #### 🔬 Technical Details
        
        **Model Architecture:**
        - **Encoder**: 4-layer Conv1D with BatchNorm and Dropout
        - **Latent Space**: 16-dimensional continuous representation
        - **Decoder**: 4-layer Conv1DTranspose for reconstruction
        - **Loss Function**: Reconstruction Loss + KL Divergence
        
        **Dataset:**
        - MIT-BIH Arrhythmia Database from PhysioNet
        - 48 records, ~110,000 annotated beats
        - Normal vs. 9 types of arrhythmias
        
        **Detection Methods:**
        1. **Reconstruction Error**: MSE between original and reconstructed ECG
        2. **Latent Distance**: Mahalanobis distance in latent space
        3. **KL Divergence**: Distribution divergence from normal
        4. **Ensemble**: Voting across multiple methods
        
        #### 📊 Performance
        
        The model is trained on normal ECG beats only (semi-supervised learning)
        and learns to reconstruct them with minimal error. Anomalies produce
        higher reconstruction errors, enabling detection.
        
        #### 🎯 Use Cases
        
        - Real-time arrhythmia screening
        - Wearable device integration
        - Clinical decision support
        - Remote patient monitoring
        
        #### 👨‍💻 Developer
        
        Built with:
        - TensorFlow/Keras for deep learning
        - Streamlit for interactive dashboard
        - WFDB for ECG signal processing
        - Plotly for advanced visualizations
        
        ---
        
        **Note:** This is a research prototype. Always consult healthcare
        professionals for medical decisions.
        """)
        
        # Model info
        st.subheader("Model Information")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.info(f"""
            **Configuration:**
            - Latent Dimension: {config['model']['latent_dim']}
            - Window Length: {config['dataset']['window_length']} samples
            - Batch Size: {config['training']['batch_size']}
            - Epochs: {config['training']['epochs']}
            """)
        
        with col2:
            st.info(f"""
            **Model Parameters:**
            - Total Parameters: {vae.count_params():,}
            - Encoder Layers: 4 Conv1D
            - Decoder Layers: 4 Conv1DTranspose
            - Activation: ReLU (hidden), Linear (output)
            """)


if __name__ == "__main__":
    main()
