"""
Master Script - Run Complete Pipeline

This script orchestrates the entire ECG anomaly detection pipeline:
1. Data loading and preprocessing
2. Model training
3. Anomaly detection
4. Comprehensive evaluation
5. Results visualization
"""

import os
import sys
import argparse
import yaml
import numpy as np
from datetime import datetime


def run_pipeline(
    skip_preprocessing: bool = False,
    skip_training: bool = False,
    config_path: str = "config.yaml"
):
    """
    Run complete ECG anomaly detection pipeline
    
    Args:
        skip_preprocessing: Skip data preprocessing step
        skip_training: Skip model training step
        config_path: Path to configuration file
    """
    
    print("\n" + "="*80)
    print("ECG ANOMALY DETECTION - COMPLETE PIPELINE")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80 + "\n")
    
    # Step 1: Data Loading and Preprocessing
    if not skip_preprocessing:
        print("\n" + "-"*80)
        print("STEP 1: DATA LOADING AND PREPROCESSING")
        print("-"*80 + "\n")
        
        from data_loader import WFDBDataLoader
        from preprocessing import ECGPreprocessor
        
        # Load raw data
        loader = WFDBDataLoader(config_path)
        
        try:
            print("Attempting to load preprocessed data...")
            X, y = loader.load_from_hdf5()
        except FileNotFoundError:
            print("Preprocessed data not found. Processing raw WFDB files...")
            X, y = loader.process_all_records(save_hdf5=True, quality_filter=True)
        
        # Preprocess
        preprocessor = ECGPreprocessor(config_path)
        data = preprocessor.process_pipeline(X, y, augment_train=True, save_processed=True)
        
        print("\n✓ Data preprocessing complete!")
    else:
        print("\n⊗ Skipping preprocessing step (using existing processed data)")
    
    # Step 2: Model Training
    if not skip_training:
        print("\n" + "-"*80)
        print("STEP 2: VAE MODEL TRAINING")
        print("-"*80 + "\n")
        
        from train import VAETrainer
        
        trainer = VAETrainer(config_path)
        trainer.load_data()
        trainer.build_model()
        history = trainer.train()
        trainer.visualize_reconstructions(num_samples=10)
        
        print("\n✓ Model training complete!")
    else:
        print("\n⊗ Skipping training step (using existing trained model)")
    
    # Step 3: Anomaly Detection
    print("\n" + "-"*80)
    print("STEP 3: ANOMALY DETECTION")
    print("-"*80 + "\n")
    
    from tensorflow import keras
    from preprocessing import ECGPreprocessor
    from anomaly_detection import AnomalyDetector
    
    # Load data
    preprocessor = ECGPreprocessor(config_path)
    data = preprocessor.load_processed_data()
    
    X_val = data['X_val']
    y_val = data['y_val']
    X_test = data['X_test']
    y_test = data['y_test']
    
    # Load model
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    model_path = os.path.join(config['paths']['models_dir'], 'vae_best_model.h5')
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        print("Please run with --no-skip-training to train the model first.")
        sys.exit(1)
    
    # Import and build VAE model
    from vae_model import VAE
    
    # Build the model with correct architecture
    vae = VAE(config_path)
    
    # Build model by calling it with dummy data
    dummy_input = np.zeros((1, config['dataset']['window_length'], 1), dtype=np.float32)
    _ = vae(dummy_input)
    
    # Load weights
    vae.load_weights(model_path)
    
    # Initialize detector
    detector = AnomalyDetector(vae, config_path)
    
    # Fit thresholds
    detector.fit_thresholds(X_val, y_val, k=3.0)
    
    # Predict
    predictions, scores_dict = detector.predict(X_test)
    
    # Save predictions
    predictions_dir = config['paths']['predictions_dir']
    os.makedirs(predictions_dir, exist_ok=True)
    
    output_path = os.path.join(predictions_dir, 'test_predictions.csv')
    detector.save_predictions(predictions, y_test, scores_dict, output_path)
    
    print("\n✓ Anomaly detection complete!")
    
    # Step 4: Comprehensive Evaluation
    print("\n" + "-"*80)
    print("STEP 4: COMPREHENSIVE EVALUATION")
    print("-"*80 + "\n")
    
    from evaluate import ModelEvaluator
    
    evaluator = ModelEvaluator(config_path)
    
    # Full evaluation
    scores = scores_dict['reconstruction_error']
    threshold = detector.thresholds['reconstruction_error']
    
    metrics = evaluator.evaluate_full(
        vae_model=vae,
        X_test=X_test,
        y_test=y_test,
        predictions=predictions,
        scores=scores,
        threshold=threshold
    )
    
    print("\n✓ Evaluation complete!")
    
    # Final Summary
    print("\n" + "="*80)
    print("PIPELINE EXECUTION SUMMARY")
    print("="*80)
    print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nResults saved to:")
    print(f"  - Models: {config['paths']['models_dir']}")
    print(f"  - Results: {config['paths']['results_dir']}")
    print(f"  - Predictions: {predictions_dir}")
    print(f"  - Logs: {config['paths']['logs_dir']}")
    
    print("\nPerformance Metrics:")
    print(f"  - Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  - Precision: {metrics['precision']:.4f}")
    print(f"  - Recall:    {metrics['recall']:.4f}")
    print(f"  - F1-Score:  {metrics['f1_score']:.4f}")
    print(f"  - ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"  - PR-AUC:    {metrics['pr_auc']:.4f}")
    
    print("\n" + "="*80)
    print("🎉 PIPELINE COMPLETED SUCCESSFULLY! 🎉")
    print("="*80)
    
    print("\nNext steps:")
    print("  1. Review results in the 'results/' directory")
    print("  2. Launch Streamlit dashboard: streamlit run streamlit_app.py")
    print("  3. Check TensorBoard logs: tensorboard --logdir logs/")
    print("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run complete ECG anomaly detection pipeline"
    )
    
    parser.add_argument(
        '--skip-preprocessing',
        action='store_true',
        help='Skip data preprocessing step (use existing processed data)'
    )
    
    parser.add_argument(
        '--skip-training',
        action='store_true',
        help='Skip model training step (use existing trained model)'
    )
    
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file (default: config.yaml)'
    )
    
    args = parser.parse_args()
    
    try:
        run_pipeline(
            skip_preprocessing=args.skip_preprocessing,
            skip_training=args.skip_training,
            config_path=args.config
        )
    except KeyboardInterrupt:
        print("\n\nPipeline interrupted by user.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nError during pipeline execution: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
