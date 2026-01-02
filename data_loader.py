"""
WFDB Data Loader for MIT-BIH Arrhythmia Database
Handles reading ECG signals and annotations from WFDB format
"""

import os
import numpy as np
import wfdb
from typing import Tuple, List, Dict
import yaml
from tqdm import tqdm
import h5py


class WFDBDataLoader:
    """
    Advanced WFDB data loader for MIT-BIH Arrhythmia Database
    
    Features:
    - Automatic annotation parsing and categorization
    - Beat segmentation with configurable windows
    - Quality filtering (remove noisy segments)
    - Progress tracking with tqdm
    """
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize data loader with configuration"""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.data_dir = self.config['dataset']['raw_data_dir']
        self.processed_dir = self.config['dataset']['processed_data_dir']
        self.window_length = self.config['dataset']['window_length']
        self.sampling_rate = self.config['dataset']['sampling_rate']
        self.overlap = self.config['dataset']['overlap']
        
        # Annotation categories
        self.normal_beats = set(self.config['dataset']['normal_beats'])
        self.anomaly_beats = set(self.config['dataset']['anomaly_beats'])
        
        # Statistics
        self.stats = {
            'total_beats': 0,
            'normal_beats': 0,
            'anomaly_beats': 0,
            'records_processed': 0
        }
    
    def get_record_list(self) -> List[str]:
        """Read available record IDs from RECORDS file"""
        records_file = self.config['dataset']['records_file']
        
        if os.path.exists(records_file):
            with open(records_file, 'r') as f:
                records = [line.strip() for line in f if line.strip()]
        else:
            # Fallback: scan directory for .hea files
            records = []
            for file in os.listdir(self.data_dir):
                if file.endswith('.hea'):
                    records.append(file.replace('.hea', ''))
        
        return sorted(records)
    
    def load_record(self, record_name: str) -> Tuple[np.ndarray, wfdb.Annotation]:
        """
        Load a single ECG record with annotations
        
        Args:
            record_name: Record identifier (e.g., '100')
        
        Returns:
            signal: ECG signal array (samples, channels)
            annotation: WFDB annotation object
        """
        record_path = os.path.join(self.data_dir, record_name)
        
        # Read signal
        record = wfdb.rdrecord(record_path)
        signal = record.p_signal  # Physical units
        
        # Read annotations
        annotation = wfdb.rdann(record_path, 'atr')
        
        return signal, annotation
    
    def segment_beats(
        self, 
        signal: np.ndarray, 
        annotation: wfdb.Annotation,
        channel: int = 0
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Segment ECG signal into fixed-length windows around R-peaks
        
        Args:
            signal: ECG signal (samples, channels)
            annotation: WFDB annotation with R-peak locations
            channel: ECG channel to use (default: 0 for MLII)
        
        Returns:
            segments: Array of ECG windows (num_beats, window_length)
            labels: Binary labels (0=normal, 1=anomaly)
        """
        ecg = signal[:, channel]
        r_peaks = annotation.sample
        beat_types = annotation.symbol
        
        segments = []
        labels = []
        
        # Calculate window boundaries
        half_window = self.window_length // 2
        
        for peak, beat_type in zip(r_peaks, beat_types):
            # Skip if window extends beyond signal boundaries
            if peak - half_window < 0 or peak + half_window > len(ecg):
                continue
            
            # Extract window
            start = peak - half_window
            end = start + self.window_length
            segment = ecg[start:end]
            
            # Ensure exact length
            if len(segment) != self.window_length:
                continue
            
            # Categorize beat
            if beat_type in self.normal_beats:
                label = 0
                self.stats['normal_beats'] += 1
            elif beat_type in self.anomaly_beats:
                label = 1
                self.stats['anomaly_beats'] += 1
            else:
                continue  # Skip unknown beat types
            
            segments.append(segment)
            labels.append(label)
            self.stats['total_beats'] += 1
        
        return np.array(segments), np.array(labels)
    
    def quality_check(self, segment: np.ndarray, threshold: float = 0.5) -> bool:
        """
        Check if ECG segment has acceptable quality
        
        Args:
            segment: ECG window
            threshold: Maximum acceptable flatline ratio
        
        Returns:
            True if quality is acceptable
        """
        # Check for flatlines (consecutive identical values)
        diff = np.diff(segment)
        flatline_ratio = np.sum(np.abs(diff) < 1e-6) / len(diff)
        
        # Check for extreme values (potential artifacts)
        if np.any(np.abs(segment) > 10 * np.std(segment)):
            return False
        
        return flatline_ratio < threshold
    
    def process_all_records(
        self, 
        save_hdf5: bool = True,
        quality_filter: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Process all records and create dataset
        
        Args:
            save_hdf5: Save processed data to HDF5 for faster loading
            quality_filter: Apply quality filtering
        
        Returns:
            X: ECG segments (num_samples, window_length)
            y: Labels (num_samples,)
        """
        records = self.get_record_list()
        all_segments = []
        all_labels = []
        
        print(f"Processing {len(records)} records from MIT-BIH database...")
        
        for record_name in tqdm(records, desc="Loading records"):
            try:
                signal, annotation = self.load_record(record_name)
                segments, labels = self.segment_beats(signal, annotation)
                
                # Quality filtering
                if quality_filter:
                    valid_indices = [
                        i for i, seg in enumerate(segments) 
                        if self.quality_check(seg)
                    ]
                    segments = segments[valid_indices]
                    labels = labels[valid_indices]
                
                all_segments.append(segments)
                all_labels.append(labels)
                self.stats['records_processed'] += 1
                
            except Exception as e:
                print(f"Error processing {record_name}: {str(e)}")
                continue
        
        # Concatenate all data
        X = np.concatenate(all_segments, axis=0)
        y = np.concatenate(all_labels, axis=0)
        
        # Print statistics
        self.print_statistics()
        
        # Save to HDF5
        if save_hdf5:
            self.save_to_hdf5(X, y)
        
        return X, y
    
    def save_to_hdf5(self, X: np.ndarray, y: np.ndarray):
        """Save processed data to HDF5 format"""
        os.makedirs(self.processed_dir, exist_ok=True)
        filepath = os.path.join(self.processed_dir, 'ecg_data.h5')
        
        with h5py.File(filepath, 'w') as f:
            f.create_dataset('X', data=X, compression='gzip')
            f.create_dataset('y', data=y, compression='gzip')
            
            # Save metadata
            f.attrs['window_length'] = self.window_length
            f.attrs['sampling_rate'] = self.sampling_rate
            f.attrs['total_beats'] = self.stats['total_beats']
            f.attrs['normal_beats'] = self.stats['normal_beats']
            f.attrs['anomaly_beats'] = self.stats['anomaly_beats']
        
        print(f"Saved processed data to {filepath}")
    
    def load_from_hdf5(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load processed data from HDF5"""
        filepath = os.path.join(self.processed_dir, 'ecg_data.h5')
        
        if not os.path.exists(filepath):
            raise FileNotFoundError(
                f"Processed data not found at {filepath}. "
                "Run process_all_records() first."
            )
        
        with h5py.File(filepath, 'r') as f:
            X = f['X'][:]
            y = f['y'][:]
            
            # Load metadata
            self.stats['window_length'] = f.attrs['window_length']
            self.stats['total_beats'] = f.attrs['total_beats']
            self.stats['normal_beats'] = f.attrs['normal_beats']
            self.stats['anomaly_beats'] = f.attrs['anomaly_beats']
        
        print(f"Loaded {len(X)} samples from HDF5")
        self.print_statistics()
        
        return X, y
    
    def print_statistics(self):
        """Print dataset statistics"""
        print("\n" + "="*60)
        print("MIT-BIH Dataset Statistics")
        print("="*60)
        print(f"Records processed: {self.stats['records_processed']}")
        print(f"Total beats: {self.stats['total_beats']}")
        print(f"Normal beats: {self.stats['normal_beats']} "
              f"({100*self.stats['normal_beats']/max(self.stats['total_beats'],1):.1f}%)")
        print(f"Anomaly beats: {self.stats['anomaly_beats']} "
              f"({100*self.stats['anomaly_beats']/max(self.stats['total_beats'],1):.1f}%)")
        print("="*60 + "\n")


if __name__ == "__main__":
    # Test data loader
    loader = WFDBDataLoader()
    
    # Process all records
    X, y = loader.process_all_records(save_hdf5=True, quality_filter=True)
    
    print(f"Final dataset shape: X={X.shape}, y={y.shape}")
