"""Parkinson's Telemonitoring Dataset preparation"""
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
from bacon.utils import SigmoidScaler

def balance_data(X_train, Y_train, device):
    """Balance dataset by upsampling the minority class.
    
    Args:
        X_train: Training features tensor
        Y_train: Training labels tensor
        device: PyTorch device
    
    Returns:
        tuple: (X_train_balanced, Y_train_balanced)
    """
    print("\n" + "="*60)
    print("BALANCING DATASET")
    print("="*60)
    
    # Get class counts
    Y_np = Y_train.cpu().numpy().flatten()
    unique, counts = np.unique(Y_np, return_counts=True)
    
    print(f"\nOriginal class distribution:")
    for label, count in zip(unique, counts):
        print(f"  Class {int(label)}: {count} samples ({count/len(Y_np)*100:.1f}%)")
    
    # Find minority and majority classes
    minority_class = unique[np.argmin(counts)]
    majority_class = unique[np.argmax(counts)]
    minority_count = counts.min()
    majority_count = counts.max()
    
    print(f"\nMinority class: {int(minority_class)} ({minority_count} samples)")
    print(f"Majority class: {int(majority_class)} ({majority_count} samples)")
    
    # Separate classes
    minority_mask = Y_np == minority_class
    majority_mask = Y_np == majority_class
    
    X_minority = X_train[minority_mask]
    Y_minority = Y_train[minority_mask]
    X_majority = X_train[majority_mask]
    Y_majority = Y_train[majority_mask]
    
    # Upsample minority class
    n_samples_needed = majority_count - minority_count
    indices = torch.randint(0, len(X_minority), (n_samples_needed,))
    
    X_minority_upsampled = X_minority[indices]
    Y_minority_upsampled = Y_minority[indices]
    
    # Combine
    X_balanced = torch.cat([X_majority, X_minority, X_minority_upsampled], dim=0)
    Y_balanced = torch.cat([Y_majority, Y_minority, Y_minority_upsampled], dim=0)
    
    # Shuffle
    shuffle_idx = torch.randperm(len(X_balanced))
    X_balanced = X_balanced[shuffle_idx]
    Y_balanced = Y_balanced[shuffle_idx]
    
    print(f"\nBalanced dataset:")
    Y_balanced_np = Y_balanced.cpu().numpy().flatten()
    unique_balanced, counts_balanced = np.unique(Y_balanced_np, return_counts=True)
    for label, count in zip(unique_balanced, counts_balanced):
        print(f"  Class {int(label)}: {count} samples ({count/len(Y_balanced_np)*100:.1f}%)")
    
    print(f"\nTotal samples: {len(Y_np)} → {len(Y_balanced_np)}")
    print("="*60)
    
    return X_balanced, Y_balanced

def _prepare_data_common(target='total_UPDRS', mode='regression'):
    """Common data preparation logic for both sklearn and PyTorch models.
    
    Args:
        target: 'motor_UPDRS' or 'total_UPDRS'
        mode: 'regression' for continuous output, 'classification' for binary output
    
    Returns:
        tuple: (X_train_np, X_test_np, y_train_np, y_test_np, feature_names)
            All arrays are numpy arrays ready for scaling
    """
    # Load dataset from local files
    print(f"Loading Parkinson's Telemonitoring Dataset (target: {target}, mode: {mode})...")
    
    data_path = r"c:\School\uci\parkinson\parkinsons_updrs.data"
    
    # Read the data file
    df = pd.read_csv(data_path)
    
    print(f"Loaded data from: {data_path}")
    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    
    # Select target column
    if target == 'motor_UPDRS':
        y_values = df['motor_UPDRS'].values
        target_desc = "Motor UPDRS score (motor function)"
    else:
        y_values = df['total_UPDRS'].values
        target_desc = "Total UPDRS score (overall disease severity)"
    
    if mode == 'classification':
        # Convert to binary classification: high (>= 75th percentile) vs low (< 75th percentile)
        threshold = np.percentile(y_values, 75)
        y_binary = (y_values >= threshold).astype(int)
        
        print(f"Target: {target_desc}")
        print(f"75th percentile threshold: {threshold:.2f}")
        print(f"Class distribution: {np.bincount(y_binary)}")
        
        df['target'] = y_binary
        df['target_score'] = y_values
    else:
        # Use continuous values for regression
        print(f"Target: {target_desc}")
        print(f"Value range: [{y_values.min():.2f}, {y_values.max():.2f}]")
        print(f"Mean: {y_values.mean():.2f}, Std: {y_values.std():.2f}")
        
        df['target'] = y_values
        df['target_score'] = y_values
    
    # Dataset Preview
    print("\n" + "="*60)
    print(f"DATASET PREVIEW - Parkinson's Telemonitoring ({target})")
    print("="*60)
    
    print("\nFeature Names and Descriptions:")
    feature_descriptions = {
        'subject#': 'Subject identifier',
        'age': 'Age of the patient (years)',
        'sex': 'Sex (0=male, 1=female)',
        'test_time': 'Time since baseline (days)',
        'Jitter(%)': 'Jitter percentage - frequency variation',
        'Jitter(Abs)': 'Jitter absolute - frequency variation',
        'Jitter:RAP': 'Jitter RAP - relative average perturbation',
        'Jitter:PPQ5': 'Jitter PPQ5 - five-point period perturbation quotient',
        'Jitter:DDP': 'Jitter DDP - average absolute difference of differences',
        'Shimmer': 'Shimmer - amplitude variation',
        'Shimmer(dB)': 'Shimmer in dB',
        'Shimmer:APQ3': 'Shimmer APQ3 - three-point amplitude perturbation quotient',
        'Shimmer:APQ5': 'Shimmer APQ5 - five-point amplitude perturbation quotient',
        'Shimmer:APQ11': 'Shimmer APQ11 - eleven-point amplitude perturbation quotient',
        'Shimmer:DDA': 'Shimmer DDA - average absolute difference of differences',
        'NHR': 'Noise-to-harmonics ratio',
        'HNR': 'Harmonics-to-noise ratio',
        'RPDE': 'Recurrence period density entropy',
        'DFA': 'Detrended fluctuation analysis',
        'PPE': 'Pitch period entropy'
    }
    
    for col in df.columns[:-2]:  # Exclude target and target_score
        desc = feature_descriptions.get(col, 'Voice measurement')
        print(f"  {col:20s} - {desc}")
    
    print("\nFeature Statistics:")
    print(df.drop(columns=['target', 'target_score']).describe().round(2))
    
    print("\nSample Records (first 5):")
    print(df.head())
    
    if mode == 'classification':
        print(f"\nClass Distribution (target: {target}):")
        print(f"  Low severity (0):  {(df['target'] == 0).sum()} samples ({(df['target'] == 0).sum() / len(df) * 100:.1f}%)")
        print(f"  High severity (1): {(df['target'] == 1).sum()} samples ({(df['target'] == 1).sum() / len(df) * 100:.1f}%)")
        print(f"  Threshold: {threshold:.2f} (75th percentile {target})")
    else:
        print(f"\nTarget Statistics (target: {target}):")
        print(f"  Min:    {df['target'].min():.2f}")
        print(f"  Max:    {df['target'].max():.2f}")
        print(f"  Mean:   {df['target'].mean():.2f}")
        print(f"  Median: {df['target'].median():.2f}")
        print(f"  Std:    {df['target'].std():.2f}")
    
    # Separate features and target
    # Note: subject# will be used for proper train/test split, then dropped
    # Also drop the target columns from features
    columns_to_drop = ['target', 'target_score', 'motor_UPDRS', 'total_UPDRS']
    X = df.drop(columns=columns_to_drop)
    y = df['target']
    
    print(f"\nAvailable columns in dataframe: {list(X.columns)}")
    
    feature_names = X.columns.tolist()
    
    # Group-based train/test split to prevent data leakage
    # All measurements from the same subject must be in either train OR test, never both
    print("\n" + "="*60)
    print("SUBJECT-BASED TRAIN/TEST SPLIT")
    print("="*60)
    
    if 'subject#' in X.columns:
        # Get unique subjects and their labels
        if mode == 'classification':
            # For classification: use predominant class label for stratification
            subject_labels = df.groupby('subject#')['target'].agg(lambda x: x.mode()[0] if len(x.mode()) > 0 else x.iloc[0])
            unique_subjects = subject_labels.index.values
            subject_targets = subject_labels.values
            
            print(f"Total subjects: {len(unique_subjects)}")
            print(f"Total measurements: {len(df)}")
            print(f"Avg measurements per subject: {len(df) / len(unique_subjects):.1f}")
            
            # Split subjects (not measurements) into train and test with stratification
            train_subjects, test_subjects = train_test_split(
                unique_subjects, 
                test_size=0.2, 
                random_state=42, 
                stratify=subject_targets
            )
        else:
            # For regression: use median target value for stratified split based on severity
            subject_medians = df.groupby('subject#')['target'].median()
            unique_subjects = subject_medians.index.values
            # Create bins for stratification (low/medium/high severity)
            bins = [subject_medians.min()-1, subject_medians.quantile(0.33), 
                    subject_medians.quantile(0.67), subject_medians.max()+1]
            subject_bins = pd.cut(subject_medians, bins=bins, labels=[0, 1, 2]).values
            
            print(f"Total subjects: {len(unique_subjects)}")
            print(f"Total measurements: {len(df)}")
            print(f"Avg measurements per subject: {len(df) / len(unique_subjects):.1f}")
            print(f"Severity distribution: Low={np.sum(subject_bins==0)}, Med={np.sum(subject_bins==1)}, High={np.sum(subject_bins==2)}")
            
            # Split subjects (not measurements) into train and test with stratification by severity
            train_subjects, test_subjects = train_test_split(
                unique_subjects, 
                test_size=0.2, 
                random_state=42, 
                stratify=subject_bins
            )
        
        # Get all measurements for train and test subjects
        train_mask = df['subject#'].isin(train_subjects)
        test_mask = df['subject#'].isin(test_subjects)
        
        X_train_df = X[train_mask]
        X_test_df = X[test_mask]
        y_train_np = y[train_mask].to_numpy()
        y_test_np = y[test_mask].to_numpy()
        
        print(f"\nTrain set: {len(train_subjects)} subjects, {len(X_train_df)} measurements")
        print(f"Test set:  {len(test_subjects)} subjects, {len(X_test_df)} measurements")
        print(f"✓ No subject appears in both training and test sets")
        
        # Drop subject# column after split
        X_train_df = X_train_df.drop(columns=['subject#'])
        X_test_df = X_test_df.drop(columns=['subject#'])
        feature_names = [f for f in feature_names if f != 'subject#']
        
        print(f"\nDropped 'subject#' column after split")
        print(f"Remaining features: {len(feature_names)}")
    else:
        # Fallback to regular split if subject# not available
        print("Warning: 'subject#' column not found, using regular random split")
        if mode == 'classification':
            X_train_df, X_test_df, y_train_np, y_test_np = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
        else:
            # For regression, no stratification
            X_train_df, X_test_df, y_train_np, y_test_np = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
        y_train_np = y_train_np.to_numpy()
        y_test_np = y_test_np.to_numpy()
    
    print("="*60)
    
    # Convert DataFrames to numpy arrays
    X_train_np = X_train_df.values.astype(np.float64)
    X_test_np = X_test_df.values.astype(np.float64)
    
    # Normalize targets for regression mode to prevent numerical instability
    target_stats = None
    if mode == 'regression':
        y_min = y_train_np.min()
        y_max = y_train_np.max()
        
        print(f"\nNormalizing targets for regression using min-max:")
        print(f"  Original range: [{y_min:.2f}, {y_max:.2f}]")
        
        # Min-max normalize targets to [0, 1] range
        y_train_np = (y_train_np - y_min) / (y_max - y_min)
        y_test_np = (y_test_np - y_min) / (y_max - y_min)
        
        # Clip to ensure strict [0, 1] bounds (handles precision issues)
        y_train_np = np.clip(y_train_np, 0.0, 1.0)
        y_test_np = np.clip(y_test_np, 0.0, 1.0)
        
        print(f"  Normalized range: [{y_train_np.min():.6f}, {y_train_np.max():.6f}]")
        
        # Save stats for denormalization later
        target_stats = {'min': y_min, 'max': y_max}
    
    print(f"\nTrain data shape: {X_train_np.shape}, dtype: {X_train_np.dtype}")
    print(f"Test data shape: {X_test_np.shape}, dtype: {X_test_np.dtype}")
    
    return X_train_np, X_test_np, y_train_np, y_test_np, feature_names, target_stats


def prepare_data_sklearn(target='motor_UPDRS', mode='classification'):
    """Prepare Parkinson's Telemonitoring dataset for sklearn models (returns numpy arrays).
    
    Args:
        target: 'motor_UPDRS' or 'total_UPDRS'
        mode: 'regression' for continuous output, 'classification' for binary output
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_names, target_stats)
            All arrays are scaled numpy arrays ready for sklearn models
            target_stats: dict with scaler params for denormalization (None for classification)
    """
    X_train_np, X_test_np, y_train_np, y_test_np, feature_names, target_stats = _prepare_data_common(target, mode)
    
    # Normalize features using min-max if regression, else SigmoidScaler
    if mode == 'regression':
        # Min-max normalization for features
        X_min = X_train_np.min(axis=0)
        X_max = X_train_np.max(axis=0)
        X_range = X_max - X_min
        X_range[X_range == 0] = 1.0  # Avoid division by zero
        X_train_np = (X_train_np - X_min) / X_range
        X_test_np = (X_test_np - X_min) / X_range
        # Clip to ensure strict [0, 1] bounds
        X_train_np = np.clip(X_train_np, 0.0, 1.0)
        X_test_np = np.clip(X_test_np, 0.0, 1.0)
    else:
        # Use SigmoidScaler for classification
        scaler = SigmoidScaler(alpha=4, beta=-1)
        X_train_np = scaler.fit_transform(X_train_np)
        X_test_np = scaler.transform(X_test_np)
    
    return X_train_np, X_test_np, y_train_np, y_test_np, feature_names, target_stats


def prepare_data(device, target='motor_UPDRS', mode='classification'):
    """Prepare Parkinson's Telemonitoring dataset for PyTorch models (returns tensors).
    
    Args:
        device: torch.device to place tensors on
        target: 'motor_UPDRS' or 'total_UPDRS'
        mode: 'regression' for continuous output, 'classification' for binary output
        
    Returns:
        tuple: (X_train, Y_train, X_test, Y_test, feature_names, target_stats)
            All arrays are PyTorch tensors on the specified device
            target_stats: dict with scaler params for denormalization (None for classification)
    """
    X_train_np, X_test_np, y_train_np, y_test_np, feature_names, target_stats = _prepare_data_common(target, mode)

    # Normalize features using min-max if regression, else SigmoidScaler
    if mode == 'regression':
        # Min-max normalization for features
        X_min = X_train_np.min(axis=0)
        X_max = X_train_np.max(axis=0)
        X_range = X_max - X_min
        X_range[X_range == 0] = 1.0  # Avoid division by zero
        X_train_np = (X_train_np - X_min) / X_range
        X_test_np = (X_test_np - X_min) / X_range
        # Clip to ensure strict [0, 1] bounds
        X_train_np = np.clip(X_train_np, 0.0, 1.0)
        X_test_np = np.clip(X_test_np, 0.0, 1.0)
    else:
        # Use SigmoidScaler for classification
        scaler = SigmoidScaler(alpha=4, beta=-1)
        X_train_np = scaler.fit_transform(X_train_np)
        X_test_np = scaler.transform(X_test_np)

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train_np, dtype=torch.float32).to(device)
    Y_train = torch.tensor(y_train_np.reshape(-1, 1), dtype=torch.float32).to(device)
    Y_test = torch.tensor(y_test_np.reshape(-1, 1), dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test_np, dtype=torch.float32).to(device)

    return X_train, Y_train, X_test, Y_test, feature_names, target_stats

