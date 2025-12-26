"""Parkinson's Telemonitoring Dataset preparation"""
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
from bacon.utils import SigmoidScaler

def _prepare_data_common(target='motor_UPDRS'):
    """Common data preparation logic for both sklearn and PyTorch models.
    
    Args:
        target: 'motor_UPDRS' or 'total_UPDRS'
    
    Returns:
        tuple: (X_train_np, X_test_np, y_train_np, y_test_np, feature_names)
            All arrays are numpy arrays ready for scaling
    """
    # Fetch dataset
    print(f"Loading Parkinson's Telemonitoring Dataset (target: {target})...")
    parkinsons = fetch_ucirepo(id=189)
    
    # Extract features and target
    X = parkinsons.data.features
    y = parkinsons.data.targets
    
    # Select target column
    if target == 'motor_UPDRS':
        y_values = y['motor_UPDRS'].values
        target_desc = "Motor UPDRS score (motor function)"
    else:
        y_values = y['total_UPDRS'].values
        target_desc = "Total UPDRS score (overall disease severity)"
    
    # Convert to binary classification: high (>= 75th percentile) vs low (< 75th percentile)
    threshold = np.percentile(y_values, 75)
    y_binary = (y_values >= threshold).astype(int)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Target: {target_desc}")
    print(f"75th percentile threshold: {threshold:.2f}")
    print(f"Class distribution: {np.bincount(y_binary)}")
    
    df = pd.DataFrame(X)
    df['target'] = y_binary
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
    
    print(f"\nClass Distribution (target: {target}):")
    print(f"  Low severity (0):  {(df['target'] == 0).sum()} samples ({(df['target'] == 0).sum() / len(df) * 100:.1f}%)")
    print(f"  High severity (1): {(df['target'] == 1).sum()} samples ({(df['target'] == 1).sum() / len(df) * 100:.1f}%)")
    print(f"  Threshold: {threshold:.2f} (75th percentile {target})")
    
    # Separate features and target
    X = df.drop(columns=['target', 'target_score'])
    y = df['target']
    
    feature_names = X.columns.tolist()
    
    # Train/test split
    X_train_df, X_test_df, y_train_np, y_test_np = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Convert DataFrames to numpy arrays
    X_train_np = X_train_df.values.astype(np.float64)
    X_test_np = X_test_df.values.astype(np.float64)
    y_train_np = y_train_np.to_numpy()
    y_test_np = y_test_np.to_numpy()
    
    print(f"\nTrain data shape: {X_train_np.shape}, dtype: {X_train_np.dtype}")
    print(f"Test data shape: {X_test_np.shape}, dtype: {X_test_np.dtype}")
    
    return X_train_np, X_test_np, y_train_np, y_test_np, feature_names


def prepare_data_sklearn(target='motor_UPDRS'):
    """Prepare Parkinson's Telemonitoring dataset for sklearn models (returns numpy arrays).
    
    Args:
        target: 'motor_UPDRS' or 'total_UPDRS'
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_names)
            All arrays are scaled numpy arrays ready for sklearn models
    """
    X_train_np, X_test_np, y_train_np, y_test_np, feature_names = _prepare_data_common(target)
    
    # Normalize features using SigmoidScaler
    scaler = SigmoidScaler(alpha=4, beta=-1)
    X_train_np = scaler.fit_transform(X_train_np)
    X_test_np = scaler.transform(X_test_np)
    
    return X_train_np, X_test_np, y_train_np, y_test_np, feature_names


def prepare_data(device, target='motor_UPDRS'):
    """Prepare Parkinson's Telemonitoring dataset for PyTorch models (returns tensors).
    
    Args:
        device: torch.device to place tensors on
        target: 'motor_UPDRS' or 'total_UPDRS'
        
    Returns:
        tuple: (X_train, Y_train, X_test, Y_test, feature_names)
            All arrays are PyTorch tensors on the specified device
    """
    X_train_np, X_test_np, y_train_np, y_test_np, feature_names = _prepare_data_common(target)

    # Normalize features using SigmoidScaler
    scaler = SigmoidScaler(alpha=4, beta=-1)
    X_train_np = scaler.fit_transform(X_train_np)
    X_test_np = scaler.transform(X_test_np)

    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train_np, dtype=torch.float32).to(device)
    Y_train = torch.tensor(y_train_np.reshape(-1, 1), dtype=torch.float32).to(device)
    Y_test = torch.tensor(y_test_np.reshape(-1, 1), dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test_np, dtype=torch.float32).to(device)

    return X_train, Y_train, X_test, Y_test, feature_names
