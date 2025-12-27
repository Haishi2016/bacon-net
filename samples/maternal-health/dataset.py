"""Maternal Health Risk Dataset preparation"""
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
from bacon.utils import SigmoidScaler

def _prepare_data_common():
    """Common data preparation logic for both sklearn and PyTorch models.
    
    Returns:
        tuple: (X_train_np, X_test_np, y_train_np, y_test_np, feature_names)
            All arrays are numpy arrays ready for scaling
    """
    # Fetch dataset
    print("Loading Maternal Health Risk Dataset...")
    maternal_health = fetch_ucirepo(id=863)
    
    # Extract features and target
    X = maternal_health.data.features
    y = maternal_health.data.targets
    
    # Convert target: 'high risk' -> 1, others -> 0
    y_values = y['RiskLevel'].values
    y_binary = (y_values == 'high risk').astype(int)
    
    print(f"Dataset shape: {X.shape}")
    print(f"Target: High Risk prediction")
    print(f"Class distribution: High Risk={np.sum(y_binary)}, Not High Risk={np.sum(1-y_binary)}")
    
    df = pd.DataFrame(X)
    df['target'] = y_binary
    df['risk_level'] = y_values
    
    # Dataset Preview
    print("\n" + "="*60)
    print("DATASET PREVIEW - Maternal Health Risk")
    print("="*60)
    
    print("\nFeature Names and Descriptions:")
    feature_descriptions = {
        'Age': 'Age of pregnant woman (years)',
        'SystolicBP': 'Systolic blood pressure (mmHg)',
        'DiastolicBP': 'Diastolic blood pressure (mmHg)',
        'BS': 'Blood sugar level (mmol/L)',
        'BodyTemp': 'Body temperature (°F)',
        'HeartRate': 'Heart rate (beats per minute)'
    }
    
    for col in df.columns[:-2]:  # Exclude target and risk_level
        desc = feature_descriptions.get(col, 'Clinical measurement')
        print(f"  {col:20s} - {desc}")
    
    print("\nFeature Statistics:")
    print(df.drop(columns=['target', 'risk_level']).describe().round(2))
    
    print("\nSample Records (first 5):")
    print(df.head())
    
    print(f"\nRisk Level Distribution (original):")
    risk_counts = df['risk_level'].value_counts()
    for risk_level in risk_counts.index:
        count = risk_counts[risk_level]
        pct = count / len(df) * 100
        print(f"  {risk_level:15s}: {count} samples ({pct:.1f}%)")
    
    print(f"\nBinary Classification (target):")
    print(f"  Not High Risk (0): {(df['target'] == 0).sum()} samples ({(df['target'] == 0).sum() / len(df) * 100:.1f}%)")
    print(f"  High Risk (1):     {(df['target'] == 1).sum()} samples ({(df['target'] == 1).sum() / len(df) * 100:.1f}%)")
    
    # Separate features and target
    X = df.drop(columns=['target', 'risk_level'])
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


def prepare_data_sklearn():
    """Prepare Maternal Health Risk dataset for sklearn models (returns numpy arrays).
    
    Returns:
        tuple: (X_train, X_test, y_train, y_test, feature_names)
            All arrays are scaled numpy arrays ready for sklearn models
    """
    X_train_np, X_test_np, y_train_np, y_test_np, feature_names = _prepare_data_common()
    
    # Normalize features using SigmoidScaler
    scaler = SigmoidScaler(alpha=4, beta=-1)
    X_train_np = scaler.fit_transform(X_train_np)
    X_test_np = scaler.transform(X_test_np)
    
    return X_train_np, X_test_np, y_train_np, y_test_np, feature_names


def prepare_data(device):
    """Prepare Maternal Health Risk dataset for PyTorch models (returns tensors).
    
    Args:
        device: torch.device to place tensors on
        
    Returns:
        tuple: (X_train, Y_train, X_test, Y_test, feature_names)
            All arrays are PyTorch tensors on the specified device
    """
    X_train_np, X_test_np, y_train_np, y_test_np, feature_names = _prepare_data_common()

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
