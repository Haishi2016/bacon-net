"""Maternal Health Risk Dataset preparation"""
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
    y_binary = (y_values != 'low risk').astype(int)
    
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
    
    # Feature Correlation Analysis
    print("\n" + "="*60)
    print("FEATURE CORRELATION ANALYSIS WITH TARGET")
    print("="*60)
    
    # Calculate correlations with target
    features_for_corr = df.drop(columns=['risk_level'])
    correlations = features_for_corr.corr()['target'].drop('target').sort_values(ascending=False)
    
    print("\nCorrelation with Target (High Risk):")
    print("Feature               Correlation")
    print("-" * 40)
    for feature, corr in correlations.items():
        direction = "↑" if corr > 0 else "↓"
        strength = "Strong" if abs(corr) > 0.5 else "Moderate" if abs(corr) > 0.3 else "Weak"
        print(f"{feature:20s} {corr:8.4f} {direction} {strength}")
    
    print(f"\nInterpretation:")
    print(f"  Positive correlation: Higher feature values → Higher risk")
    print(f"  Negative correlation: Higher feature values → Lower risk")
    print(f"  |r| > 0.5: Strong, |r| > 0.3: Moderate, |r| ≤ 0.3: Weak")
    
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
