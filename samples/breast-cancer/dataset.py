"""Breast Cancer Wisconsin Dataset preparation"""
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from ucimlrepo import fetch_ucirepo
from bacon.utils import SigmoidScaler

def _prepare_data_common():
    """Common data preparation logic for both sklearn and PyTorch models.
    
    Returns:
        tuple: (X_train_np, X_test_np, y_train_np, y_test_np, feature_names)
            All arrays are numpy arrays ready for scaling
    """
    # Fetch dataset
    print("Loading Breast Cancer Wisconsin Dataset...")
    breast_cancer = fetch_ucirepo(id=17)
    
    # Extract features (mean values only - first 30 features)
    X = breast_cancer.data.features.iloc[:, 0:30]
    y = LabelEncoder().fit_transform(breast_cancer.data.targets.values.ravel())
    
    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution: Malignant={np.sum(y)}, Benign={np.sum(1-y)}")
    
    df = pd.DataFrame(X, columns=breast_cancer.data.features.columns[:30])
    df['target'] = y
    
    # Dataset Preview
    print("\n" + "="*60)
    print("DATASET PREVIEW - Breast Cancer Wisconsin")
    print("="*60)
    
    print("\nFeature Names (Mean Values):")
    print("  All features are mean values of cell nuclei characteristics")
    print("  computed from digitized images of fine needle aspirate")
    
    feature_names = X.columns.tolist()
    for i, name in enumerate(feature_names, 1):
        print(f"  {i:2d}. {name}")
    
    print("\nFeature Statistics:")
    print(df.drop(columns=['target']).describe().round(2))
    
    print("\nSample Records (first 5):")
    print(df.head())
    
    print(f"\nClass Distribution:")
    print(f"  Benign (0):    {(df['target'] == 0).sum()} samples ({(df['target'] == 0).sum() / len(df) * 100:.1f}%)")
    print(f"  Malignant (1): {(df['target'] == 1).sum()} samples ({(df['target'] == 1).sum() / len(df) * 100:.1f}%)")
    
    # Separate features and target
    X = df.drop(columns=['target'])
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
    """Prepare Breast Cancer dataset for sklearn models (returns numpy arrays).
    
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
    """Prepare Breast Cancer dataset for PyTorch models (returns tensors).
    
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
