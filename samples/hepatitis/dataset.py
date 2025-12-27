"""Hepatitis Dataset preparation"""
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
    print("Loading Hepatitis Dataset...")
    hepatitis = fetch_ucirepo(id=46)
    
    # Extract features and target
    X = hepatitis.data.features
    y = hepatitis.data.targets
    
    # Convert target: 1 (die) -> 1, 2 (live) -> 0
    y_values = y['Class'].values
    y_binary = (y_values == 1).astype(int)  # 1=die, 0=live
    
    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution: Die={np.sum(y_binary)}, Live={np.sum(1-y_binary)}")
    
    df = pd.DataFrame(X)
    df['target'] = y_binary
    
    # Dataset Preview
    print("\n" + "="*60)
    print("DATASET PREVIEW - Hepatitis")
    print("="*60)
    
    print("\nFeature Names and Descriptions:")
    feature_descriptions = {
        'AGE': 'Age of patient (years)',
        'SEX': 'Sex (1=male, 2=female)',
        'STEROID': 'Steroid treatment (1=no, 2=yes)',
        'ANTIVIRALS': 'Antiviral treatment (1=no, 2=yes)',
        'FATIGUE': 'Fatigue symptom (1=no, 2=yes)',
        'MALAISE': 'Malaise symptom (1=no, 2=yes)',
        'ANOREXIA': 'Anorexia symptom (1=no, 2=yes)',
        'LIVER BIG': 'Liver big (1=no, 2=yes)',
        'LIVER FIRM': 'Liver firm (1=no, 2=yes)',
        'SPLEEN PALPABLE': 'Spleen palpable (1=no, 2=yes)',
        'SPIDERS': 'Spider angiomas (1=no, 2=yes)',
        'ASCITES': 'Ascites (1=no, 2=yes)',
        'VARICES': 'Esophageal varices (1=no, 2=yes)',
        'BILIRUBIN': 'Serum bilirubin (mg/dL)',
        'ALK PHOSPHATE': 'Alkaline phosphatase (U/L)',
        'SGOT': 'Serum glutamic-oxaloacetic transaminase (U/L)',
        'ALBUMIN': 'Serum albumin (g/dL)',
        'PROTIME': 'Prothrombin time (seconds)',
        'HISTOLOGY': 'Liver histology (1=no, 2=yes)'
    }
    
    for col in df.columns[:-1]:  # Exclude target
        desc = feature_descriptions.get(col, 'Clinical measurement')
        print(f"  {col:20s} - {desc}")
    
    # Handle missing values
    print("\nMissing Values:")
    missing_counts = df.isnull().sum()
    for col in missing_counts[missing_counts > 0].index:
        print(f"  {col}: {missing_counts[col]} missing values")
    
    # Fill missing values with median for numerical columns
    for col in df.columns[:-1]:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    
    print("\nFeature Statistics (after filling missing values):")
    print(df.drop(columns=['target']).describe().round(2))
    
    print("\nSample Records (first 5):")
    print(df.head())
    
    print(f"\nClass Distribution:")
    print(f"  Live (0):  {(df['target'] == 0).sum()} samples ({(df['target'] == 0).sum() / len(df) * 100:.1f}%)")
    print(f"  Die (1):   {(df['target'] == 1).sum()} samples ({(df['target'] == 1).sum() / len(df) * 100:.1f}%)")
    
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
    """Prepare Hepatitis dataset for sklearn models (returns numpy arrays).
    
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
    """Prepare Hepatitis dataset for PyTorch models (returns tensors).
    
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
