"""AIDS Clinical Trials Group Study 175 Dataset preparation"""
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
    print("Loading AIDS Clinical Trials Group Study 175 Dataset...")
    aids = fetch_ucirepo(id=890)
    
    # Extract features and target
    X = aids.data.features
    y = aids.data.targets
    
    # Debug: Check what columns are in targets
    print(f"Target columns: {y.columns.tolist()}")
    
    # Convert target: cid (censoring indicator for death)
    # cid=1 means death occurred, cid=0 means censored (alive)
    if 'cid' in y.columns:
        y_binary = y['cid'].values.astype(int)
    elif 'cid_True' in y.columns:
        y_binary = y['cid_True'].values.astype(int)
    else:
        # If neither exists, check all boolean columns
        print(f"Available target columns: {y.columns.tolist()}")
        raise ValueError("Could not find 'cid' or 'cid_True' in target columns")
    
    print(f"Dataset shape: {X.shape}")
    print(f"Target: Patient survival (cid)")
    print(f"Class distribution: Died={np.sum(y_binary)}, Alive={np.sum(1-y_binary)}")
    
    df = pd.DataFrame(X)
    df['target'] = y_binary
    
    # One-hot encode treatment (trt) variable
    # trt values: 0, 1, 2, 3 representing different treatment arms
    if 'trt' in df.columns:
        print(f"\nTreatment arms distribution:")
        print(df['trt'].value_counts().sort_index())
        
        # Create one-hot encoding for trt
        trt_dummies = pd.get_dummies(df['trt'], prefix='trt')
        
        # Drop original trt column and add one-hot encoded columns
        df = df.drop(columns=['trt'])
        df = pd.concat([df.drop(columns=['target']), trt_dummies, df[['target']]], axis=1)
    
    # Dataset Preview
    print("\n" + "="*60)
    print("DATASET PREVIEW - AIDS Clinical Trials Group Study 175")
    print("="*60)
    
    print("\nFeature Names and Descriptions:")
    feature_descriptions = {
        'age': 'Age at baseline (years)',
        'wtkg': 'Weight at baseline (kg)',
        'hemo': 'Hemophilia (0=no, 1=yes)',
        'homo': 'Homosexual activity (0=no, 1=yes)',
        'drugs': 'History of IV drug use (0=no, 1=yes)',
        'karnof': 'Karnofsky score (scale of 0-100)',
        'oprior': 'Non-ZDV antiretroviral therapy pre-175 (0=no, 1=yes)',
        'z30': 'ZDV in the 30 days prior to 175 (0=no, 1=yes)',
        'zprior': 'Days of prior ZDV therapy',
        'preanti': 'Days of prior antiretroviral therapy',
        'race': 'Race (0=white, 1=non-white)',
        'gender': 'Gender (0=female, 1=male)',
        'str2': 'Antiretroviral history (0=naive, 1=experienced)',
        'strat': 'Antiretroviral history stratification',
        'symptom': 'Symptomatic status (0=asymptomatic, 1=symptomatic)',
        'treat': 'Treatment indicator (0=ZDV only, 1=others)',
        'offtrt': 'Off-treatment indicator (0=no, 1=yes)',
        'cd40': 'CD4 count at baseline',
        'cd420': 'CD4 count at 20 weeks',
        'cd496': 'CD4 count at 96 weeks (=CD4 at end of follow-up)',
        'r': 'Missing CD4 indicator',
        'cd80': 'CD8 count at baseline',
        'cd820': 'CD8 count at 20 weeks',
        'trt_0': 'Treatment arm 0 (ZDV only)',
        'trt_1': 'Treatment arm 1 (ZDV + ddI)',
        'trt_2': 'Treatment arm 2 (ZDV + Zal)',
        'trt_3': 'Treatment arm 3 (ddI only)'
    }
    
    for col in df.columns[:-1]:  # Exclude target
        desc = feature_descriptions.get(col, 'Clinical measurement')
        print(f"  {col:20s} - {desc}")
    
    # Handle missing values
    print("\nMissing Values:")
    missing_counts = df.isnull().sum()
    has_missing = False
    for col in missing_counts[missing_counts > 0].index:
        print(f"  {col}: {missing_counts[col]} missing values")
        has_missing = True
    
    if has_missing:
        # Fill missing values with median for numerical columns
        for col in df.columns[:-1]:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
        print("\nMissing values filled with median")
    else:
        print("  No missing values")
    
    print("\nFeature Statistics (after preprocessing):")
    print(df.drop(columns=['target']).describe().round(2))
    
    print("\nSample Records (first 5):")
    print(df.head())
    
    print(f"\nClass Distribution:")
    print(f"  Alive (0): {(df['target'] == 0).sum()} samples ({(df['target'] == 0).sum() / len(df) * 100:.1f}%)")
    print(f"  Died (1):  {(df['target'] == 1).sum()} samples ({(df['target'] == 1).sum() / len(df) * 100:.1f}%)")
    
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
    print(f"Total features after one-hot encoding: {len(feature_names)}")
    
    return X_train_np, X_test_np, y_train_np, y_test_np, feature_names


def prepare_data_sklearn():
    """Prepare AIDS Clinical Trials dataset for sklearn models (returns numpy arrays).
    
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
    """Prepare AIDS Clinical Trials dataset for PyTorch models (returns tensors).
    
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
