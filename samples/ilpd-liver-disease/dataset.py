"""ILPD (Indian Liver Patient Dataset) preparation"""
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
from bacon.utils import SigmoidScaler

def prepare_data(device):
    """Prepare ILPD (Indian Liver Patient Dataset)
    
    Returns:
        X_train, Y_train, X_test, Y_test, feature_names
    """
    
    # Fetch ILPD dataset
    print("Loading ILPD (Indian Liver Patient Dataset)...")
    ilpd = fetch_ucirepo(id=225)
    
    # Extract features and target
    X = ilpd.data.features.copy()  # Make explicit copy to avoid SettingWithCopyWarning
    y = ilpd.data.targets.iloc[:, 0]  # 'Selector' column
    
    # Convert target: 1 (disease) -> 1, 2 (no disease) -> 0
    y_binary = (y == 1).astype(int).values
    
    # Handle categorical features (Gender: Male/Female) - MUST DO BEFORE FILLING MISSING VALUES
    if 'Gender' in X.columns:
        X['Gender'] = X['Gender'].map({'Male': 1, 'Female': 0})
    
    # Handle missing values
    if X.isnull().any().any():
        X = X.fillna(X.median())
    
    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution: {np.bincount(y_binary)}")
    
    df = pd.DataFrame(X)
    df['target'] = y_binary
    
    # Dataset Preview
    print("\n" + "="*60)
    print("DATASET PREVIEW - ILPD (Indian Liver Patient Dataset)")
    print("="*60)
    
    print("\nFeature Names and Descriptions:")
    feature_descriptions = {
        'Age': 'Age of the patient (years)',
        'Gender': 'Gender (1=Male, 0=Female)',
        'Total_Bilirubin': 'Total Bilirubin (mg/dL)',
        'Direct_Bilirubin': 'Direct Bilirubin (mg/dL)',
        'Total_Protiens': 'Total Proteins (g/dL)',
        'Albumin': 'Albumin (g/dL)',
        'A/G_Ratio': 'Albumin/Globulin Ratio',
        'SGPT': 'Serum Glutamic Pyruvic Transaminase (IU/L)',
        'SGOT': 'Serum Glutamic Oxaloacetic Transaminase (IU/L)',
        'Alkphos': 'Alkaline Phosphatase (IU/L)'
    }
    
    for col in df.columns[:-1]:  # Exclude target
        desc = feature_descriptions.get(col, col)
        print(f"  {col:20s} - {desc}")
    
    print("\nFeature Statistics:")
    print(df.describe().round(2))
    
    print("\nSample Records (first 5):")
    print(df.head())
    
    print("\nClass Distribution:")
    print(f"  No Disease (0): {(df['target'] == 0).sum()} patients ({(df['target'] == 0).sum() / len(df) * 100:.1f}%)")
    print(f"  Disease (1):    {(df['target'] == 1).sum()} patients ({(df['target'] == 1).sum() / len(df) * 100:.1f}%)")
    
    # Separate features and target
    X = df.drop(columns=['target'])
    y = df['target']
    
    feature_names = X.columns.tolist()
    
    # Train/test split
    X_train_df, X_test_df, y_train_np, y_test_np = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Convert DataFrames to numpy arrays for scaling
    X_train_np = X_train_df.values.astype(np.float64)
    X_test_np = X_test_df.values.astype(np.float64)
    
    print(f"\nTrain data shape: {X_train_np.shape}, dtype: {X_train_np.dtype}")
    print(f"Test data shape: {X_test_np.shape}, dtype: {X_test_np.dtype}")
    
    # Normalize features using SigmoidScaler
    scaler = SigmoidScaler(alpha=4, beta=-1)
    X_train_np = scaler.fit_transform(X_train_np)
    X_test_np = scaler.transform(X_test_np)
    
    # Convert to PyTorch tensors
    X_train = torch.tensor(X_train_np, dtype=torch.float32).to(device)
    Y_train = torch.tensor(y_train_np.to_numpy().reshape(-1, 1), dtype=torch.float32).to(device)
    Y_test = torch.tensor(y_test_np.to_numpy().reshape(-1, 1), dtype=torch.float32).to(device)
    X_test = torch.tensor(X_test_np, dtype=torch.float32).to(device)
    
    return X_train, Y_train, X_test, Y_test, feature_names
