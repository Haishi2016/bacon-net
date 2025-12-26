"""Cervical Cancer Behavior Risk dataset preparation"""
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo
from bacon.utils import SigmoidScaler

def prepare_data(device):
    """Prepare Cervical Cancer Behavior Risk dataset
    
    Returns:
        X_train, Y_train, X_test, Y_test, feature_names
    """
    
    # Fetch dataset
    print("Loading Cervical Cancer Behavior Risk dataset...")
    cervical = fetch_ucirepo(id=537)
    
    # Extract features and target
    X = cervical.data.features
    y = cervical.data.targets
    
    # Target is ca_cervix: 0 = no cervical cancer, 1 = cervical cancer
    y_binary = y.values.ravel()
    
    print(f"Dataset shape: {X.shape}")
    print(f"Class distribution: {np.bincount(y_binary)}")
    
    df = pd.DataFrame(X)
    df['target'] = y_binary
    
    # Dataset Preview
    print("\n" + "="*60)
    print("DATASET PREVIEW")
    print("="*60)
    
    print("\nFeature Names and Descriptions:")
    feature_descriptions = {
        'behavior_eating': 'Eating behavior score',
        'behavior_personalHygiene': 'Personal hygiene behavior score',
        'behavior_sexualRisk': 'Sexual risk behavior score',
        'intention_aggregation': 'Aggregated intention score',
        'intention_commitment': 'Commitment intention score',
        'attitude_consistency': 'Attitude consistency score',
        'attitude_spontaneity': 'Attitude spontaneity score',
        'norm_significantPerson': 'Significant person norm score',
        'norm_fulfillment': 'Norm fulfillment score',
        'perception_vulnerability': 'Perceived vulnerability score',
        'perception_severity': 'Perceived severity score',
        'motivation_strength': 'Motivation strength score',
        'motivation_willingness': 'Motivation willingness score',
        'socialSupport_emotionality': 'Emotional social support score',
        'socialSupport_appreciation': 'Appreciation social support score',
        'socialSupport_instrumental': 'Instrumental social support score',
        'empowerment_knowledge': 'Empowerment knowledge score',
        'empowerment_abilities': 'Empowerment abilities score',
        'empowerment_desires': 'Empowerment desires score'
    }
    
    for col in df.columns[:-1]:  # Exclude target
        desc = feature_descriptions.get(col, 'Behavioral/psychological measure')
        print(f"  {col:30s} - {desc}")
    
    print("\nFeature Statistics:")
    print(df.describe().round(2))
    
    print("\nSample Records (first 5):")
    print(df.head())
    
    print("\nClass Distribution:")
    print(f"  No Cervical Cancer (0): {(df['target'] == 0).sum()} people ({(df['target'] == 0).sum() / len(df) * 100:.1f}%)")
    print(f"  Cervical Cancer (1):    {(df['target'] == 1).sum()} people ({(df['target'] == 1).sum() / len(df) * 100:.1f}%)")
    
    # Separate features and target
    X = df.drop(columns=['target'])
    y = df['target']
    
    feature_names = X.columns.tolist()
    
    print("\n" + "="*60)
    print("FEATURE TYPES")
    print("="*60)
    print("All features are already numeric (integer scores)")
    print("No one-hot encoding required")
    
    # Train/test split (stratified to maintain class balance in small dataset)
    X_train_df, X_test_df, y_train_np, y_test_np = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
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
