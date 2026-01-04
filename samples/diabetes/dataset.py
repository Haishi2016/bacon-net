# Dataset preparation for CDC Diabetes Health Indicators

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
import torch
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '../../')
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


def prepare_data(device):
    """Load and prepare CDC Diabetes Health Indicators dataset.
    
    Returns:
        tuple: (X_train, Y_train, X_test, Y_test, feature_names)
    """
    
    # Fetch CDC Diabetes Health Indicators dataset
    print("Loading CDC Diabetes Health Indicators dataset...")
    diabetes = fetch_ucirepo(id=891)
    
    # Extract features and target
    X = diabetes.data.features
    y = diabetes.data.targets
    
    # Target is Diabetes_binary: 0 = no diabetes, 1 = prediabetes or diabetes
    y_binary = y['Diabetes_binary'].values
    
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
        'HighBP': 'High blood pressure (0=no, 1=yes)',
        'HighChol': 'High cholesterol (0=no, 1=yes)',
        'CholCheck': 'Cholesterol check in 5 years (0=no, 1=yes)',
        'BMI': 'Body Mass Index',
        'Smoker': 'Smoked at least 100 cigarettes (0=no, 1=yes)',
        'Stroke': 'Ever had a stroke (0=no, 1=yes)',
        'HeartDiseaseorAttack': 'Coronary heart disease or MI (0=no, 1=yes)',
        'PhysActivity': 'Physical activity in past 30 days (0=no, 1=yes)',
        'Fruits': 'Consume fruit 1+ times per day (0=no, 1=yes)',
        'Veggies': 'Consume vegetables 1+ times per day (0=no, 1=yes)',
        'HvyAlcoholConsump': 'Heavy alcohol consumption (0=no, 1=yes)',
        'AnyHealthcare': 'Have any health care coverage (0=no, 1=yes)',
        'NoDocbcCost': 'Could not see doctor due to cost (0=no, 1=yes)',
        'GenHlth': 'General health (1=excellent to 5=poor)',
        'MentHlth': 'Days of poor mental health (past 30 days)',
        'PhysHlth': 'Days of poor physical health (past 30 days)',
        'DiffWalk': 'Difficulty walking or climbing stairs (0=no, 1=yes)',
        'Sex': 'Sex (0=female, 1=male)',
        'Age': 'Age category (1-13, binned)',
        'Education': 'Education level (1-6)',
        'Income': 'Income level (1-8)'
    }
    
    for col in df.columns[:-1]:  # Exclude target
        desc = feature_descriptions.get(col, 'Unknown')
        print(f"  {col:20s} - {desc}")
    
    print("\nFeature Statistics:")
    print(df.describe().round(2))
    
    print("\nSample Records (first 5):")
    print(df.head())
    
    print("\nClass Distribution:")
    print(f"  No Diabetes (0): {(df['target'] == 0).sum()} people ({(df['target'] == 0).sum() / len(df) * 100:.1f}%)")
    print(f"  Diabetes (1):    {(df['target'] == 1).sum()} people ({(df['target'] == 1).sum() / len(df) * 100:.1f}%)")
    
    # Separate features and target
    X = df.drop(columns=['target'])
    y = df['target']
    
    feature_names = X.columns.tolist()
    
    # Note: Most features are already binary or ordinal integers, no one-hot encoding needed
    print("\n" + "="*60)
    print("FEATURE TYPES")
    print("="*60)
    print("All features are already numeric (binary or ordinal)")
    print("No one-hot encoding required")
    
    # Train/test split using full dataset
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
