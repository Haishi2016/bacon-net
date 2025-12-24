from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
import torch
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '../../')
from bacon.utils import SigmoidScaler


def _prepare_data_common():
    """Common data preparation logic (fetching, cleaning, encoding).
    
    Returns:
        tuple: (X_train_np, X_test_np, y_train_np, y_test_np, feature_names)
            All arrays are numpy arrays, ready for scaling
    """
    # Fetch heart disease dataset (Cleveland database)
    heart_disease = fetch_ucirepo(id=45)

    # Extract features and target
    X = heart_disease.data.features
    y = heart_disease.data.targets

    # The target 'num' is 0-4, convert to binary: 0 = no disease, 1-4 = disease present
    y_binary = (y['num'] > 0).astype(int).values

    # Handle missing values (marked as '?')
    X = X.replace('?', np.nan)
    X = X.apply(pd.to_numeric, errors='coerce')

    # Drop rows with missing values
    valid_indices = ~X.isnull().any(axis=1)
    X = X[valid_indices]
    y_binary = y_binary[valid_indices]

    print(f"Dataset shape after removing missing values: {X.shape}")
    print(f"Class distribution: {np.bincount(y_binary)}")

    df = pd.DataFrame(X)
    df['target'] = y_binary

    # Dataset Preview
    print("\n" + "="*60)
    print("DATASET PREVIEW")
    print("="*60)

    print("\nFeature Names and Descriptions:")
    feature_descriptions = {
        'age': 'Age in years',
        'sex': 'Sex (1=male, 0=female)',
        'cp': 'Chest pain type (1-4)',
        'trestbps': 'Resting blood pressure (mm Hg)',
        'chol': 'Serum cholesterol (mg/dl)',
        'fbs': 'Fasting blood sugar > 120 mg/dl (1=true)',
        'restecg': 'Resting ECG results (0-2)',
        'thalach': 'Maximum heart rate achieved',
        'exang': 'Exercise induced angina (1=yes)',
        'oldpeak': 'ST depression induced by exercise',
        'slope': 'Slope of peak exercise ST segment (1-3)',
        'ca': 'Number of major vessels (0-3)',
        'thal': 'Thalassemia (3=normal, 6=fixed, 7=reversible)'
    }

    for col in df.columns[:-1]:  # Exclude target
        desc = feature_descriptions.get(col, 'Unknown')
        print(f"  {col:12s} - {desc}")

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

    # One-hot encode categorical features
    print("\n" + "="*60)
    print("ENCODING CATEGORICAL FEATURES")
    print("="*60)

    categorical_features = ['cp', 'restecg', 'slope', 'thal']
    print(f"\nCategorical features to encode: {categorical_features}")

    # Check unique values for each categorical feature
    for col in categorical_features:
        if col in X.columns:
            unique_vals = sorted(X[col].unique())
            print(f"  {col}: {unique_vals}")

    # One-hot encode
    X_encoded = pd.get_dummies(X, columns=categorical_features, prefix=categorical_features, drop_first=False)

    print(f"\nOriginal features: {len(X.columns)}")
    print(f"After one-hot encoding: {len(X_encoded.columns)}")
    print(f"New feature names: {X_encoded.columns.tolist()}")

    feature_names = X_encoded.columns.tolist()
    X = X_encoded

    # Train/test split
    X_train_df, X_test_df, y_train_np, y_test_np = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Convert DataFrames to numpy arrays for scaling
    X_train_np = X_train_df.values.astype(np.float64)
    X_test_np = X_test_df.values.astype(np.float64)
    y_train_np = y_train_np.to_numpy()
    y_test_np = y_test_np.to_numpy()

    print(f"\nTrain data shape: {X_train_np.shape}, dtype: {X_train_np.dtype}")
    print(f"Test data shape: {X_test_np.shape}, dtype: {X_test_np.dtype}")

    return X_train_np, X_test_np, y_train_np, y_test_np, feature_names


def prepare_data_sklearn():
    """Prepare heart disease dataset for sklearn models (returns numpy arrays).
    
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
    """Prepare heart disease dataset for PyTorch models (returns tensors).
    
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