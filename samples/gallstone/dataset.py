from sklearn.model_selection import train_test_split
import torch
import pandas as pd
import numpy as np
import sys
sys.path.insert(0, '../../')
from bacon.utils import SigmoidScaler


def prepare_data(device):
    """Prepare gallstone dataset for training.
    
    Args:
        device: torch.device to place tensors on
        
    Returns:
        tuple: (X_train, Y_train, X_test, Y_test, feature_names)
    """
    # Load Gallstone dataset from local CSV
    print("Loading Gallstone dataset...")
    df = pd.read_csv('c:/School/lsp/dataset-uci.csv')

    # Target is first column 'Gallstone Status': 0 = no gallstone, 1 = gallstone disease
    y_binary = df['Gallstone Status'].values

    # Features are all other columns
    X = df.drop(columns=['Gallstone Status'])

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
        'Age': 'Age in years',
        'Gender': 'Gender (0=female, 1=male)',
        'Height': 'Height in cm',
        'Weight': 'Weight in kg',
        'BMI': 'Body Mass Index',
        'TBW': 'Total body water (L)',
        'ECW': 'Extracellular water (L)',
        'ICW': 'Intracellular water (L)',
        'Muscle_mass': 'Muscle mass (kg)',
        'Fat_mass': 'Fat mass (kg)',
        'Protein': 'Protein (kg)',
        'VFA': 'Visceral fat area (cm²)',
        'Hepatic_fat': 'Hepatic fat (%)',
        'Glucose': 'Blood glucose (mg/dL)',
        'Total_cholesterol': 'Total cholesterol (mg/dL)',
        'HDL': 'High-density lipoprotein (mg/dL)',
        'LDL': 'Low-density lipoprotein (mg/dL)',
        'Triglycerides': 'Triglycerides (mg/dL)',
        'AST': 'Aspartate aminotransferase (U/L)',
        'ALT': 'Alanine aminotransferase (U/L)',
        'ALP': 'Alkaline phosphatase (U/L)',
        'Creatinine': 'Creatinine (mg/dL)',
        'GFR': 'Glomerular filtration rate',
        'CRP': 'C-reactive protein (mg/L)',
        'Hemoglobin': 'Hemoglobin (g/dL)',
        'Vitamin_D': 'Vitamin D (ng/mL)'
    }

    for col in df.columns[:-1]:  # Exclude target
        desc = feature_descriptions.get(col, 'Bioimpedance/Laboratory measure')
        print(f"  {col:20s} - {desc}")

    print("\nFeature Statistics:")
    print(df.describe().round(2))

    print("\nSample Records (first 5):")
    print(df.head())

    print("\nClass Distribution:")
    print(f"  No Gallstone (0): {(df['target'] == 0).sum()} people ({(df['target'] == 0).sum() / len(df) * 100:.1f}%)")
    print(f"  Gallstone (1):    {(df['target'] == 1).sum()} people ({(df['target'] == 1).sum() / len(df) * 100:.1f}%)")

    # Separate features and target
    X = df.drop(columns=['target'])
    y = df['target']

    feature_names = X.columns.tolist()

    # Note: All features are already numeric, no one-hot encoding needed
    print("\n" + "="*60)
    print("FEATURE TYPES")
    print("="*60)
    print("All features are already numeric (continuous)")
    print("No one-hot encoding required")

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
