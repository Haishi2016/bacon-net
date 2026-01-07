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

    print(f"Dataset shape before encoding: {X.shape}")
    print(f"Class distribution: {np.bincount(y_binary)}")

    # Identify multi-class categorical columns that need one-hot encoding
    # Binary columns (CAD, Hypothyroidism, etc.) are already 0/1 and don't need encoding
    categorical_columns = [
        'Comorbidity',  # 0,1,2,3 - multiple comorbidity levels
        'Hepatic Fat Accumulation (HFA)'  # 0,1,2,3,4 - fat accumulation scale
    ]
    
    # Verify which categorical columns exist
    categorical_columns = [col for col in categorical_columns if col in X.columns]
    
    print(f"\nMulti-class categorical columns to one-hot encode: {len(categorical_columns)}")
    for col in categorical_columns:
        print(f"  {col}: {X[col].nunique()} unique values")
    
    # One-hot encode only multi-class categorical columns
    X_encoded = pd.get_dummies(X, columns=categorical_columns, prefix=categorical_columns, drop_first=False)
    
    print(f"\nDataset shape after encoding: {X_encoded.shape}")
    print(f"Added {X_encoded.shape[1] - X.shape[1]} new binary features from one-hot encoding")

    df = pd.DataFrame(X_encoded)
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
        'Body Mass Index (BMI)': 'Body Mass Index',
        'Total Body Water (TBW)': 'Total body water (L)',
        'Extracellular Water (ECW)': 'Extracellular water (L)',
        'Intracellular Water (ICW)': 'Intracellular water (L)',
        'Extracellular Fluid/Total Body Water (ECF/TBW)': 'ECF/TBW ratio',
        'Total Body Fat Ratio (TBFR) (%)': 'Total body fat ratio (%)',
        'Lean Mass (LM) (%)': 'Lean mass (%)',
        'Body Protein Content (Protein) (%)': 'Body protein content (%)',
        'Visceral Fat Rating (VFR)': 'Visceral fat rating',
        'Bone Mass (BM)': 'Bone mass',
        'Muscle Mass (MM)': 'Muscle mass (kg)',
        'Obesity (%)': 'Obesity percentage',
        'Total Fat Content (TFC)': 'Total fat content',
        'Visceral Fat Area (VFA)': 'Visceral fat area (cm²)',
        'Visceral Muscle Area (VMA) (Kg)': 'Visceral muscle area (kg)',
        'Glucose': 'Blood glucose (mg/dL)',
        'Total Cholesterol (TC)': 'Total cholesterol (mg/dL)',
        'High Density Lipoprotein (HDL)': 'HDL cholesterol (mg/dL)',
        'Low Density Lipoprotein (LDL)': 'LDL cholesterol (mg/dL)',
        'Triglyceride': 'Triglycerides (mg/dL)',
        'Aspartat Aminotransferaz (AST)': 'AST enzyme (U/L)',
        'Alanin Aminotransferaz (ALT)': 'ALT enzyme (U/L)',
        'Alkaline Phosphatase (ALP)': 'ALP enzyme (U/L)',
        'Creatinine': 'Creatinine (mg/dL)',
        'Glomerular Filtration Rate (GFR)': 'GFR',
        'C-Reactive Protein (CRP)': 'CRP (mg/L)',
        'Hemoglobin (HGB)': 'Hemoglobin (g/dL)',
        'Vitamin D': 'Vitamin D (ng/mL)',
        'Comorbidity': 'Number of comorbidities',
        'Coronary Artery Disease (CAD)': 'CAD status (0=no, 1=yes)',
        'Hypothyroidism': 'Hypothyroidism status (0=no, 1=yes)',
        'Hyperlipidemia': 'Hyperlipidemia status (0=no, 1=yes)',
        'Diabetes Mellitus (DM)': 'Diabetes status (0=no, 1=yes)',
        'Hepatic Fat Accumulation (HFA)': 'HFA level (0-4 scale)'
    }

    for col in df.columns[:-1]:  # Exclude target
        # For one-hot encoded columns, show the original column description
        base_col = col
        for cat_col in categorical_columns:
            if col.startswith(cat_col + '_'):
                base_col = cat_col
                break
        desc = feature_descriptions.get(base_col, 'Bioimpedance/Laboratory measure')
        if col.startswith(tuple([c + '_' for c in categorical_columns])):
            # This is a one-hot encoded column
            print(f"  {col:50s} - Binary indicator")
        else:
            print(f"  {col:50s} - {desc}")

    print("\nFeature Statistics (first 10 columns):")
    print(df.iloc[:, :10].describe().round(2))

    print("\nSample Records (first 5 rows, first 10 columns):")
    print(df.iloc[:5, :10])

    print("\nClass Distribution:")
    print(f"  No Gallstone (0): {(df['target'] == 0).sum()} people ({(df['target'] == 0).sum() / len(df) * 100:.1f}%)")
    print(f"  Gallstone (1):    {(df['target'] == 1).sum()} people ({(df['target'] == 1).sum() / len(df) * 100:.1f}%)")

    # Separate features and target
    X = df.drop(columns=['target'])
    y = df['target']

    feature_names = X.columns.tolist()

    # Feature type summary
    print("\n" + "="*60)
    print("FEATURE TYPES")
    print("="*60)
    print(f"Total features: {len(feature_names)}")
    print(f"  Continuous features: {X_encoded.shape[1] - sum([col.startswith(tuple([c + '_' for c in categorical_columns])) for col in X.columns])}")
    print(f"  One-hot encoded binary features: {sum([col.startswith(tuple([c + '_' for c in categorical_columns])) for col in X.columns])}")
    print(f"    (from {len(categorical_columns)} categorical columns)")

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
