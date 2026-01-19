"""Parkinson's Telemonitoring - Subject-Averaged Binary Classification

This version averages all measurements per subject before classification.
This approach:
1. Reduces temporal noise within subjects
2. Creates one sample per subject (better independence)
3. Uses each subject's average UPDRS as their severity level
4. Better suited for binary classification than per-measurement prediction
"""

# Note: required to import baconNet from local folder
import sys
sys.path.insert(0, '../../')
sys.path.insert(0, '../')

import torch
import logging
import numpy as np
import pandas as pd
from bacon.baconNet import baconNet
from bacon.transformationLayer import IdentityTransformation, NegationTransformation
from bacon.visualization import visualize_tree_structure, print_tree_structure
from bacon.utils import SigmoidScaler
from sklearn.model_selection import train_test_split
from common import create_bacon_model, train_bacon_model, run_standard_analysis

logging.basicConfig(level=logging.INFO, format='%(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and prepare subject-averaged data
print("Loading Parkinson's Telemonitoring Dataset...")
data_path = r"c:\School\uci\parkinson\parkinsons_updrs.data"
df = pd.read_csv(data_path)

print(f"Original data: {len(df)} measurements from {df['subject#'].nunique()} subjects")

# Average all measurements per subject
print("\nAveraging measurements per subject...")
subject_groups = df.groupby('subject#')

# Average features (excluding subject#, motor_UPDRS, total_UPDRS)
feature_cols = [col for col in df.columns if col not in ['subject#', 'motor_UPDRS', 'total_UPDRS']]
X_avg = subject_groups[feature_cols].mean()

# Use average motor_UPDRS as the target for each subject
y_avg = subject_groups['motor_UPDRS'].mean()

print(f"Averaged data: {len(X_avg)} subjects")
print(f"Features: {list(X_avg.columns)}")

# Convert to binary classification using 75th percentile
threshold_75 = np.percentile(y_avg.values, 75)
y_binary = (y_avg.values >= threshold_75).astype(int)

print(f"\n75th percentile threshold: {threshold_75:.2f} UPDRS")
print(f"Class distribution:")
print(f"  Low severity (0):  {np.sum(y_binary == 0)} subjects ({np.sum(y_binary == 0) / len(y_binary) * 100:.1f}%)")
print(f"  High severity (1): {np.sum(y_binary == 1)} subjects ({np.sum(y_binary == 1) / len(y_binary) * 100:.1f}%)")

# Train/test split (stratified by class)
X_train, X_test, y_train, y_test = train_test_split(
    X_avg.values, y_binary, 
    test_size=0.2, 
    random_state=42, 
    stratify=y_binary
)

print(f"\nTrain set: {len(X_train)} subjects")
print(f"Test set:  {len(X_test)} subjects")

# Normalize features using SigmoidScaler
scaler = SigmoidScaler(alpha=4, beta=-1)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
Y_train = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32).to(device)
X_test = torch.tensor(X_test, dtype=torch.float32).to(device)
Y_test = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32).to(device)

feature_names = list(X_avg.columns)
num_features = len(feature_names)

print(f"\n📊 Model will use {num_features} input features")
print(f"🎯 Training samples: {len(X_train)}")
print(f"🎯 Test samples: {len(X_test)}")

# Configure transformations
trans = [
    IdentityTransformation(1), 
    NegationTransformation(1)
]

# Create model with standard configuration
bacon = create_bacon_model(
    input_size=num_features,
    aggregator='lsp.half_weight',
    weight_mode='trainable',
    use_transformation_layer=True,
    weight_normalization='softmax',
    use_class_weighting=True,
    permutation_initial_temperature=5.0,
    permutation_final_temperature=0.5,
    weight_penalty_strength=1e-4
)

# Train model
train_bacon_model(
    bacon,
    X_train, Y_train, X_test, Y_test,
    attempts=10,
    acceptance_threshold=1.0,
    hierarchical_epochs_per_attempt=3000,
    hierarchical_group_size=8,
     binary_threshold=0.5
)

# Run standard analysis pipeline
run_standard_analysis(
    bacon,
    X_train, Y_train, X_test, Y_test,
    feature_names,
    title_prefix="Parkinson's Telemonitoring (Subject-Averaged)",
    device=device,
    pruning_threshold=0.5
)

