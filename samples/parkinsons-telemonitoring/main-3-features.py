# Note: required to import baconNet from local folder
import sys
sys.path.insert(0, '../../')
sys.path.insert(0, '../')

import torch
import numpy as np
import logging
from bacon.transformationLayer import IdentityTransformation, NegationTransformation, PeakTransformation, ValleyTransformation, StepUpTransformation, StepDownTransformation
from dataset import prepare_data, balance_data
from common import create_bacon_model, train_bacon_model, run_standard_analysis

logging.basicConfig(level=logging.INFO, format='%(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare data (using motor_UPDRS as target)
X_train, Y_train, X_test, Y_test, feature_names, _ = prepare_data(device, target='motor_UPDRS')

# Select only Jitter, Shimmer and Age features
# Feature names from dataset: age, sex, test_time, Jitter(%), Jitter(Abs), Jitter:RAP, Jitter:PPQ5, Jitter:DDP,
#                             Shimmer, Shimmer(dB), Shimmer:APQ3, Shimmer:APQ5, Shimmer:APQ11, Shimmer:DDA,
#                             NHR, HNR, RPDE, DFA, PPE
selected_features = ['age', 'Jitter(%)', 'Shimmer']
selected_indices = [i for i, name in enumerate(feature_names) if name in selected_features]

print(f"\n📊 Original features: {len(feature_names)}")
print(f"📊 Selected features: {selected_features}")
print(f"📊 Selected indices: {selected_indices}")

# Filter to selected features only
X_train = X_train[:, selected_indices]
X_test = X_test[:, selected_indices]
feature_names = [feature_names[i] for i in selected_indices]
num_features = len(feature_names)

print(f"📊 Model will use {num_features} input features: {feature_names}")

# Balance training data by upsampling minority class
X_train, Y_train = balance_data(X_train, Y_train, device)

# Configure transformations
trans = [
    IdentityTransformation(1), 
    NegationTransformation(1),
    PeakTransformation(1),
    ValleyTransformation(1),
    StepUpTransformation(1),
    StepDownTransformation(1)
]

# Create model with custom configuration
bacon = create_bacon_model(
    input_size=num_features,
    aggregator='lsp.half_weight',
    weight_mode='trainable',
    transformations=trans,
    use_transformation_layer=True,
    weight_normalization='softmax',
    use_class_weighting=True,
    loss_amplifier=1000,
    permutation_initial_temperature=5.0,
    permutation_final_temperature=0.5,
    weight_penalty_strength=1e-4
)

# Train model
train_bacon_model(
    bacon,
    X_train, Y_train, X_test, Y_test,
    attempts=15,
    acceptance_threshold=0.5,
    use_hierarchical_permutation=False,  # Not needed for only 3 features
    hierarchical_bleed_ratio=0.5,
    hierarchical_group_size=3,
    binary_threshold=0.5,
    save_path='./assembler-3-features.pth'
)

# Run standard analysis (use test data for both val and test since no separate val set)
run_standard_analysis(
    bacon,
    X_train, Y_train,  # Training data
    X_test, Y_test,    # Validation (using test data)
    X_test, Y_test,    # Test data
    feature_names,
    pruning_threshold=0.5
)
