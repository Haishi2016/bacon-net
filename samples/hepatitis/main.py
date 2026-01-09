# Note: required to import baconNet from local folder
import sys
sys.path.insert(0, '../../')
sys.path.insert(0, '../')

import torch
import logging
from bacon.transformationLayer import IdentityTransformation, NegationTransformation, PeakTransformation, ValleyTransformation, StepUpTransformation, StepDownTransformation
from dataset import prepare_data, balance_data
from common import create_bacon_model, train_bacon_model, run_standard_analysis

logging.basicConfig(level=logging.INFO, format='%(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare data
X_train, Y_train, X_test, Y_test, feature_names = prepare_data(device)
num_features = len(feature_names)

# Balance training data by upsampling minority class
X_train, Y_train = balance_data(X_train, Y_train, device)

print(f"\n📊 Model will use {num_features} input features")

# Configure transformations
trans = [
    IdentityTransformation(1), 
    NegationTransformation(1),
    PeakTransformation(1),
    ValleyTransformation(1),
    StepUpTransformation(1),
    StepDownTransformation(1)
]

# Create model
bacon = create_bacon_model(
    input_size=num_features,
    aggregator='lsp.half_weight',
    weight_mode='trainable',
    transformations=trans,
    use_transformation_layer=True,
    weight_normalization='softmax',
    use_class_weighting=True,
    weight_penalty_strength=1e-4,
    permutation_initial_temperature=5.0,
    permutation_final_temperature=0.5,  # Fixed: was 4.0 (way too high - prevents proper convergence)
)

# Train model
train_bacon_model(
    bacon,
    X_train, Y_train, X_test, Y_test,
    attempts=10,
    acceptance_threshold=1.0,
    hierarchical_epochs_per_attempt=5000,  # Increased: more time for convergence with proper annealing
    hierarchical_group_size=8,
    frozen_training_epochs=500,  # Reduced: if frozen perm is bad, more training won't help
    binary_threshold=0.5
)

# Run standard analysis
run_standard_analysis(
    bacon,
    X_train, Y_train, X_test, Y_test,
    feature_names,
    title_prefix="Hepatitis",
    pruning_threshold=0.5
)
