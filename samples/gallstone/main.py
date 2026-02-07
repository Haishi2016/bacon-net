# Note: required to import baconNet from local folder
import sys
sys.path.insert(0, '../../')
sys.path.insert(0, '../')  # For common module

import torch
import argparse
import logging
from dataset import prepare_data, balance_data
from common import create_bacon_model, train_bacon_model, run_standard_analysis
from bacon.transformationLayer import (
    IdentityTransformation, NegationTransformation, 
    PeakTransformation, ValleyTransformation,
    StepUpTransformation, StepDownTransformation
)
from bacon.policies import FixedAndnessPolicy

logging.basicConfig(level=logging.INFO, format='%(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Configuration: choose aggregator type
USE_SOFTMAX_LSP = False  # Set to False for traditional lsp.half_weight

# Policy configuration
# NOTE: Andness regularization is disabled. Strong AND (1.0) or neutral (0.5) 
# targets cause degenerate solutions. Let the model learn freely.
USE_FIXED_ANDNESS = True  # Disabled - causes 50% accuracy trap
FIXED_ANDNESS_VALUE = 0.5  # AND-like (min operator)
ANDNESS_PENALTY_WEIGHT = 0.01  # Not used when disabled

parser = argparse.ArgumentParser()
parser.add_argument("--loss-trim-percentile", type=float, default=0.0)
parser.add_argument("--loss-trim-mode", type=str, default="none", choices=["none", "drop_high", "drop_low"])
parser.add_argument("--loss-trim-start-epoch", type=int, default=0)
args = parser.parse_args()

# Prepare data
X_train, Y_train, X_test, Y_test, feature_names = prepare_data(device)

# Balance training data by upsampling minority class
X_train, Y_train = balance_data(X_train, Y_train, device)

# Model configuration
num_features = len(feature_names)
aggregator_type = 'lsp.softmax' if USE_SOFTMAX_LSP else 'lsp.half_weight'
print(f"\n📊 Model will use {num_features} input features")
print(f"📊 Aggregator: {aggregator_type}")

# Training policy (optional)
training_policy = None
if USE_FIXED_ANDNESS:
    training_policy = FixedAndnessPolicy(andness=FIXED_ANDNESS_VALUE, penalty_weight=ANDNESS_PENALTY_WEIGHT)
    print(f"📊 Policy: Andness regularization toward {FIXED_ANDNESS_VALUE} (penalty={ANDNESS_PENALTY_WEIGHT})")

# Custom transformations for gallstone
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
    tree_layout="full",
    aggregator=aggregator_type,
    weight_mode='trainable',
    transformations=trans,
    use_transformation_layer=True,
    weight_normalization='softmax',
    use_class_weighting=True,
    permutation_initial_temperature=5.0,
    permutation_final_temperature=4.0,
    weight_penalty_strength=1e-4,
    training_policy=training_policy,
    loss_trim_percentile=args.loss_trim_percentile,
    loss_trim_mode=args.loss_trim_mode,
    loss_trim_start_epoch=args.loss_trim_start_epoch,
     # Full tree settings
    full_tree_temperature=1.0,
    full_tree_final_temperature=0.5,
    full_tree_max_egress=None,  # No egress constraint for simple test
    loss_weight_full_tree_egress=0.5,
)

# Train model
train_bacon_model(
    bacon,
    X_train, Y_train, X_test, Y_test,
    attempts=10,
    acceptance_threshold=1.0,
    hierarchical_epochs_per_attempt=4000,
    hierarchical_group_size=15,
    frozen_training_epochs=2000,
    binary_threshold=0.5
)

# Run standard analysis pipeline (using test data for both val and test)
run_standard_analysis(
    bacon,
    X_train, Y_train,  # Training data
    X_test, Y_test,    # Validation (using test data)
    X_test, Y_test,    # Test data
    feature_names,
    title_prefix="Gallstone",
    device=device,
    pruning_threshold=0.5
)