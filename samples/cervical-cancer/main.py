# Note: required to import baconNet from local folder
import sys
sys.path.insert(0, '../../')
sys.path.insert(0, '../')

import torch
import logging
from bacon.transformationLayer import IdentityTransformation, NegationTransformation
from dataset import prepare_data
from common import create_bacon_model, train_bacon_model, run_standard_analysis

logging.basicConfig(level=logging.INFO, format='%(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare data
X_train, Y_train, X_test, Y_test, feature_names = prepare_data(device)
num_features = len(feature_names)

print(f"\n📊 Model will use {num_features} input features")

# Configure transformations
trans = [
    IdentityTransformation(1), 
    NegationTransformation(1)
]

# Create model with custom configuration (adapted for small dataset)
bacon = create_bacon_model(
    input_size=num_features,
    aggregator='lsp.half_weight',
    weight_mode='fixed',
    transformations=trans,
    use_transformation_layer=True,
    weight_normalization='softmax',
    use_class_weighting=True,
    freeze_loss_threshold=0.05,
    loss_amplifier=1000,
    permutation_initial_temperature=5.0,
    permutation_final_temperature=0.5,
    weight_penalty_strength=1e-3
)

# Train model (adapted for small dataset - fewer attempts, shorter training)
train_bacon_model(
    bacon,
    X_train, Y_train, X_test, Y_test,
    attempts=20,
    acceptance_threshold=0.88,
    use_hierarchical_permutation=True,
    hierarchical_bleed_ratio=0.5,
    hierarchical_epochs_per_attempt=3000,
    hierarchical_group_size=8,
    loss_weight_perm_sparsity=5.0,
    sinkhorn_iters=150,
    freeze_confidence_threshold=0.92,
    freeze_min_confidence=0.80,
    freeze_loss_threshold=0.08,
    frozen_training_epochs=1000,
    max_epochs=4000
)

# Run standard analysis
run_standard_analysis(
    bacon,
    X_train, Y_train, X_test, Y_test,
    feature_names
)

