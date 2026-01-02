# Note: required to import baconNet from local folder
import sys
sys.path.insert(0, '../../')
sys.path.insert(0, '../')  # For common module

import torch
import logging
from dataset import prepare_data
from common import create_bacon_model, train_bacon_model, run_standard_analysis

logging.basicConfig(level=logging.INFO, format='%(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare data
X_train, Y_train, X_test, Y_test, feature_names = prepare_data(device)

# Model configuration
num_features = len(feature_names)
print(f"\n📊 Model will use {num_features} input features (after one-hot encoding)")

# Create model with standard configuration
bacon = create_bacon_model(
    input_size=num_features,
    aggregator='lsp.full_weight',
    weight_mode='fixed',
    use_transformation_layer=True,
    weight_normalization='softmax',
    use_class_weighting=True,
    permutation_initial_temperature=5.0,
    permutation_final_temperature=4.0,
    weight_penalty_strength=1e-3
)

# Train model
train_bacon_model(
    bacon,
    X_train, Y_train, X_test, Y_test,
    attempts=10,
    acceptance_threshold=0.75,
    hierarchical_epochs_per_attempt=3000,
    hierarchical_group_size=8
)

# Run standard analysis pipeline
run_standard_analysis(
    bacon,
    X_train, Y_train, X_test, Y_test,
    feature_names,
    title_prefix="Heart Disease",
    device=device
)
