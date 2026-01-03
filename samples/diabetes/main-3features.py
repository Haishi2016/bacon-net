# Note: required to import baconNet from local folder
import sys
sys.path.insert(0, '../../')
sys.path.insert(0, '../')  # For common module

import torch
import logging
from dataset import prepare_data
from common import create_bacon_model, train_bacon_model, run_standard_analysis
from bacon.transformationLayer import IdentityTransformation, NegationTransformation, PeakTransformation

logging.basicConfig(level=logging.INFO, format='%(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare full data
X_train_full, Y_train, X_test_full, Y_test, feature_names_full = prepare_data(device)

# Select only 3 features: Stroke, NoDocbcCost, and HighChol
selected_features = ['Stroke', 'NoDocbcCost', 'HighChol']
selected_indices = [feature_names_full.index(feat) for feat in selected_features]

print(f"\n📊 Using only {len(selected_features)} features: {selected_features}")
print(f"Feature indices: {selected_indices}")

# Extract selected features
X_train = X_train_full[:, selected_indices]
X_test = X_test_full[:, selected_indices]
feature_names = selected_features

# Apply 1-x transformation to Stroke and NoDocbcCost
# Stroke is at index 0, NoDocbcCost is at index 1
print("\n📊 Applying 1-x transformation to Stroke and NoDocbcCost...")
X_train[:, 0] = 1 - X_train[:, 0]  # Stroke
X_train[:, 1] = 1 - X_train[:, 1]  # NoDocbcCost
X_test[:, 0] = 1 - X_test[:, 0]    # Stroke
X_test[:, 1] = 1 - X_test[:, 1]    # NoDocbcCost

# Model configuration
num_features = len(feature_names)
print(f"📊 Model will use {num_features} input features")

# Custom transformations for diabetes
trans = [
    IdentityTransformation(1), 
    NegationTransformation(1), 
    PeakTransformation(1)
]

# Create model with custom configuration
bacon = create_bacon_model(
    input_size=num_features,
    aggregator='lsp.full_weight',
    weight_mode='fixed',
    transformations=trans,
    use_transformation_layer=False,
    weight_normalization='softmax',
    use_class_weighting=True,
    permutation_initial_temperature=5.0,
    permutation_final_temperature=4.0
)

# Train model
train_bacon_model(
    bacon,
    X_train, Y_train, X_test, Y_test,
    attempts=10,    
    acceptance_threshold=0.75,
    hierarchical_epochs_per_attempt=2000,
    hierarchical_group_size=1
)

# Run standard analysis pipeline
run_standard_analysis(
    bacon,
    X_train, Y_train, X_test, Y_test,
    feature_names,
    title_prefix="Diabetes (3 Features)",
    device=device,
    pruning_threshold=0.5
)
