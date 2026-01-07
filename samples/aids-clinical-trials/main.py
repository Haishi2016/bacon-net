# Note: required to import baconNet from local folder
import sys
sys.path.insert(0, '../../')
sys.path.insert(0, '../')

import torch
import logging
from bacon.transformationLayer import IdentityTransformation, NegationTransformation
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
    NegationTransformation(1)
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
)

# Train model
train_bacon_model(
    bacon,
    X_train, Y_train, X_test, Y_test,
    attempts=10,
    acceptance_threshold=0.8,
     binary_threshold=0.234
)

# Run standard analysis
run_standard_analysis(
    bacon,
    X_train, Y_train, X_test, Y_test,
    feature_names,
    title_prefix="AIDS Clinical Trials",
    pruning_threshold=0.234
)
