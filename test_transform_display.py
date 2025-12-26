"""Test script to verify transformation display with parameters"""
import sys
sys.path.insert(0, './')

import torch
from bacon.binaryTreeLogicNet import binaryTreeLogicNet
from bacon.transformationLayer import (
    IdentityTransformation, NegationTransformation, 
    PeakTransformation, ValleyTransformation,
    StepUpTransformation, StepDownTransformation
)
from bacon.visualization import print_tree_structure
from bacon.utils import export_tree_structure_to_json
import json

# Create a simple model with various transformations
device = torch.device("cpu")

trans = [
    IdentityTransformation(1), 
    NegationTransformation(1), 
    PeakTransformation(1),
    ValleyTransformation(1),
    StepUpTransformation(1),
    StepDownTransformation(1)
]

model = binaryTreeLogicNet(
    input_size=6,
    aggregator='lsp.half_weight',
    use_transformation_layer=True,
    transformations=trans,
    weight_mode='fixed',
    device=device
)

feature_names = ['Age', 'Income', 'BMI', 'Debt', 'Credit', 'Savings']

print("\n" + "="*80)
print("TESTING TRANSFORMATION DISPLAY")
print("="*80)

# Manually set some transformation selections to test display
if hasattr(model, 'transformation_layer'):
    tl = model.transformation_layer
    # Set different transformations for each feature
    tl.transform_selectors.data = torch.tensor([
        [10.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # Feature 0: Identity
        [0.0, 10.0, 0.0, 0.0, 0.0, 0.0],  # Feature 1: Negation
        [0.0, 0.0, 10.0, 0.0, 0.0, 0.0],  # Feature 2: Peak
        [0.0, 0.0, 0.0, 10.0, 0.0, 0.0],  # Feature 3: Valley
        [0.0, 0.0, 0.0, 0.0, 10.0, 0.0],  # Feature 4: Step Up
        [0.0, 0.0, 0.0, 0.0, 0.0, 10.0],  # Feature 5: Step Down
    ], device=device)
    
    # Set some example learned parameters
    # Peak location for feature 2
    tl.transform_params[2]['peak_loc'].data[2] = 0.3  # Will be sigmoidized
    # Valley location for feature 3
    tl.transform_params[3]['valley_loc'].data[3] = -0.5  # Will be sigmoidized
    # Threshold for feature 4
    tl.transform_params[4]['threshold'].data[4] = 1.0  # Will be sigmoidized and scaled
    # Threshold for feature 5
    tl.transform_params[5]['threshold'].data[5] = -1.0  # Will be sigmoidized and scaled

print("\n📊 Tree Structure (should show transformation types and parameters):")
print_tree_structure(model, feature_names)

print("\n📄 JSON Export (should include transformation_params):")
json_data = export_tree_structure_to_json(model, feature_names)
print(json.dumps(json_data['features'], indent=2))

print("\n✅ Test complete!")
