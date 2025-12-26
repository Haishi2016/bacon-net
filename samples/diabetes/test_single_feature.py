"""Test single feature prediction accuracy for diabetes dataset"""
import sys
sys.path.insert(0, '../../')

import torch
import numpy as np
import json
from dataset import prepare_data

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare data
X_train, Y_train, X_test, Y_test, feature_names = prepare_data(device)

# Load trained model structure from JSON
print("Loading model structure from JSON...")
try:
    with open('diabetes_tree_structure.json', 'r') as f:
        tree_structure = json.load(f)
    print("Tree structure loaded successfully")
except FileNotFoundError:
    print("Warning: diabetes_tree_structure.json not found. Run main.py first to train and export model.")
    tree_structure = None

print("\n" + "="*60)
print("SINGLE FEATURE PREDICTION TEST")
print("="*60)

# Test HighBP alone
highbp_idx = feature_names.index('HighBP')
print(f"\nFeature: HighBP (index {highbp_idx})")

highbp_test = X_test[:, highbp_idx]
predictions = (highbp_test > 0.5).float()
accuracy = (predictions == Y_test.squeeze()).float().mean().item()

print(f"Test Set Results:")
print(f"  Total samples: {len(Y_test)}")
print(f"  Accuracy with threshold=0.5: {accuracy * 100:.2f}%")

# Test HighChol alone
highchol_idx = feature_names.index('HighChol')
print(f"\nFeature: HighChol (index {highchol_idx})")

highchol_test = X_test[:, highchol_idx]
predictions_chol = (highchol_test > 0.5).float()
accuracy_chol = (predictions_chol == Y_test.squeeze()).float().mean().item()

print(f"Test Set Results:")
print(f"  Total samples: {len(Y_test)}")
print(f"  Accuracy with threshold=0.5: {accuracy_chol * 100:.2f}%")

# Test with learned baseline + HighChol aggregation
if tree_structure is not None:
    print("\n" + "="*60)
    print("BASELINE + HighChol TEST")
    print("="*60)
    
    # Get feature order from JSON
    locked_perm = tree_structure['locked_perm']
    
    
    highchol_idx = 8

    # Find HighChol position in the tree
    highchol_position = locked_perm.index(highchol_idx)
    print(f"\nHighChol is at position {highchol_position} in the tree")
    
    # Get baseline features (first 2 in tree)
    baseline_feat_names = [tree_structure['features'][i]['display_name'] for i in [0, 1]]
    print(f"Baseline features: {baseline_feat_names}")
    
    # Get the aggregator parameters for combining baseline
    node0 = tree_structure['nodes'][0]
    agg0_andness = node0['andness']
    agg0_weights = [node0['weights']['left'], node0['weights']['right']]
    
    print(f"  Aggregator 0: weights={agg0_weights}, andness={agg0_andness:.4f}")
    
    # Compute baseline aggregation WITH TRANSFORMATIONS
    from bacon.aggregators.lsp.half_weight import HalfWeightAggregator
    aggregator = HalfWeightAggregator()
    
    baseline_feat_indices = [locked_perm[0], locked_perm[1]]
    
    # Apply transformations to baseline features
    baseline_left = X_test[:, baseline_feat_indices[0]]
    if tree_structure['features'][0]['transformation'] == 'negation':
        baseline_left = 1.0 - baseline_left
    elif tree_structure['features'][0]['transformation'] == 'peak':
        if 'transformation_params' in tree_structure['features'][0]:
            peak_loc = float(tree_structure['features'][0]['transformation_params']['peak_location'])
            baseline_left = 1.0 - torch.abs(baseline_left - peak_loc)
    
    baseline_right = X_test[:, baseline_feat_indices[1]]
    if tree_structure['features'][1]['transformation'] == 'negation':
        baseline_right = 1.0 - baseline_right
    elif tree_structure['features'][1]['transformation'] == 'peak':
        if 'transformation_params' in tree_structure['features'][1]:
            peak_loc = float(tree_structure['features'][1]['transformation_params']['peak_location'])
            baseline_right = 1.0 - torch.abs(baseline_right - peak_loc)
    
    baseline_result = aggregator.aggregate_tensor(
        baseline_left, baseline_right, 
        torch.tensor(agg0_andness, device=device),
        agg0_weights[0], agg0_weights[1]
    )
    
    # Now aggregate baseline with HighChol
    print(f"\nAggregating baseline with HighChol (position {highchol_position})...")
    
    # Aggregate all features from position 2 to highchol_position
    current_result = baseline_result
    
    for pos in range(2, highchol_position + 1):
        agg_idx = pos - 1
        node = tree_structure['nodes'][agg_idx]
        andness = node['andness']
        weights = [node['weights']['left'], node['weights']['right']]
        
        feat_idx = locked_perm[pos]
        feature_values = X_test[:, feat_idx]
        
        # Apply transformation
        transformation = tree_structure['features'][pos]['transformation']
        if transformation == 'negation':
            feature_values = 1.0 - feature_values
        elif transformation == 'peak' and 'transformation_params' in tree_structure['features'][pos]:
            peak_loc = float(tree_structure['features'][pos]['transformation_params']['peak_location'])
            feature_values = 1.0 - torch.abs(feature_values - peak_loc)
        
        current_result = aggregator.aggregate_tensor(
            current_result, feature_values,
            torch.tensor(andness, device=device),
            weights[0], weights[1]
        )
    
    # Test with 0.5 threshold
    combined_predictions = (current_result > 0.5).float()
    combined_accuracy = (combined_predictions == Y_test.squeeze()).float().mean().item()
    
    print(f"\nResults (threshold=0.5):")
    print(f"  Accuracy: {combined_accuracy * 100:.2f}%")
    
    print(f"\nComparison:")
    print(f"  HighChol alone:          {accuracy_chol * 100:.2f}%")
    print(f"  Baseline + HighChol:     {combined_accuracy * 100:.2f}%")
    print(f"  Improvement:             {(combined_accuracy - accuracy_chol) * 100:+.2f}%")
    
    # Now test baseline + HighChol directly (skip middle features)
    print("\n" + "="*60)
    print("BASELINE + HighChol ONLY (skip middle features)")
    print("="*60)
    
    # Aggregate baseline directly with HighChol using the aggregator at HighChol's position
    agg_idx_highchol = highchol_position - 1
    node_highchol = tree_structure['nodes'][agg_idx_highchol]
    andness_highchol = node_highchol['andness']
    weights_highchol = [node_highchol['weights']['left'], node_highchol['weights']['right']]
    
    print(f"Using aggregator {agg_idx_highchol} for HighChol at position {highchol_position}")
    print(f"  Weights: {weights_highchol}, andness: {andness_highchol:.4f}")
    
    # Get HighChol values and apply transformation
    highchol_values = X_test[:, highchol_idx]
    if tree_structure['features'][highchol_position]['transformation'] == 'negation':
        highchol_values = 1.0 - highchol_values
    elif tree_structure['features'][highchol_position]['transformation'] == 'peak':
        if 'transformation_params' in tree_structure['features'][highchol_position]:
            peak_loc = float(tree_structure['features'][highchol_position]['transformation_params']['peak_location'])
            highchol_values = 1.0 - torch.abs(highchol_values - peak_loc)
    
    # Aggregate baseline directly with HighChol
    direct_result = aggregator.aggregate_tensor(
        baseline_result, highchol_values,
        torch.tensor(andness_highchol, device=device),
        weights_highchol[0], weights_highchol[1]
    )
    
    direct_predictions = (direct_result > 0.5).float()
    direct_accuracy = (direct_predictions == Y_test.squeeze()).float().mean().item()
    
    print(f"\nResults (threshold=0.5):")
    print(f"  Accuracy: {direct_accuracy * 100:.2f}%")
    
    print(f"\nComparison:")
    print(f"  HighChol alone:                          {accuracy_chol * 100:.2f}%")
    print(f"  Baseline + HighChol (skip middle):       {direct_accuracy * 100:.2f}%")
    print(f"  Baseline + all features through HighChol: {combined_accuracy * 100:.2f}%")
    print(f"  Benefit of middle features:              {(combined_accuracy - direct_accuracy) * 100:+.2f}%")
