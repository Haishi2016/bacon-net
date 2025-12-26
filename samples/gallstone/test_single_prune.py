"""
Test single feature pruning to understand where the accuracy drop comes from
"""

import sys
sys.path.insert(0, '../../')

import torch
from bacon.baconNet import baconNet
from bacon.transformationLayer import (
    IdentityTransformation, 
    NegationTransformation,
    PeakTransformation,
    ValleyTransformation,
    StepUpTransformation,
    StepDownTransformation
)
from dataset import prepare_data

# Prepare data
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
X_train, Y_train, X_test, Y_test, feature_names = prepare_data(device)
X_all = torch.cat([X_train, X_test], dim=0)
Y_all = torch.cat([Y_train, Y_test], dim=0)

print(f"\nData splits:")
print(f"  Train: {X_train.shape[0]} samples")
print(f"  Test:  {X_test.shape[0]} samples")
print(f"  All:   {X_all.shape[0]} samples")


# Load model
bacon = baconNet(
    input_size=len(feature_names),
    aggregator='lsp.half_weight',
    weight_mode='fixed',
    use_transformation_layer=True,
    transformations=[
        IdentityTransformation(1),
        NegationTransformation(1),
        PeakTransformation(1),
        ValleyTransformation(1),
        StepUpTransformation(1),
        StepDownTransformation(1)
    ],
    weight_normalization='softmax',
    use_class_weighting=True
)

checkpoint = torch.load('assembler.pth', map_location=device, weights_only=False)
bacon.assembler.is_frozen = checkpoint.get('is_frozen', False)
bacon.assembler.locked_perm = checkpoint.get('locked_perm', None)
bacon.assembler.tree_layout = checkpoint.get('tree_layout', 'left')

from bacon.frozonInputToLeaf import frozenInputToLeaf
bacon.assembler.input_to_leaf = frozenInputToLeaf(
    bacon.assembler.locked_perm, 
    bacon.assembler.original_input_size
).to(device)

bacon.assembler.load_state_dict(checkpoint['model_state_dict'])
bacon.eval()

# Get baseline accuracy on different splits
with torch.no_grad():
    train_output = bacon.assembler(X_train)
    train_accuracy = ((train_output > 0.5).float() == Y_train).float().mean().item()
    
    test_output = bacon.assembler(X_test)
    test_accuracy = ((test_output > 0.5).float() == Y_test).float().mean().item()
    
    all_output = bacon.assembler(X_all)
    all_accuracy = ((all_output > 0.5).float() == Y_all).float().mean().item()

print("="*70)
print("BASELINE ACCURACIES (NO PRUNING)")
print("="*70)
print(f"Train set accuracy: {train_accuracy*100:.2f}%")
print(f"Test set accuracy:  {test_accuracy*100:.2f}%")
print(f"All data accuracy:  {all_accuracy*100:.2f}%")
print(f"\nBaseline features: 0={feature_names[bacon.assembler.locked_perm[0].item()]}, "
      f"1={feature_names[bacon.assembler.locked_perm[1].item()]}")

# Save original weights
original_weights = [w.data.clone() for w in bacon.assembler.weights]

# Test pruning individual features
print("\n" + "="*70)
print("INDIVIDUAL FEATURE PRUNING (one at a time)")
print("="*70)

for i in range(min(10, len(feature_names))):  # Test first 10 features
    # Restore weights
    for j, w in enumerate(bacon.assembler.weights):
        w.data.copy_(original_weights[j])
    bacon.assembler.pruned_aggregators.clear()
    
    # Prune single feature
    bacon.assembler.prune_features(i)
    
    with torch.no_grad():
        pruned_output = bacon.assembler(X_all)
        pruned_accuracy = ((pruned_output > 0.5).float() == Y_all).float().mean().item()
    
    feature_name = feature_names[bacon.assembler.locked_perm[i].item()]
    drop = all_accuracy - pruned_accuracy
    
    print(f"Feature {i:2d} ({feature_name:40s}): {pruned_accuracy*100:.2f}% (drop: {drop*100:+.2f}%)")

# Restore weights
for j, w in enumerate(bacon.assembler.weights):
    w.data.copy_(original_weights[j])
bacon.assembler.pruned_aggregators.clear()

print("\n" + "="*70)
print("KEY FINDINGS")
print("="*70)
print("""
If the drop appears when pruning feature 2 individually, then the issue is
that feature 2 (ECW) is important even though it's the 3rd feature in the tree.

The 5% drop we saw earlier (78.12% → 73.35%) is NOT from the baseline being
removed, but from feature 2 being removed.
""")
