"""Debug script to understand why model outputs 50% accuracy."""
import sys
sys.path.insert(0, '../../')
sys.path.insert(0, '../')

import torch
from dataset import prepare_data, balance_data
from common import create_bacon_model
from bacon.transformationLayer import IdentityTransformation, NegationTransformation
from bacon.policies import FixedAndnessPolicy

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Load data
X_train, Y_train, X_test, Y_test, feature_names = prepare_data(device)
X_train_bal, Y_train_bal = balance_data(X_train, Y_train, device)

print(f"Features: {len(feature_names)}")
print(f"Train samples: {len(X_train_bal)}, Test samples: {len(X_test)}")
print(f"Y_train distribution: {Y_train_bal.sum().item()} positives / {len(Y_train_bal)} total")
print(f"Y_test distribution: {Y_test.sum().item()} positives / {len(Y_test)} total")

# Create model without policy
bacon = create_bacon_model(
    input_size=len(feature_names),
    aggregator='lsp.half_weight',
    weight_mode='trainable',
    weight_normalization='softmax',
)

print("\n--- WITHOUT POLICY ---")
with torch.no_grad():
    out = bacon(X_test[:10])
    print(f"Outputs (first 10): {[f'{x:.4f}' for x in out.squeeze().tolist()]}")
    print(f"Biases: {[f'{b.item():.4f}' for b in bacon.assembler.biases[:5]]}...")
    print(f"Min/Max output: {out.min().item():.4f} / {out.max().item():.4f}")
    
# Check if outputs vary
with torch.no_grad():
    all_out = bacon(X_test)
    print(f"All outputs - min: {all_out.min():.4f}, max: {all_out.max():.4f}, std: {all_out.std():.4f}")

# Now with policy (penalty-based regularization)
print("\n--- WITH ANDNESS REGULARIZATION POLICY (target=0.5, penalty_weight=1.0) ---")
policy = FixedAndnessPolicy(andness=0.5, penalty_weight=1.0)
print(f"Policy target_bias: {policy.target_bias:.4f}")
print(f"Policy penalty_weight: {policy.penalty_weight}")

# Reset model
bacon2 = create_bacon_model(
    input_size=len(feature_names),
    aggregator='lsp.half_weight',
    weight_mode='trainable',
    weight_normalization='softmax',
)

# Compute penalty before any training
penalty_before = policy.penalty(bacon2.assembler)
print(f"Penalty (before training): {penalty_before.item():.4f}")

print(f"Biases (random init): {[f'{b.item():.4f}' for b in bacon2.assembler.biases[:5]]}...")
print(f"Biases requires_grad: {[b.requires_grad for b in bacon2.assembler.biases[:5]]}...")

with torch.no_grad():
    out_pol = bacon2(X_test[:10])
    print(f"Outputs (first 10): {[f'{x:.4f}' for x in out_pol.squeeze().tolist()]}")
    
# Show what the penalty would be for different andness values
print("\n--- PENALTY VS ANDNESS ---")
for test_bias in [-2.0, -1.0, 0.0, 1.0, 2.0]:
    test_andness = (torch.sigmoid(torch.tensor(test_bias)) * 3 - 1).item()
    penalty_val = policy.penalty_weight * (test_bias - policy.target_bias) ** 2
    print(f"bias={test_bias:+.1f} -> andness={test_andness:.3f} -> penalty={penalty_val:.4f}")

# Check andness values
print("\n--- CURRENT ANDNESS VALUES (from bacon2) ---")
for i, b in enumerate(bacon2.assembler.biases[:5]):
    andness = (torch.sigmoid(b) * 3 - 1).item()
    print(f"Bias {i}: {b.item():.4f} -> andness: {andness:.4f}")
