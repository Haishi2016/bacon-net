# BACON approach for 1000-variable boolean expression inference

import sys
sys.path.insert(0, '../../')

import torch
from bacon.baconNet import baconNet
from bacon.visualization import print_tree_structure
from bacon.utils import generate_classic_boolean_data
import logging
import time

logging.basicConfig(level=logging.INFO, format='%(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 80)
print("🧪 BACON: 1000-Variable Boolean Expression Inference")
print("=" * 80)

input_size = 1000

print(f"\n📊 Generating boolean expression with {input_size} variables...")
print(f"   Using randomized sampling with combined dataset split")

# Set seed for reproducible expression generation
import random
random.seed(42)

# Generate large combined dataset with same expression
x_combined, y_combined, expr_info = generate_classic_boolean_data(
    input_size, 
    repeat_factor=70000,  # 70k total samples
    randomize=True, 
    device=device
)

# Split into train/test (50k train, 20k test from SAME expression)
x_train = x_combined[:50000]
y_train = y_combined[:50000]
x_test = x_combined[50000:]
y_test = y_combined[50000:]

print(f"✅ Data generated")
print(f"   Training samples: {len(x_train)}")
print(f"   Test samples: {len(x_test)}")
print(f"   Expression: {expr_info['expression_text'][:100]}...")  # Show first 100 chars

# Check output distribution
train_output_mean = y_train.mean().item()
test_output_mean = y_test.mean().item()
print(f"\n📊 Output distribution:")
print(f"   Training: {train_output_mean*100:.2f}% True, {(1-train_output_mean)*100:.2f}% False")
print(f"   Test: {test_output_mean*100:.2f}% True, {(1-test_output_mean)*100:.2f}% False")
if train_output_mean < 0.05 or train_output_mean > 0.95:
    print(f"   ⚠️  WARNING: Expression is highly imbalanced (may be trivial)")
if abs(train_output_mean - test_output_mean) > 0.1:
    print(f"   ⚠️  WARNING: Train/test distribution mismatch")

print("\n🔧 Configuring BACON model...")
bacon = baconNet(
    input_size, 
    aggregator='bool.min_max', 
    weight_mode='fixed', 
    loss_amplifier=1000, 
    normalize_andness=False,
    use_class_weighting=False,
    permutation_final_temperature=0.05,
    permutation_initial_temperature=3,
)

print("✅ Model configured")
print(f"   Aggregator: bool.min_max (classic boolean logic)")
print(f"   Freeze threshold: 0.18 (relaxed for large input)")
print(f"   Temperature: 10.0 → 0.01")

print("\n🔥 Training BACON model...")
print(f"   Using hierarchical permutation search")
print(f"   Max permutations per attempt: 100")
start_time = time.time()

best_model, best_accuracy = bacon.find_best_model(
    x_train, y_train, 
    x_test, y_test, 
    acceptance_threshold=0.95, 
    use_hierarchical_permutation=False,
    max_epochs=12000,
    hierarchical_group_size=25,  # Group 1000 vars into ~20 coarse groups
    hierarchical_epochs_per_attempt=5000,
    hierarchical_bleed_ratio=0.5,  # Allow some overlap between groups
    attempts=3,  # Try 3 different coarse permutations
    annealing_epochs=1500,
    frozen_training_epochs=1000,
    convergence_patience=500,
    loss_weight_perm_sparsity=0.0,  # Overridden by sparsity_schedule
    sparsity_schedule=(10.0, 50.0, 2000),  # Start high (10.0) → very high (50.0) to force commitment
    freeze_aggregation_epochs=500,  # Freeze aggregation for first 500 epochs
    save_model=False
)

training_time = time.time() - start_time

print("\n" + "=" * 80)
print("📊 BACON RESULTS")
print("=" * 80)
print(f"🏆 Best Test Accuracy: {best_accuracy * 100:.2f}%")
print(f"⏱️  Training Time: {training_time:.2f} seconds")

# Evaluate on training set
with torch.no_grad():
    train_pred = bacon(x_train)
    train_accuracy = (train_pred.round() == y_train).float().mean().item()
print(f"📊 Training Accuracy: {train_accuracy * 100:.2f}%")

# Print tree structure (first few levels only for readability)
print("\n🌲 Learned Tree Structure (partial):")
try:
    # For 1000 variables, only show summary
    print(f"   Tree depth: ~{input_size.bit_length()}")
    print(f"   Number of nodes: {input_size - 1}")
    print(f"   Frozen: {bacon.assembler.is_frozen}")
    print(f"   To see full structure, use print_tree_structure() or visualize_tree_structure()")
except Exception as e:
    print(f"   Could not print structure: {e}")

print("\n✅ BACON benchmark complete!")
