# BACON approach for 1000-variable boolean expression inference
# Fixed expression: ((((x1 AND x2) OR x3) OR x4) ... OR x999) AND x1000
# Key insight: Only x1, x2, and x1000 matter! Middle variables (x3-x999) are irrelevant.

import sys
sys.path.insert(0, '../../')

import torch
from bacon.baconNet import baconNet
from bacon.visualization import print_tree_structure
import logging
import time
import random

logging.basicConfig(level=logging.INFO, format='%(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 80)
print("🧪 BACON: 1000-Variable Fixed Boolean Expression Inference")
print("=" * 80)

input_size = 1000

print(f"\n📊 Generating fixed boolean expression with {input_size} variables...")
print(f"   Expression: ((((x1 AND x2) OR x3) OR x4) ... OR x999) AND x1000")
print(f"   Key variables: x1, x2, x1000 (middle vars irrelevant)")

# Set seed for reproducibility
random.seed(42)

def evaluate_fixed_expression(x):
    """
    Evaluate fixed expression: ((((x1 AND x2) OR x3) OR x4) ... OR x999) AND x1000
    
    This simplifies to: (x1 AND x2 OR <any x3-x999 is True>) AND x1000
    Which further simplifies to:
    - If x1000 == 0: always False
    - If x1000 == 1: True if (x1 AND x2) OR (any of x3-x999 is True)
    """
    # Start with x1 AND x2
    result = bool(x[0]) and bool(x[1])
    
    # OR with x3 through x999
    for i in range(2, input_size - 1):
        result = result or bool(x[i])
    
    # Final AND with x1000
    result = result and bool(x[input_size - 1])
    
    return int(result)

# Generate random samples
print(f"   Generating 70,000 random samples...")
num_samples = 70000
data = []
labels = []

for _ in range(num_samples):
    x = [random.randint(0, 1) for _ in range(input_size)]
    y = evaluate_fixed_expression(x)
    data.append(x)
    labels.append([y])

x_combined = torch.tensor(data, dtype=torch.float32, device=device)
y_combined = torch.tensor(labels, dtype=torch.float32, device=device)

# Split into train/test (50k train, 20k test from SAME expression)
x_train = x_combined[:50000]
y_train = y_combined[:50000]
x_test = x_combined[50000:]
y_test = y_combined[50000:]

print(f"✅ Data generated")
print(f"   Training samples: {len(x_train)}")
print(f"   Test samples: {len(x_test)}")

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

# Analyze key variables
print(f"\n🔍 Key variable analysis:")
print(f"   x1 (index 0): Critical for (x1 AND x2)")
print(f"   x2 (index 1): Critical for (x1 AND x2)")
print(f"   x3-x999: Only matter if ALL are False AND (x1 AND x2) is False")
print(f"   x1000 (index 999): Critical final gate")

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
print(f"   Temperature: 3.0 → 0.05")

print("\n🔥 Training BACON model...")
print(f"   Testing if BACON can discover x1, x2, x1000 as critical variables")
start_time = time.time()

best_model, best_accuracy = bacon.find_best_model(
    x_train, y_train, 
    x_test, y_test, 
    acceptance_threshold=0.95, 
    use_hierarchical_permutation=False,
    max_epochs=12000,
    hierarchical_group_size=25,
    hierarchical_epochs_per_attempt=5000,
    hierarchical_bleed_ratio=0.5,
    attempts=3,
    annealing_epochs=1500,
    frozen_training_epochs=1000,
    convergence_patience=500,
    loss_weight_perm_sparsity=0.0,  # Overridden by sparsity_schedule
    sparsity_schedule=(10.0, 100.0, 2000),  # Start high → very high to force commitment
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

# Check if BACON discovered the critical variables
print("\n🔍 Critical Variable Discovery:")
print(f"   Checking if BACON placed x1, x2, x1000 in important positions...")

if hasattr(bacon.assembler, 'locked_perm') and bacon.assembler.locked_perm is not None:
    perm = bacon.assembler.locked_perm.cpu().numpy()
    
    # Find where original variables ended up in the tree
    # perm[i] tells us which original variable goes to position i in tree
    pos_of_x1 = None
    pos_of_x2 = None
    pos_of_x1000 = None
    
    for tree_pos in range(len(perm)):
        orig_var = perm[tree_pos]
        if orig_var == 0:  # x1
            pos_of_x1 = tree_pos
        elif orig_var == 1:  # x2
            pos_of_x2 = tree_pos
        elif orig_var == 999:  # x1000
            pos_of_x1000 = tree_pos
    
    print(f"   x1 (orig index 0) → tree position {pos_of_x1}")
    print(f"   x2 (orig index 1) → tree position {pos_of_x2}")
    print(f"   x1000 (orig index 999) → tree position {pos_of_x1000}")
    
    # For left-associative tree, rightmost position (999) is most critical
    # because it's the last AND gate
    if pos_of_x1000 is not None and pos_of_x1000 > 990:
        print(f"   ✅ SUCCESS: x1000 placed near end (position {pos_of_x1000}/999)")
    else:
        print(f"   ❌ MISS: x1000 not at end (position {pos_of_x1000}/999)")
    
    # x1 and x2 should ideally be early (leftmost) for (x1 AND x2)
    if pos_of_x1 is not None and pos_of_x2 is not None:
        if pos_of_x1 < 100 and pos_of_x2 < 100:
            print(f"   ✅ SUCCESS: x1, x2 placed early (positions {pos_of_x1}, {pos_of_x2})")
        else:
            print(f"   ⚠️  PARTIAL: x1, x2 not both early (positions {pos_of_x1}, {pos_of_x2})")
else:
    print(f"   ⚠️  Model not frozen, cannot analyze permutation")

print("\n✅ BACON fixed expression benchmark complete!")
