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

input_size = 100

print(f"\n📊 Generating rare-gate expression with {input_size} variables...")
print(f"   Expression: y = A ∧ ((B ∧ C) ∨ ⋁(Gi ∧ Xi))")
print(f"   Critical: A (x1), B (x2), C (x3)")
print(f"   Guards: G0-G47 at odd indices, X0-X47 at even indices")
print(f"   Goal: BACON should place A, B, C at rightmost tree positions")

# Set seed for reproducibility
random.seed(42)

def evaluate_bacon_gate_or_pairs(x, p=5, *, a_idx=0, b_idx=1, gate_idx=99, start_idx=2):
    """
    Implements:
        y = X_gate AND ( (A AND B) OR OR_{i=1..p} (Gi AND Xi) )

    Default mapping for m=100:
      - A        = x[a_idx]     (default 0)
      - B        = x[b_idx]     (default 1)
      - X_gate   = x[gate_idx]  (default 99)
      - Guard/payload pairs start at x[start_idx] (default 2) as:
          (x[2] & x[3]), (x[4] & x[5]), ..., total p pairs.

    Notes:
      - Inputs outside these indices are unused by the ground-truth rule (true distractors).
      - If you want to "use all variables", set p large enough and/or change mapping.
    """
    n = len(x)
    if n == 0:
        raise ValueError("x must be non-empty")
    if not (0 <= a_idx < n and 0 <= b_idx < n and 0 <= gate_idx < n):
        raise ValueError(f"Index out of range: a_idx={a_idx}, b_idx={b_idx}, gate_idx={gate_idx}, len(x)={n}")
    if p < 0:
        raise ValueError("p must be >= 0")

    A = bool(x[a_idx])
    B = bool(x[b_idx])
    gate = bool(x[gate_idx])

    direct = A and B

    guarded_or = False
    # Need 2*p elements starting from start_idx: indices [start_idx .. start_idx + 2*p - 1]
    end_needed = start_idx + 2 * p
    if end_needed > n:
        raise ValueError(
            f"Not enough inputs for p={p} pairs starting at {start_idx}. "
            f"Need at least {end_needed} elements, got {n}."
        )

    for i in range(start_idx, start_idx + 2 * p, 2):
        guarded_or |= (bool(x[i]) and bool(x[i + 1]))

    return int(gate and (direct or guarded_or))

# Generate random samples
print(f"   Generating 70,000 random samples...")
num_samples = 60000
data = []
labels = []

for _ in range(num_samples):
    x = [random.randint(0, 1) for _ in range(input_size)]
    y = evaluate_bacon_gate_or_pairs(x, p=10)
    data.append(x)
    labels.append([y])

x_combined = torch.tensor(data, dtype=torch.float32, device=device)
y_combined = torch.tensor(labels, dtype=torch.float32, device=device)

# Split into train/test (50k train, 20k test from SAME expression)
x_train = x_combined[:40000]
y_train = y_combined[:40000]
x_test = x_combined[20000:]
y_test = y_combined[20000:]

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
print(f"   A (x1, index 0): Critical outer gate - if False, always False")
print(f"   B (x2, index 1): Critical for rare direct path (B AND C)")
print(f"   C (x3, index 2): Critical for rare direct path (B AND C)")
print(f"   G0-G47, X0-X47: Guard pairs, contribute to OR but individually weak")
print(f"   Expected: BACON places A, B, C at tree positions 97-99 (rightmost)")

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
    acceptance_threshold=1.0, 
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

# Try locked_perm first (hard permutation), then fall back to soft permutation
perm = None
is_hard = False

if hasattr(bacon.assembler, 'locked_perm') and bacon.assembler.locked_perm is not None:
    perm = bacon.assembler.locked_perm.cpu().numpy()
    is_hard = True
    print(f"   Using hard permutation (frozen)")
elif hasattr(bacon.assembler, 'input_to_leaf') and hasattr(bacon.assembler.input_to_leaf, 'logits'):
    # Use argmax of soft permutation matrix
    import numpy as np
    if hasattr(bacon.assembler.input_to_leaf, 'sinkhorn'):
        soft_perm = bacon.assembler.input_to_leaf.sinkhorn(
            bacon.assembler.input_to_leaf.logits,
            temperature=bacon.assembler.input_to_leaf.temperature,
            n_iters=bacon.assembler.input_to_leaf.sinkhorn_iters
        ).detach().cpu().numpy()
    else:
        import torch.nn.functional as F
        soft_perm = F.softmax(bacon.assembler.input_to_leaf.logits, dim=1).detach().cpu().numpy()
    
    perm = np.argmax(soft_perm, axis=1)
    is_hard = False
    print(f"   Using soft permutation argmax (not frozen)")

if perm is not None:
    # Find where original variables ended up in the tree
    # perm[i] tells us which original variable goes to position i in tree
    pos_of_x1 = None
    pos_of_x2 = None
    pos_of_x100 = None
    
    pos_of_x3 = None
    
    for tree_pos in range(len(perm)):
        orig_var = perm[tree_pos]
        if orig_var == 0:  # A
            pos_of_x1 = tree_pos
        elif orig_var == 1:  # B
            pos_of_x2 = tree_pos
        elif orig_var == 2:  # C
            pos_of_x3 = tree_pos
    
    print(f"\n   📍 Variable Placement:")
    print(f"      A (x1, index 0) → tree position {pos_of_x1}")
    print(f"      B (x2, index 1) → tree position {pos_of_x2}")
    print(f"      C (x3, index 2) → tree position {pos_of_x3}")
    
    # For right-to-left evaluation: rightmost positions (97-99) are most critical
    # A, B, C should be at positions 97-99 to act as final gates
    print(f"\n   🎯 Structure Analysis:")
    critical_positions = [pos_of_x1, pos_of_x2, pos_of_x3]
    critical_at_end = sum(1 for pos in critical_positions if pos is not None and pos >= input_size - 5)
    
    if critical_at_end == 3:
        print(f"      ✅ EXCELLENT: All 3 critical vars at end positions ({pos_of_x1}, {pos_of_x2}, {pos_of_x3})")
    elif critical_at_end >= 2:
        print(f"      ✅ GOOD: {critical_at_end}/3 critical vars near end")
    elif critical_at_end == 1:
        print(f"      ⚠️  PARTIAL: Only 1/3 critical var near end")
    else:
        print(f"      ❌ MISS: No critical vars at end (positions {pos_of_x1}, {pos_of_x2}, {pos_of_x3})")
    
    # Check if A, B, C are clustered together
    if all(pos is not None for pos in critical_positions):
        min_pos = min(critical_positions)
        max_pos = max(critical_positions)
        spread = max_pos - min_pos
        
        if spread < 5:
            print(f"      ✅ Critical vars clustered together (spread: {spread} positions)")
        elif spread < 15:
            print(f"      ⚠️  Critical vars somewhat spread (spread: {spread} positions)")
        else:
            print(f"      ❌ Critical vars scattered (spread: {spread} positions)")
    
    # Count how many critical variables are correctly placed
    success_count = critical_at_end
    
    print(f"\n   📊 Discovery Score: {success_count}/3 critical variables at correct positions")
    
    # Analyze confidence distribution - show top positions with highest confidence
    if not is_hard and hasattr(bacon.assembler, 'input_to_leaf') and hasattr(bacon.assembler.input_to_leaf, 'logits'):
        print(f"\n   🔍 Top Confidence Analysis:")
        print(f"      Checking if high-confidence positions correspond to critical variables...")
        
        # Get confidence (max probability) for each row
        if hasattr(bacon.assembler.input_to_leaf, 'sinkhorn'):
            soft_perm = bacon.assembler.input_to_leaf.sinkhorn(
                bacon.assembler.input_to_leaf.logits,
                temperature=bacon.assembler.input_to_leaf.temperature,
                n_iters=bacon.assembler.input_to_leaf.sinkhorn_iters
            ).detach().cpu().numpy()
        else:
            import torch.nn.functional as F
            soft_perm = F.softmax(bacon.assembler.input_to_leaf.logits, dim=1).detach().cpu().numpy()
        
        # Get max probability (confidence) for each tree position
        confidences = np.max(soft_perm, axis=1)
        assigned_vars = np.argmax(soft_perm, axis=1)
        
        # Create list of (tree_pos, confidence, assigned_var)
        conf_list = [(i, confidences[i], assigned_vars[i]) for i in range(len(confidences))]
        conf_list.sort(key=lambda x: x[1], reverse=True)
        
        # Show top 10 most confident positions
        print(f"\n      Top 10 Most Confident Positions:")
        critical_in_top10 = 0
        for rank, (tree_pos, conf, orig_var) in enumerate(conf_list[:10], 1):
            is_critical = orig_var in [0, 1, 2]
            marker = "🌟" if is_critical else "  "
            var_name = f"x{orig_var+1}"
            if is_critical:
                critical_in_top10 += 1
                if orig_var == 0:
                    var_name = "A/x1 (CRITICAL)"
                elif orig_var == 1:
                    var_name = "B/x2 (CRITICAL)"
                elif orig_var == 2:
                    var_name = "C/x3 (CRITICAL)"
            print(f"      {marker} #{rank}: Tree pos {tree_pos:3d} → {var_name:20s} (confidence: {conf:.3f})")
        
        print(f"\n      💡 Critical variables in top 10: {critical_in_top10}/3")
        if critical_in_top10 == 3:
            print(f"      ✅ EXCELLENT: All 3 critical variables have high confidence!")
        elif critical_in_top10 >= 2:
            print(f"      ✅ GOOD: {critical_in_top10}/3 critical variables have high confidence")
        elif critical_in_top10 == 1:
            print(f"      ⚠️  PARTIAL: Only 1/3 critical variables have high confidence")
        else:
            print(f"      ❌ MISS: No critical variables in top 10 confidence")
        
        # Show overall confidence statistics
        avg_conf = confidences.mean()
        print(f"\n      📊 Overall Statistics:")
        print(f"         Average confidence: {avg_conf:.3f}")
        print(f"         Min confidence: {confidences.min():.3f}")
        print(f"         Max confidence: {confidences.max():.3f}")
        
        # Show confidence for critical variables specifically
        conf_x1 = confidences[pos_of_x1] if pos_of_x1 is not None else 0
        conf_x2 = confidences[pos_of_x2] if pos_of_x2 is not None else 0
        conf_x3 = confidences[pos_of_x3] if pos_of_x3 is not None else 0
        
        print(f"\n      🎯 Critical Variable Confidence:")
        print(f"         A (x1): {conf_x1:.3f} (tree pos {pos_of_x1})")
        print(f"         B (x2): {conf_x2:.3f} (tree pos {pos_of_x2})")
        print(f"         C (x3): {conf_x3:.3f} (tree pos {pos_of_x3})")
        
        avg_critical_conf = (conf_x1 + conf_x2 + conf_x3) / 3
        print(f"         Average critical confidence: {avg_critical_conf:.3f}")
        
        if avg_critical_conf > avg_conf * 1.5:
            print(f"         ✅ Critical variables have {avg_critical_conf/avg_conf:.1f}x higher confidence than average!")
        elif avg_critical_conf > avg_conf:
            print(f"         ✅ Critical variables have above-average confidence")
        else:
            print(f"         ⚠️  Critical variables have below-average confidence")
else:
    print(f"   ⚠️  Cannot analyze permutation (no locked_perm or logits found)")

# Tree pruning analysis - test feature importance by neutralizing from left
print("\n" + "=" * 80)
print("🔍 TREE PRUNING ANALYSIS")
print("=" * 80)
print("Testing feature importance by neutralizing features from left (least important first)")
print("Expected: accuracy should remain high until critical variables (A, B, C) are removed")

def evaluate_with_neutralized_features(model, X, y, num_neutralized):
    """Evaluate accuracy with the first num_neutralized features neutralized.
    
    Neutralization: Set weights so neutralized inputs have weight=0, keeping inputs have weight=1.
    This makes the aggregator pass through the kept input.
    
    In left-associative tree:
    - Aggregator 0: combines input[0] (left) and input[1] (right)
    - Aggregator i: combines result[i-1] (left) and input[i+1] (right)
    
    To neutralize the first k features, set weight_left=0, weight_right=1 for first k aggregators.
    """
    if num_neutralized == 0:
        with torch.no_grad():
            return (model(X).round() == y).float().mean().item()
    
    # Store original weight_mode and weights
    original_weight_mode = model.assembler.weight_mode
    original_weights = []
    
    # Temporarily switch to trainable mode so forward pass uses our modified weights
    model.assembler.weight_mode = "trainable"
    
    for i in range(num_neutralized):
        original_weights.append(model.assembler.weights[i].data.clone())
        # Set left weight to 0 (neutralize), right weight to 1 (pass through)
        model.assembler.weights[i].data[0] = 0.0  # left input weight = 0
        model.assembler.weights[i].data[1] = 1.0  # right input weight = 1
    
    with torch.no_grad():
        result = (model(X).round() == y).float().mean().item()
    
    # Restore original weights and weight_mode
    for i in range(num_neutralized):
        model.assembler.weights[i].data.copy_(original_weights[i])
    model.assembler.weight_mode = original_weight_mode
    
    return result

# Baseline accuracy
accuracies = []
with torch.no_grad():
    baseline_output = bacon(x_test)
    baseline_accuracy = (baseline_output.round() == y_test).float().mean().item()
    print(f"\n✅ Baseline accuracy: {baseline_accuracy * 100:.2f}%")

# Test pruning from left (removing least important features first)
print(f"\n📊 Pruning features from left (tree positions 0 → {input_size-1}):")
critical_features_removed = []

for i in range(1, input_size):
    pruned_accuracy = evaluate_with_neutralized_features(bacon, x_test, y_test, i)
    accuracies.append(pruned_accuracy)
    
    # Track when we hit critical variables
    if perm is not None:
        removed_var = perm[i-1]  # Feature at position i-1 was just removed
        if removed_var in [0, 1, 2]:  # Critical: A, B, or C
            var_name = {0: "A", 1: "B", 2: "C"}[removed_var]
            critical_features_removed.append((i-1, var_name, pruned_accuracy))
    
    # Print every 10th position or when accuracy drops significantly
    if i % 10 == 0 or (i > 1 and abs(pruned_accuracy - accuracies[-2]) > 0.05):
        print(f"   Position {i:3d}: {pruned_accuracy * 100:.2f}% accuracy ({input_size - i} features remaining)")

# Summary of critical variable removal
if critical_features_removed:
    print(f"\n🎯 Critical Variable Removal Impact:")
    for pos, var_name, acc in critical_features_removed:
        acc_drop = baseline_accuracy - acc
        print(f"   Tree pos {pos}: Removed {var_name} → accuracy {acc*100:.2f}% (drop: {acc_drop*100:.2f}%)")
else:
    print(f"\n⚠️  No critical variables identified in pruning sequence")

# Identify the "cliff" where accuracy drops
print(f"\n📉 Accuracy Cliff Analysis:")
max_drop_idx = 0
max_drop = 0
for i in range(1, len(accuracies)):
    drop = accuracies[i-1] - accuracies[i]
    if drop > max_drop:
        max_drop = drop
        max_drop_idx = i

if max_drop > 0.05:
    print(f"   Largest drop at position {max_drop_idx}: {max_drop*100:.2f}% accuracy loss")
    if perm is not None and max_drop_idx < len(perm):
        removed_var = perm[max_drop_idx]
        print(f"   Removed variable: x{removed_var+1} (original index {removed_var})")
else:
    print(f"   No significant accuracy cliff found (max drop: {max_drop*100:.2f}%)")

# Plot accuracy vs number of features pruned
try:
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(accuracies) + 1), [a * 100 for a in accuracies], marker='o', markersize=2)
    plt.axhline(y=baseline_accuracy * 100, color='g', linestyle='--', label='Baseline')
    
    # Mark critical variable removals
    if critical_features_removed:
        for pos, var_name, acc in critical_features_removed:
            plt.axvline(x=pos+1, color='r', linestyle=':', alpha=0.5)
            plt.text(pos+1, acc*100, f' {var_name}', rotation=90, verticalalignment='bottom')
    
    plt.title("Tree Pruning Analysis: Accuracy vs Features Removed")
    plt.xlabel("Number of Features Pruned from Left (tree positions)")
    plt.ylabel("Test Accuracy (%)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("bacon_fixed_pruning_analysis.png")
    print(f"\n💾 Saved pruning plot to: bacon_fixed_pruning_analysis.png")
    plt.show()
except Exception as e:
    print(f"\n⚠️  Could not generate plot: {e}")

print("\n✅ BACON fixed expression benchmark complete!")
