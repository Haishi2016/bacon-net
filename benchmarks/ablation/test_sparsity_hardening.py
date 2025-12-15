"""
Ablation Study: Sparsity Weight Impact on Hardening Performance

Tests how different sparsity weights during training affect accuracy when 
the soft permutation matrix is hardened (argmax) after training.

Setup:
- 100 variables (smaller than main benchmark)
- Random boolean expressions
- Train with different loss_weight_perm_sparsity values
- After training, harden the permutation and measure accuracy drop

Expected findings:
- Low sparsity (e.g., 0.01): High soft accuracy, large drop when hardened
- Medium sparsity (e.g., 1.0): Moderate soft accuracy, moderate drop
- High sparsity (e.g., 10.0): Lower soft accuracy, small drop (better hardening)
"""

import sys
sys.path.insert(0, '../../')

import torch
import numpy as np
from bacon.baconNet import baconNet
from bacon.utils import generate_classic_boolean_data
from bacon.visualization import print_tree_structure
import logging
import matplotlib.pyplot as plt
import json
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"🖥️  Device: {device}")
print("=" * 80)

# Configuration
input_size = 100  # Start with small size to verify NaN fix, then scale up
sparsity_values = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]  # Different sparsity weights to test
num_epochs = 1000  # Moderate training duration
annealing_epochs = 600  # Moderate annealing
freeze_aggregation_epochs = 0  # No aggregation freezing
acceptance_threshold = 1.0  # High threshold so model is never frozen during training 

# Generate combined dataset with same expression
print(f"📊 Generating dataset...")
print(f"   Input size: {input_size} variables")
print(f"   Training samples: 10,000")
print(f"   Test samples: 5,000")


# Generate large dataset with balanced sampling strategy
# For 100 variables, random sampling often gives very imbalanced data
# We'll generate multiple batches and filter for balance
print(f"   Generating balanced dataset...")

train_size = 40000
test_size = 40000
x, y,  expr_info = generate_classic_boolean_data(input_size, repeat_factor=(train_size+test_size), randomize=True, device=device)

x_train = x[:train_size]
y_train = y[:train_size]
x_test = x[train_size:train_size+test_size]
y_test = y[train_size:train_size+test_size]

print(f"✅ Dataset generated")
print(f"   Expression: {expr_info['expression_text'][:100]}...")
print(f"   Train distribution: {y_train.mean().item() * 100:.2f}% True")
print(f"   Test distribution: {y_test.mean().item() * 100:.2f}% True")

# Store results
results = []

print("\n" + "=" * 80)
print("🧪 ABLATION STUDY: Sparsity Weight vs Hardening Robustness")
print("=" * 80)

for sparsity_weight in sparsity_values:
    print(f"\n{'=' * 80}")
    print(f"🔬 Testing sparsity_weight = {sparsity_weight}")
    print(f"{'=' * 80}")
    
    # # Create model
    # bacon = baconNet(
    #     freeze_loss_threshold=0.0,
    #     input_size=input_size,
    #     use_transformation_layer=False,
    #     aggregator='bool.min_max',
    #     permutation_final_temperature=0.05,
    #     use_class_weighting=True,  # Enable class weighting like hello-world
    #     permutation_initial_temperature=5.0,
    #     loss_weight_perm_sparsity=sparsity_weight,  # Test different values
    #     weight_mode='fixed',
    #     loss_amplifier=1000,  # Use strong loss amplifier like hello-world
    #     normalize_andness=False  # Disable normalization for bool.min_max
    # )
    
    bacon = baconNet(input_size, aggregator='bool.min_max',
                  weight_mode='fixed', loss_amplifier=1000, normalize_andness=False)
    
    # Train model WITHOUT automatic freezing (we'll evaluate before freezing)
    print(f"\n📚 Training with sparsity_weight={sparsity_weight}...")
    # best_model, best_accuracy = bacon.find_best_model(
    #     x_train, y_train, 
    #     x_test, y_test,
    #     acceptance_threshold=acceptance_threshold,
    #     use_hierarchical_permutation=False,
    #     max_epochs=num_epochs,
    #     force_freeze=False,  # Disable forced freezing after training
    #     annealing_epochs=annealing_epochs,
    #     frozen_training_epochs=0,  # DISABLE frozen training to keep model soft
    #     convergence_patience=99999,  # Set very high to prevent convergence detection (which triggers plateau freezing)
    #     convergence_delta=0.0,  # Require improvement every epoch (prevents convergence)
    #     loss_weight_perm_sparsity=sparsity_weight,  # Use the test value
    #     freeze_aggregation_epochs=freeze_aggregation_epochs,        
    #     freeze_confidence_threshold=1.0,  # Set to 1.0 to prevent confidence-based freezing
    #     attempts=1,
    #     save_model=False
    # )
    
    (best_model, best_accuracy) = bacon.find_best_model(x_train, y_train, x_test, y_test,                                                        
        acceptance_threshold=0.95, 
        force_freeze=False,
        loss_weight_perm_sparsity=sparsity_weight,
        attempts=1, max_epochs=2000, save_model=False)


    # IMPORTANT: Evaluate BEFORE any freezing happens
    with torch.no_grad():
        # Model should still be soft at this point
        if not hasattr(bacon.assembler.input_to_leaf, 'logits'):
            print(f"   ⚠️  Warning: Model was frozen during training despite freeze_loss_threshold=999")
            # Extract from frozen model
            soft_perm = bacon.assembler.input_to_leaf.P_hard.cpu().numpy()
            # For frozen models, soft = hard
            soft_output = bacon(x_test)
            soft_accuracy = (soft_output.round() == y_test).float().mean().item()
            hard_accuracy = soft_accuracy
            accuracy_drop = 0.0
            relative_drop = 0.0
        else:
            # Model is still soft - extract soft permutation
            if hasattr(bacon.assembler.input_to_leaf, 'sinkhorn'):
                soft_perm = bacon.assembler.input_to_leaf.sinkhorn(
                    bacon.assembler.input_to_leaf.logits,
                    temperature=bacon.assembler.input_to_leaf.temperature,
                    n_iters=bacon.assembler.input_to_leaf.sinkhorn_iters
                ).detach().cpu().numpy()
            else:
                import torch.nn.functional as F
                soft_perm = F.softmax(bacon.assembler.input_to_leaf.logits, dim=1).detach().cpu().numpy()
            
            # Evaluate SOFT accuracy (with continuous Gumbel-Sinkhorn)
            soft_output = bacon(x_test)
            soft_accuracy = (soft_output.round() == y_test).float().mean().item()
            
            # Now simulate HARD accuracy by temporarily replacing with frozen permutation
            hard_perm = np.argmax(soft_perm, axis=1)
            
            # Temporarily freeze the model
            from bacon.frozonInputToLeaf import frozenInputToLeaf
            original_input_to_leaf = bacon.assembler.input_to_leaf
            bacon.assembler.input_to_leaf = frozenInputToLeaf(
                hard_perm, 
                bacon.assembler.original_input_size
            ).to(x_test.device)
            
            # Evaluate with hard permutation
            hard_output = bacon(x_test)
            hard_accuracy = (hard_output.round() == y_test).float().mean().item()
            
            # Restore soft model
            bacon.assembler.input_to_leaf = original_input_to_leaf
            
            # Calculate accuracy drop
            accuracy_drop = soft_accuracy - hard_accuracy
            relative_drop = (accuracy_drop / soft_accuracy * 100) if soft_accuracy > 0 else 0
        
        # Confidence: average max probability per row
        confidence = np.mean(np.max(soft_perm, axis=1))
        
        # Uniqueness: count unique argmax columns
        hard_perm = np.argmax(soft_perm, axis=1)
        unique_cols = len(np.unique(hard_perm))
        uniqueness = unique_cols / input_size
    
    # Store results
    result = {
        'sparsity_weight': sparsity_weight,
        'soft_accuracy': soft_accuracy,
        'hard_accuracy': hard_accuracy,
        'accuracy_drop': accuracy_drop,
        'relative_drop_pct': relative_drop,
        'confidence': confidence,
        'uniqueness': uniqueness,
        'unique_columns': unique_cols
    }
    results.append(result)
    
    # Print summary
    print(f"\n📊 Results for sparsity_weight={sparsity_weight}:")
    print(f"   Soft accuracy:  {soft_accuracy * 100:.2f}%")
    print(f"   Hard accuracy:  {hard_accuracy * 100:.2f}%")
    print(f"   Accuracy drop:  {accuracy_drop * 100:.2f}% (relative: {relative_drop:.2f}%)")
    print(f"   Confidence:     {confidence:.4f}")
    print(f"   Uniqueness:     {unique_cols}/{input_size} columns ({uniqueness * 100:.1f}%)")

# Save results to JSON
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_file = f"sparsity_ablation_results_{timestamp}.json"
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\n💾 Results saved to: {results_file}")

# Generate visualization
print(f"\n📈 Generating visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Soft vs Hard Accuracy
ax1 = axes[0, 0]
sparsity_vals = [r['sparsity_weight'] for r in results]
soft_accs = [r['soft_accuracy'] * 100 for r in results]
hard_accs = [r['hard_accuracy'] * 100 for r in results]

ax1.plot(sparsity_vals, soft_accs, marker='o', label='Soft (training)', linewidth=2)
ax1.plot(sparsity_vals, hard_accs, marker='s', label='Hard (argmax)', linewidth=2)
ax1.set_xscale('log')
ax1.set_xlabel('Sparsity Weight (log scale)')
ax1.set_ylabel('Accuracy (%)')
ax1.set_title('Soft vs Hard Accuracy')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Accuracy Drop
ax2 = axes[0, 1]
accuracy_drops = [r['accuracy_drop'] * 100 for r in results]
relative_drops = [r['relative_drop_pct'] for r in results]

ax2_twin = ax2.twinx()
line1 = ax2.plot(sparsity_vals, accuracy_drops, marker='o', color='red', 
                 label='Absolute drop', linewidth=2)
line2 = ax2_twin.plot(sparsity_vals, relative_drops, marker='s', color='orange', 
                      label='Relative drop (%)', linewidth=2)
ax2.set_xscale('log')
ax2.set_xlabel('Sparsity Weight (log scale)')
ax2.set_ylabel('Absolute Accuracy Drop (%)', color='red')
ax2_twin.set_ylabel('Relative Drop (%)', color='orange')
ax2.set_title('Accuracy Drop When Hardened')
ax2.tick_params(axis='y', labelcolor='red')
ax2_twin.tick_params(axis='y', labelcolor='orange')

# Combine legends
lines = line1 + line2
labels = [l.get_label() for l in lines]
ax2.legend(lines, labels, loc='upper right')
ax2.grid(True, alpha=0.3)

# Plot 3: Confidence
ax3 = axes[1, 0]
confidences = [r['confidence'] for r in results]

ax3.plot(sparsity_vals, confidences, marker='o', color='green', linewidth=2)
ax3.set_xscale('log')
ax3.set_xlabel('Sparsity Weight (log scale)')
ax3.set_ylabel('Average Confidence')
ax3.set_title('Permutation Confidence')
ax3.grid(True, alpha=0.3)

# Plot 4: Uniqueness
ax4 = axes[1, 1]
uniqueness_vals = [r['uniqueness'] * 100 for r in results]

ax4.plot(sparsity_vals, uniqueness_vals, marker='o', color='purple', linewidth=2)
ax4.axhline(y=100, color='gray', linestyle='--', alpha=0.5, label='Perfect (100%)')
ax4.set_xscale('log')
ax4.set_xlabel('Sparsity Weight (log scale)')
ax4.set_ylabel('Unique Columns (%)')
ax4.set_title('Permutation Uniqueness')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plot_file = f"sparsity_ablation_plot_{timestamp}.png"
plt.savefig(plot_file, dpi=150)
print(f"💾 Plot saved to: {plot_file}")
plt.show()

# Print summary table
print("\n" + "=" * 80)
print("📊 SUMMARY TABLE")
print("=" * 80)
print(f"{'Sparsity':>10} | {'Soft Acc':>8} | {'Hard Acc':>8} | {'Drop':>6} | {'Rel Drop':>8} | {'Conf':>6} | {'Unique':>7}")
print("-" * 80)
for r in results:
    print(f"{r['sparsity_weight']:>10.2f} | "
          f"{r['soft_accuracy']*100:>7.2f}% | "
          f"{r['hard_accuracy']*100:>7.2f}% | "
          f"{r['accuracy_drop']*100:>5.2f}% | "
          f"{r['relative_drop_pct']:>7.2f}% | "
          f"{r['confidence']:>6.3f} | "
          f"{r['unique_columns']:>3d}/{input_size}")

# Find optimal sparsity weight (minimum relative drop while maintaining >95% soft accuracy)
print("\n" + "=" * 80)
print("🎯 RECOMMENDATIONS")
print("=" * 80)

viable_results = [r for r in results if r['soft_accuracy'] >= acceptance_threshold]
if viable_results:
    best_result = min(viable_results, key=lambda r: r['relative_drop_pct'])
    print(f"✨ Optimal sparsity weight: {best_result['sparsity_weight']}")
    print(f"   Soft accuracy: {best_result['soft_accuracy']*100:.2f}%")
    print(f"   Hard accuracy: {best_result['hard_accuracy']*100:.2f}%")
    print(f"   Relative drop: {best_result['relative_drop_pct']:.2f}%")
    print(f"   Confidence: {best_result['confidence']:.4f}")
else:
    print("⚠️  No configuration achieved >{acceptance_threshold*100:.0f}% soft accuracy")
    best_result = min(results, key=lambda r: r['relative_drop_pct'])
    print(f"💡 Best relative robustness: sparsity_weight={best_result['sparsity_weight']}")
    print(f"   (But only {best_result['soft_accuracy']*100:.2f}% soft accuracy)")

print("\n✅ Ablation study complete!")
