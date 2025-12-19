# Note: required to import baconNet from local folder
import sys
sys.path.insert(0, '../../')

import torch
from bacon.baconNet import baconNet
from bacon.visualization import visualize_tree_structure, print_tree_structure
from bacon.utils import generate_classic_boolean_data
import logging
import os
import numpy as np
import matplotlib.pyplot as plt
from noise_utils import (
    add_uniform_noise,
    compute_nAUDC
)


logging.basicConfig(level=logging.INFO, format='%(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_size = 100
max_epochs = 4000
# noise_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
noise_ratios = [0.0, 0.1]
train_size = 50000
test_size = 20000
seed = 42
clean_accuracy = 0.0
noise_accuracies = []  # Accuracies at different noise levels (for nAUDC)
pruning_accuracies = []  # Accuracies after pruning features (for feature analysis)
sf_scores = []

x, y,  expr_info = generate_classic_boolean_data(input_size, repeat_factor=train_size+test_size, randomize=True, device=device)
feature_names = expr_info['var_names']

for noise_ratio in noise_ratios:
    print(f"\n  Noise ratio: {noise_ratio:.1f}")
    
    X_noisy = add_uniform_noise(x, noise_ratio, seed=seed)

    print(f"➗ Expression: {expr_info['expression_text']}")
    X_train = X_noisy[:train_size]
    Y_train = y[:train_size]
    X_test = X_noisy[train_size:train_size+test_size]
    Y_test = y[train_size:train_size+test_size]

    bacon = baconNet(input_size, aggregator='bool.min_max',
                    weight_mode='fixed', loss_amplifier=1000, normalize_andness=False)

    (best_model, best_accuracy) = bacon.find_best_model(X_train, Y_train, X_test, Y_test, 
            acceptance_threshold=0.95, attempts=5, max_epochs=max_epochs, save_model=False, force_freeze=True)    
    
    noise_accuracies.append(best_accuracy)        
    if len(noise_accuracies) == 1:
        clean_accuracy = best_accuracy
    
    print(f"🏆 Best accuracy: {best_accuracy * 100:.2f}%")
    
    accuracy_drops = []
    feature_contributions = []
    X_all = torch.cat([X_train, X_test], dim=0)
    Y_all = torch.cat([Y_train, Y_test], dim=0)
    
    # Initialize pruning_accuracies with the full model accuracy for this noise level
    pruning_accuracies = [best_accuracy]

    for i in range(1, input_size):
        func_eval = bacon.prune_features(i)
        kept_indices = bacon.assembler.locked_perm[i:].tolist()
        removed_feature_idx = bacon.assembler.locked_perm[i - 1].item()
        X_test_pruned = X_all[:, kept_indices]
        with torch.no_grad():
            pruned_output = func_eval(X_test_pruned)
            pruned_accuracy = (pruned_output.round() == X_all).float().mean().item()
            pruning_accuracies.append(pruned_accuracy)
            drop = pruning_accuracies[i - 1] - pruned_accuracy
            accuracy_drops.append(drop)
            feature_contributions.append((removed_feature_idx, drop))
            print(f"✅ Accuracy after pruning {i} feature(s): {pruned_accuracy * 100:.2f}%")

    sorted_contributions = sorted(feature_contributions, key=lambda x: x[1], reverse=True)

    print("\n📊 Feature contributions (sorted by accuracy drop):")
    for idx, drop in sorted_contributions:
        relevance = "❌ Irrelevant" if drop <= 0 else ""
        print(f"📉 Feature {feature_names[idx]}: accuracy drop = {drop * 100:.2f}% {relevance}")

    plt.figure(figsize=(10, 5))
    plt.plot(range(0, len(pruning_accuracies)), [a * 100 for a in pruning_accuracies], marker='o')
    plt.title("Accuracy vs. Number of Features Pruned")
    plt.xlabel("Number of Features Pruned from Left")
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
# Compute nAUDC
nAUDC = compute_nAUDC(noise_accuracies, noise_ratios)

print("\n[4/4] Results Summary")
print("=" * 60)
print(f"  nAUDC (performance stability): {nAUDC:.4f}")
print(f"  Higher nAUDC = more graceful degradation under noise")
print(f"  Perfect model (no degradation) would have nAUDC = 1.0")

# Visualize results
print("\n  Generating visualizations...")

fig, ax1 = plt.subplots(1, 1, figsize=(8, 5))
    
# Plot: Performance degradation curve
ax1.plot(noise_ratios, [a * 100 for a in noise_accuracies], 'o-', linewidth=2, markersize=8)
ax1.axhline(clean_accuracy * 100, color='gray', linestyle='--', label='Clean accuracy')
ax1.set_xlabel('Noise Ratio', fontsize=12)
ax1.set_ylabel('Accuracy (%)', fontsize=12)
ax1.set_title('Performance Degradation Curve', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend()
ax1.set_ylim([0, 105])
    
# Add nAUDC annotation
ax1.text(0.95, 0.05, f'nAUDC = {nAUDC:.4f}', 
            transform=ax1.transAxes, 
            fontsize=11, 
            verticalalignment='bottom',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()

# print out accuracies for each noise ratio
print("\n  Accuracies by Noise Ratio:") 
for r, a in zip(noise_ratios, noise_accuracies):
    print(f"   Noise Ratio {r:.1f}: Accuracy = {a * 100:.2f}%") 
