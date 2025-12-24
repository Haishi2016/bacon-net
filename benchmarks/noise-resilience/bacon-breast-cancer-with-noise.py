# Note: required to import baconNet from local folder
import sys
sys.path.insert(0, '../../')

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, LabelEncoder
import torch
from bacon.baconNet import baconNet
from bacon.visualization import (
    print_tree_structure, 
    visualize_tree_structure, 
    plot_sorted_predictions_with_labels, 
    print_metrics, 
    plot_precision_vs_threshold, 
    plot_sorted_predictions_with_errors,
    plot_feature_sensitivity,
    plot_all_feature_correlations,
    plot_feature_correlation,
    overlay_sorted_predictions_and_feature,
    plot_feature_sensitivity_synthetic,
    plot_feature_aggregator_response_aligned,
    print_table_structure,
    plot_gcd_aggregator_3d_minimal
)
from bacon.utils import (
    balance_classes, 
    find_best_threshold,
    analyze_bacon_tree_conjunctive_disjunctive,
    SigmoidScaler
)
import logging
import matplotlib.pyplot as plt
import pandas as pd
from noise_utils import (
    add_uniform_noise,
    compute_nAUDC
)

logging.basicConfig(level=logging.INFO, format='%(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
breast_cancer = fetch_ucirepo(id=17)

# Configuration - ADJUST THIS TO TEST DIFFERENT NOISE LEVELS
noise_ratio = 0.0  # 0.0, 0.1, 0.2, 0.3, 0.4, 0.5

print("="*60)
print("BACON - BREAST CANCER WITH NOISE")
print("="*60)
print(f"Noise ratio: {noise_ratio:.2%}\n")

X = breast_cancer.data.features.iloc[:, 0:30]  # mean values only
feature_names = X.columns.tolist()

# Convert to numpy array for noise addition
X_np = X.values
X_tensor = torch.tensor(X_np, dtype=torch.float32)
X_noisy_tensor, corrupted_indices = add_uniform_noise(X_tensor, noise_ratio, seed=42)
X_noisy = X_noisy_tensor.numpy()

# Display noise information
print("\n" + "="*60)
print("NOISE APPLICATION SUMMARY")
print("="*60)
print(f"Noise ratio: {noise_ratio:.2%}")
print(f"Features per sample to corrupt: {int(noise_ratio * len(feature_names))}/{len(feature_names)}")
print(f"Total samples: {len(X_noisy)}")

if noise_ratio > 0:
    # Show which features were corrupted in first few samples
    print("\nCorrupted features in first 5 samples:")
    for i in range(min(5, len(corrupted_indices))):
        if corrupted_indices[i]:
            corrupted_names = [feature_names[idx] for idx in sorted(corrupted_indices[i])]
            print(f"  Sample {i}: {', '.join(corrupted_names)}")
        else:
            print(f"  Sample {i}: None")
    
    # Overall statistics
    all_corrupted = set()
    for indices in corrupted_indices:
        all_corrupted.update(indices)
    
    if all_corrupted:
        print(f"\nTotal unique features affected across all samples: {len(all_corrupted)}/{len(feature_names)}")
        print(f"Features affected: {', '.join([feature_names[idx] for idx in sorted(all_corrupted)])}")
else:
    print("\nNo noise applied (noise_ratio = 0.0)")

y = LabelEncoder().fit_transform(breast_cancer.data.targets.values.ravel())


df = pd.DataFrame(X_noisy, columns=breast_cancer.data.features.columns[:30])
df['target'] = y

# Balance the dataset
# balanced_df = balance_classes(df, target_col='target', replication_factor=5)

# Separate back
X = df.drop(columns=['target'])
y = df['target']

feature_names = X.columns.tolist()

# Train/test split
# CRITICAL: This random_state must match the one used when the saved model was created
# If there's a mismatch, you'll see test accuracy > train accuracy (data leakage)
X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(X_noisy, y, test_size=0.2, random_state=42)

scaler2 = SigmoidScaler(alpha=4, beta=-1)
X_train_np = scaler2.fit_transform(X_train_np)
X_test_np = scaler2.transform(X_test_np)

# Standardize
# scaler2 = MinMaxScaler()
# X_train_np = scaler2.fit_transform(X_train_np)
# X_test_np = scaler2.transform(X_test_np)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train_np, dtype=torch.float32).to(device)
Y_train = torch.tensor(y_train_np.to_numpy().reshape(-1, 1), dtype=torch.float32).to(device)
Y_test = torch.tensor(y_test_np.to_numpy().reshape(-1, 1), dtype=torch.float32).to(device)
X_test = torch.tensor(X_test_np, dtype=torch.float32).to(device)

freeze_loss_threshold = 0.07
aggregator = 'lsp.half_weight' 
weight_mode = 'fixed'
acceptance_threshold = 0.98
weight_penalty_strength=1e-2 


bacon = baconNet(
    input_size=30, 
    freeze_loss_threshold=freeze_loss_threshold, 
    aggregator=aggregator, 
    loss_amplifier=1000,     
    permutation_initial_temperature=5.0,
    permutation_final_temperature=4.0,
    weight_penalty_strength=weight_penalty_strength, 
    weight_normalization='softmax', 
    use_class_weighting=False,
    weight_mode=weight_mode)
(best_model, best_accuracy) = bacon.find_best_model(X_train, Y_train, X_test, Y_test, 
                                                        attempts=10, 
                                                        use_hierarchical_permutation=True,
                                                        hierarchical_epochs_per_attempt=3000,
                                                        hierarchical_group_size=8,
                                                        hierarchical_bleed_ratio=0.5,
                                                        acceptance_threshold=acceptance_threshold, 
                                                        loss_weight_perm_sparsity=5.0,
                                                        sinkhorn_iters=300,
                                                        freeze_confidence_threshold=0.95,  # Still freeze at very high confidence
                                                        freeze_min_confidence=0.85,        # Or freeze at 0.90 confidence if...
                                                        freeze_loss_threshold=0.09,        # ...loss is below 0.08 * loss_amplifier
                                                        max_epochs=4000)
print(f"🏆 Best accuracy: {best_accuracy * 100:.2f}%")
X_all = torch.cat([X_train, X_test], dim=0)
Y_all = torch.cat([Y_train, Y_test], dim=0)

# plot_gcd_aggregator_3d_minimal(bacon.assembler, 0.5, 1.3, grid_points=20)
# plot_gcd_aggregator_3d_minimal(bacon.assembler, 0.5, 0, grid_points=20)
# plot_gcd_aggregator_3d_minimal(bacon.assembler, 0.5, 1.99, grid_points=20)
# plot_gcd_aggregator_3d_minimal(bacon.assembler, 0.5, -0.99, grid_points=20)
# plot_gcd_aggregator_3d_minimal(bacon.assembler, 0.5, 0.5, grid_points=20)
# plot_gcd_aggregator_3d_minimal(bacon.assembler, 0.3, 0.3, grid_points=20)
# col_idx = df.columns.get_loc('concave_points3')
# X_all[:, col_idx] = 0.8

analysis = analyze_bacon_tree_conjunctive_disjunctive(bacon.assembler)
print(analysis)

# X_normalized = normalize_features(X_all, feature_names)
# X_tensor_normalized = torch.tensor(X_normalized.values, dtype=torch.float32).to(device)

# plot_all_feature_correlations(X_all, feature_names)

print_tree_structure(bacon.assembler, X.columns.tolist())
print_table_structure(bacon.assembler, X.columns.tolist())
visualize_tree_structure(bacon.assembler, X.columns.tolist())

accuracies = []
accuracy_drops = []
feature_contributions = []

# Evaluate on both train and test sets to check for overfitting
print("\n" + "="*60)
print("OVERFITTING CHECK")
print("="*60)
print("\nTraining Set Performance:")
print_metrics(bacon, X_train, Y_train, threshold=0.5)

print("\nTest Set Performance:")
print_metrics(bacon, X_test, Y_test, threshold=0.5)

best_threshold, best_score = find_best_threshold(bacon, X_test, Y_test, metric='recall')
print(f"Best threshold for recall: {best_threshold:.2f}, Best score: {best_score:.4f}")
print_metrics(bacon, X_test, Y_test, threshold=best_threshold)
print_metrics(bacon, X_test, Y_test, threshold=0.02)
print_metrics(bacon, X_test, Y_test, threshold=0.1)


best_threshold, best_score = find_best_threshold(bacon, X_test, Y_test, metric='precision')
print(f"Best threshold for precision: {best_threshold:.2f}, Best score: {best_score:.4f}")
print_metrics(bacon, X_test, Y_test, threshold=best_threshold)

best_threshold, best_score = find_best_threshold(bacon, X_test, Y_test, metric='accuracy')
print(f"Best threshold for accuracy: {best_threshold:.2f}, Best score: {best_score:.4f}")
print_metrics(bacon, X_test, Y_test, threshold=best_threshold)

bacon.evaluate(X_test, Y_test, threshold=best_threshold)
accuracies.append(best_score)

for i in range(1, 30):
    func_eval = bacon.prune_features(i)
    kept_indices = bacon.assembler.locked_perm[i:].tolist()
    removed_feature_idx = bacon.assembler.locked_perm[i - 1].item()
    X_test_pruned = X_all[:, kept_indices]
    with torch.no_grad():
        pruned_output = func_eval(X_test_pruned)
        pruned_accuracy = (pruned_output.round() == Y_all).float().mean().item()
        accuracies.append(pruned_accuracy)
        drop = accuracies[i - 1] - pruned_accuracy
        accuracy_drops.append(drop)
        feature_contributions.append((removed_feature_idx, drop))
        print(f"✅ Accuracy after pruning {i} feature(s): {pruned_accuracy * 100:.2f}%")

sorted_contributions = sorted(feature_contributions, key=lambda x: x[1], reverse=True)

print("\n📊 Feature contributions (sorted by accuracy drop):")
for idx, drop in sorted_contributions:
    relevance = "❌ Irrelevant" if drop <= 0 else ""
    print(f"📉 Feature {feature_names[idx]}: accuracy drop = {drop * 100:.2f}% {relevance}")

plt.figure(figsize=(10, 5))
plt.plot(range(1, len(accuracies) + 1), [a * 100 for a in accuracies], marker='o')
plt.title("Accuracy vs. Number of Features Pruned")
plt.xlabel("Number of Features Pruned from Left")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.tight_layout()
plt.savefig('bacon_pruning.png', dpi=150)
plt.show()

plot_sorted_predictions_with_labels(bacon, X_all, Y_all, threshold=best_threshold)
plot_sorted_predictions_with_errors(bacon, X_all, Y_all, threshold=best_threshold)

# Extract features used by BACON for structural stability analysis
features_used_indices = bacon.assembler.locked_perm.cpu().numpy()
features_used = [feature_names[idx] for idx in features_used_indices]

print(f"\n{'='*60}")
print(f"FEATURES USED BY BACON (F_{noise_ratio})")
print(f"{'='*60}")
print(f"Total features used: {len(features_used)}/{len(feature_names)}")
print("\nFeatures in tree (ordered by importance):")
for i, feat in enumerate(features_used, 1):
    print(f"  {i:2d}. {feat}")

# Save results for structural stability analysis
import json
results = {
    'noise_ratio': noise_ratio,
    'test_accuracy': best_score,
    'train_accuracy': accuracy_score(y_train_np, (bacon.assembler(X_train) > 0.5).cpu().numpy()),
    'features_used': features_used,
    'num_features_used': len(features_used),
    'feature_contributions': {feature_names[idx]: float(drop) for idx, drop in sorted_contributions}
}

with open(f'bacon_results_noise_{noise_ratio:.1f}.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Results saved to bacon_results_noise_{noise_ratio:.1f}.json")
print(f"\n{'='*60}")
print(f"FINAL TEST ACCURACY: {best_score:.2%}")
print(f"{'='*60}")

# for feature in feature_names:
#     plot_feature_sensitivity_synthetic(bacon, X_all, feature, feature_names)

# plot_feature_aggregator_response_aligned(bacon.assembler, X_all, 'concave_points1', feature_names)
# plot_feature_aggregator_response_aligned(bacon.assembler, X_all, 'concave_points3', feature_names)
# plot_feature_aggregator_response_aligned(bacon.assembler, X_all, 'area3', feature_names)
# plot_feature_aggregator_response_aligned(bacon.assembler, X_all, 'radius2', feature_names)
# plot_feature_aggregator_response_aligned(bacon.assembler, X_all, 'area2', feature_names)
# plot_feature_aggregator_response_aligned(bacon.assembler, X_all, 'concavity1', feature_names)
# plot_feature_aggregator_response_aligned(bacon.assembler, X_all, 'concavity3', feature_names)
# plot_feature_aggregator_response_aligned(bacon.assembler, X_all, 'perimeter3', feature_names)
# plot_feature_sensitivity_synthetic(bacon, X_all, 'concave_points1', feature_names)
# plot_feature_sensitivity_synthetic(bacon, X_all, 'concave_points3', feature_names)
# plot_feature_sensitivity_synthetic(bacon, X_all, 'area3', feature_names)
# plot_feature_sensitivity_synthetic(bacon, X_all, 'radius2', feature_names)
# plot_feature_sensitivity_synthetic(bacon, X_all, 'area2', feature_names)
# plot_feature_sensitivity_synthetic(bacon, X_all, 'concavity1', feature_names)
# plot_feature_sensitivity_synthetic(bacon, X_all, 'concavity3', feature_names)
# plot_feature_sensitivity_synthetic(bacon, X_all, 'perimeter3', feature_names)