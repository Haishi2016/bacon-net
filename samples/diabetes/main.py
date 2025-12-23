# Note: required to import baconNet from local folder
import sys
sys.path.insert(0, '../../')

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
from bacon.baconNet import baconNet
from bacon.visualization import (
    print_tree_structure, 
    visualize_tree_structure, 
    plot_sorted_predictions_with_labels, 
    print_metrics, 
    plot_sorted_predictions_with_errors,
    print_table_structure
)
from bacon.utils import (
    find_best_threshold,
    analyze_bacon_tree_conjunctive_disjunctive,
    SigmoidScaler
)
import logging
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fetch CDC Diabetes Health Indicators dataset
print("Loading CDC Diabetes Health Indicators dataset...")
diabetes = fetch_ucirepo(id=891)

# Extract features and target
X = diabetes.data.features
y = diabetes.data.targets

# Target is Diabetes_binary: 0 = no diabetes, 1 = prediabetes or diabetes
y_binary = y['Diabetes_binary'].values

print(f"Dataset shape: {X.shape}")
print(f"Class distribution: {np.bincount(y_binary)}")

df = pd.DataFrame(X)
df['target'] = y_binary

# Dataset Preview
print("\n" + "="*60)
print("DATASET PREVIEW")
print("="*60)

print("\nFeature Names and Descriptions:")
feature_descriptions = {
    'HighBP': 'High blood pressure (0=no, 1=yes)',
    'HighChol': 'High cholesterol (0=no, 1=yes)',
    'CholCheck': 'Cholesterol check in 5 years (0=no, 1=yes)',
    'BMI': 'Body Mass Index',
    'Smoker': 'Smoked at least 100 cigarettes (0=no, 1=yes)',
    'Stroke': 'Ever had a stroke (0=no, 1=yes)',
    'HeartDiseaseorAttack': 'Coronary heart disease or MI (0=no, 1=yes)',
    'PhysActivity': 'Physical activity in past 30 days (0=no, 1=yes)',
    'Fruits': 'Consume fruit 1+ times per day (0=no, 1=yes)',
    'Veggies': 'Consume vegetables 1+ times per day (0=no, 1=yes)',
    'HvyAlcoholConsump': 'Heavy alcohol consumption (0=no, 1=yes)',
    'AnyHealthcare': 'Have any health care coverage (0=no, 1=yes)',
    'NoDocbcCost': 'Could not see doctor due to cost (0=no, 1=yes)',
    'GenHlth': 'General health (1=excellent to 5=poor)',
    'MentHlth': 'Days of poor mental health (past 30 days)',
    'PhysHlth': 'Days of poor physical health (past 30 days)',
    'DiffWalk': 'Difficulty walking or climbing stairs (0=no, 1=yes)',
    'Sex': 'Sex (0=female, 1=male)',
    'Age': 'Age category (1-13, binned)',
    'Education': 'Education level (1-6)',
    'Income': 'Income level (1-8)'
}

for col in df.columns[:-1]:  # Exclude target
    desc = feature_descriptions.get(col, 'Unknown')
    print(f"  {col:20s} - {desc}")

print("\nFeature Statistics:")
print(df.describe().round(2))

print("\nSample Records (first 5):")
print(df.head())

print("\nClass Distribution:")
print(f"  No Diabetes (0): {(df['target'] == 0).sum()} people ({(df['target'] == 0).sum() / len(df) * 100:.1f}%)")
print(f"  Diabetes (1):    {(df['target'] == 1).sum()} people ({(df['target'] == 1).sum() / len(df) * 100:.1f}%)")

# Separate features and target
X = df.drop(columns=['target'])
y = df['target']

feature_names = X.columns.tolist()

# Note: Most features are already binary or ordinal integers, no one-hot encoding needed
print("\n" + "="*60)
print("FEATURE TYPES")
print("="*60)
print("All features are already numeric (binary or ordinal)")
print("No one-hot encoding required")

# Train/test split using full dataset
X_train_df, X_test_df, y_train_np, y_test_np = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Convert DataFrames to numpy arrays for scaling
X_train_np = X_train_df.values.astype(np.float64)
X_test_np = X_test_df.values.astype(np.float64)

print(f"\nTrain data shape: {X_train_np.shape}, dtype: {X_train_np.dtype}")
print(f"Test data shape: {X_test_np.shape}, dtype: {X_test_np.dtype}")

# Normalize features using SigmoidScaler
scaler = SigmoidScaler(alpha=4, beta=-1)
X_train_np = scaler.fit_transform(X_train_np)
X_test_np = scaler.transform(X_test_np)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train_np, dtype=torch.float32).to(device)
Y_train = torch.tensor(y_train_np.to_numpy().reshape(-1, 1), dtype=torch.float32).to(device)
Y_test = torch.tensor(y_test_np.to_numpy().reshape(-1, 1), dtype=torch.float32).to(device)
X_test = torch.tensor(X_test_np, dtype=torch.float32).to(device)

# Model configuration
freeze_loss_threshold = 0.07
aggregator = 'lsp.half_weight' 
weight_mode = 'fixed'
acceptance_threshold = 0.87
weight_penalty_strength = 1e-3

# Update input size based on features
num_features = len(feature_names)
print(f"\n📊 Model will use {num_features} input features")

from bacon.transformationLayer import IdentityTransformation, NegationTransformation, PeakTransformation
trans = [IdentityTransformation(1), NegationTransformation(1), PeakTransformation(1)]

bacon = baconNet(
    input_size=num_features, 
    freeze_loss_threshold=freeze_loss_threshold, 
    aggregator=aggregator, 
    use_transformation_layer=True,
    transformations=trans,
    loss_amplifier=1000,     
    permutation_initial_temperature=5.0,
    permutation_final_temperature=0.5,
    weight_penalty_strength=weight_penalty_strength, 
    weight_normalization='softmax', 
    use_class_weighting=True,
    weight_mode=weight_mode
)

# Train model
(best_model, best_accuracy) = bacon.find_best_model(
    X_train, Y_train, X_test, Y_test, 
    attempts=10, 
    use_hierarchical_permutation=True,
    hierarchical_bleed_ratio=0.5,
    hierarchical_epochs_per_attempt=2000,  
    hierarchical_group_size=10, 
    acceptance_threshold=acceptance_threshold, 
    loss_weight_perm_sparsity=5.0,
    sinkhorn_iters=200,
    freeze_confidence_threshold=0.95,
    freeze_min_confidence=0.85,
    freeze_loss_threshold=0.09,
    max_epochs=5000
)

print(f"🏆 Best accuracy: {best_accuracy * 100:.2f}%")

# Debug: Check locked_perm for duplicates
print("\n" + "="*60)
print("DEBUG: Checking locked_perm")
print("="*60)
locked_perm = bacon.assembler.locked_perm.tolist()
print(f"locked_perm: {locked_perm}")
print(f"Length: {len(locked_perm)}")
print(f"Unique values: {len(set(locked_perm))}")
print(f"Has duplicates: {len(locked_perm) != len(set(locked_perm))}")
if len(locked_perm) != len(set(locked_perm)):
    from collections import Counter
    counts = Counter(locked_perm)
    duplicates = {k: v for k, v in counts.items() if v > 1}
    print(f"Duplicate indices: {duplicates}")
    for idx, count in duplicates.items():
        print(f"  Feature '{feature_names[idx]}' (index {idx}) appears {count} times")
print("="*60)

# Combine train and test for comprehensive analysis
X_all = torch.cat([X_train, X_test], dim=0)
Y_all = torch.cat([Y_train, Y_test], dim=0)

# Analyze tree structure
analysis = analyze_bacon_tree_conjunctive_disjunctive(bacon.assembler)
print(analysis)

print_tree_structure(bacon.assembler, feature_names)
print_table_structure(bacon.assembler, feature_names)
visualize_tree_structure(bacon.assembler, feature_names)

# Find optimal thresholds for different metrics
print("\n" + "="*60)
print("OVERFITTING CHECK")
print("="*60)

# Evaluate on both train and test sets separately
print("\nTraining Set Performance (threshold=0.5):")
print_metrics(bacon, X_train, Y_train, threshold=0.5)

print("\nTest Set Performance (threshold=0.5):")
print_metrics(bacon, X_test, Y_test, threshold=0.5)

print("\n" + "="*60)
print("THRESHOLD OPTIMIZATION (on combined data)")
print("="*60)

best_threshold_recall, best_score_recall = find_best_threshold(bacon, X_all, Y_all, metric='recall')
print(f"\nBest threshold for recall: {best_threshold_recall:.3f}, Best score: {best_score_recall:.4f}")
print_metrics(bacon, X_all, Y_all, threshold=best_threshold_recall)

best_threshold_precision, best_score_precision = find_best_threshold(bacon, X_all, Y_all, metric='precision')
print(f"\nBest threshold for precision: {best_threshold_precision:.3f}, Best score: {best_score_precision:.4f}")
print_metrics(bacon, X_all, Y_all, threshold=best_threshold_precision)

best_threshold_accuracy, best_score_accuracy = find_best_threshold(bacon, X_all, Y_all, metric='accuracy')
print(f"\nBest threshold for accuracy: {best_threshold_accuracy:.3f}, Best score: {best_score_accuracy:.4f}")
print_metrics(bacon, X_all, Y_all, threshold=best_threshold_accuracy)

best_threshold_f1, best_score_f1 = find_best_threshold(bacon, X_all, Y_all, metric='f1')
print(f"\nBest threshold for F1: {best_threshold_f1:.3f}, Best score: {best_score_f1:.4f}")
print_metrics(bacon, X_all, Y_all, threshold=best_threshold_f1)

# Feature importance analysis through pruning
print("\n" + "="*60)
print("FEATURE IMPORTANCE ANALYSIS")
print("="*60)

# Use the baseline accuracy at threshold 0.5 for consistency
with torch.no_grad():
    baseline_output = bacon.assembler(X_all)
    baseline_accuracy = ((baseline_output > 0.5).float() == Y_all).float().mean().item()

accuracies = [baseline_accuracy]
accuracy_drops = []
feature_contributions = []

for i in range(1, num_features):
    func_eval = bacon.prune_features(i)
    kept_indices = bacon.assembler.locked_perm[i:].tolist()
    removed_feature_idx = bacon.assembler.locked_perm[i - 1].item()
    X_all_pruned = X_all[:, kept_indices]
    
    with torch.no_grad():
        pruned_output = func_eval(X_all_pruned)
        # Use threshold=0.5 consistently throughout pruning analysis
        pruned_accuracy = ((pruned_output > 0.5).float() == Y_all).float().mean().item()
        accuracies.append(pruned_accuracy)
        drop = accuracies[i - 1] - pruned_accuracy
        accuracy_drops.append(drop)
        feature_contributions.append((removed_feature_idx, drop))
        print(f"✅ Accuracy after pruning {i} feature(s): {pruned_accuracy * 100:.2f}%")

sorted_contributions = sorted(feature_contributions, key=lambda x: x[1], reverse=True)

print("\n📊 Feature contributions (sorted by accuracy drop):")
for idx, drop in sorted_contributions:
    relevance = "❌ Low impact" if drop <= 0 else ("🔥 Critical" if drop > 0.05 else "✓ Important")
    print(f"  {feature_names[idx]:25s}: accuracy drop = {drop * 100:.2f}% {relevance}")

# Plot accuracy vs. pruning
plt.figure(figsize=(10, 5))
plt.plot(range(len(accuracies)), [a * 100 for a in accuracies], marker='o', linewidth=2)
plt.title("Diabetes: Accuracy vs. Number of Features Pruned")
plt.xlabel("Number of Features Pruned from Left")
plt.ylabel("Accuracy (%)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('diabetes_pruning.png', dpi=150)
plt.show()

# Visualization
plot_sorted_predictions_with_labels(bacon, X_all, Y_all, threshold=best_threshold_accuracy)
plot_sorted_predictions_with_errors(bacon, X_all, Y_all, threshold=best_threshold_accuracy)

print("\n✅ Analysis complete!")
