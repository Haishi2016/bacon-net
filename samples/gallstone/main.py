# Note: required to import baconNet from local folder
import sys
sys.path.insert(0, '../../')

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
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

# Fetch Gallstone dataset
print("Loading Gallstone dataset...")
gallstone = fetch_ucirepo(id=1150)

# Extract features and target
X = gallstone.data.features
y = gallstone.data.targets

# Target is Gallstone: 0 = no gallstone, 1 = gallstone disease
y_binary = y.values.ravel()

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
    'Age': 'Age in years',
    'Gender': 'Gender (0=female, 1=male)',
    'Height': 'Height in cm',
    'Weight': 'Weight in kg',
    'BMI': 'Body Mass Index',
    'TBW': 'Total body water (L)',
    'ECW': 'Extracellular water (L)',
    'ICW': 'Intracellular water (L)',
    'Muscle_mass': 'Muscle mass (kg)',
    'Fat_mass': 'Fat mass (kg)',
    'Protein': 'Protein (kg)',
    'VFA': 'Visceral fat area (cm²)',
    'Hepatic_fat': 'Hepatic fat (%)',
    'Glucose': 'Blood glucose (mg/dL)',
    'Total_cholesterol': 'Total cholesterol (mg/dL)',
    'HDL': 'High-density lipoprotein (mg/dL)',
    'LDL': 'Low-density lipoprotein (mg/dL)',
    'Triglycerides': 'Triglycerides (mg/dL)',
    'AST': 'Aspartate aminotransferase (U/L)',
    'ALT': 'Alanine aminotransferase (U/L)',
    'ALP': 'Alkaline phosphatase (U/L)',
    'Creatinine': 'Creatinine (mg/dL)',
    'GFR': 'Glomerular filtration rate',
    'CRP': 'C-reactive protein (mg/L)',
    'Hemoglobin': 'Hemoglobin (g/dL)',
    'Vitamin_D': 'Vitamin D (ng/mL)'
}

for col in df.columns[:-1]:  # Exclude target
    desc = feature_descriptions.get(col, 'Bioimpedance/Laboratory measure')
    print(f"  {col:20s} - {desc}")

print("\nFeature Statistics:")
print(df.describe().round(2))

print("\nSample Records (first 5):")
print(df.head())

print("\nClass Distribution:")
print(f"  No Gallstone (0): {(df['target'] == 0).sum()} people ({(df['target'] == 0).sum() / len(df) * 100:.1f}%)")
print(f"  Gallstone (1):    {(df['target'] == 1).sum()} people ({(df['target'] == 1).sum() / len(df) * 100:.1f}%)")

# Separate features and target
X = df.drop(columns=['target'])
y = df['target']

feature_names = X.columns.tolist()

# Note: All features are already numeric, no one-hot encoding needed
print("\n" + "="*60)
print("FEATURE TYPES")
print("="*60)
print("All features are already numeric (continuous)")
print("No one-hot encoding required")

# Train/test split
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
acceptance_threshold = 0.90
weight_penalty_strength = 1e-3

# Update input size based on features
num_features = len(feature_names)
print(f"\n📊 Model will use {num_features} input features")

from bacon.transformationLayer import IdentityTransformation, NegationTransformation
trans = [IdentityTransformation(1), NegationTransformation(1)]

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
    use_hierarchical_permutation=False,
    acceptance_threshold=acceptance_threshold, 
    loss_weight_perm_sparsity=5.0,
    sinkhorn_iters=200,
    freeze_confidence_threshold=0.95,
    freeze_min_confidence=0.85,
    freeze_loss_threshold=0.09,
    max_epochs=5000
)

print(f"🏆 Best accuracy: {best_accuracy * 100:.2f}%")

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
plt.title("Gallstone: Accuracy vs. Number of Features Pruned")
plt.xlabel("Number of Features Pruned from Left")
plt.ylabel("Accuracy (%)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('gallstone_pruning.png', dpi=150)
plt.show()

# Visualization
plot_sorted_predictions_with_labels(bacon, X_all, Y_all, threshold=best_threshold_accuracy)
plot_sorted_predictions_with_errors(bacon, X_all, Y_all, threshold=best_threshold_accuracy)

print("\n✅ Analysis complete!")
