# Note: required to import baconNet from local folder
import sys
sys.path.insert(0, '../../')

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
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
    print_table_structure
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
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fetch heart disease dataset (Cleveland database)
heart_disease = fetch_ucirepo(id=45)

# Extract features and target
X = heart_disease.data.features
y = heart_disease.data.targets

# The target 'num' is 0-4, convert to binary: 0 = no disease, 1-4 = disease present
y_binary = (y['num'] > 0).astype(int).values

# Handle missing values (marked as '?')
X = X.replace('?', np.nan)
X = X.apply(pd.to_numeric, errors='coerce')

# Drop rows with missing values
valid_indices = ~X.isnull().any(axis=1)
X = X[valid_indices]
y_binary = y_binary[valid_indices]

print(f"Dataset shape after removing missing values: {X.shape}")
print(f"Class distribution: {np.bincount(y_binary)}")

df = pd.DataFrame(X)
df['target'] = y_binary

# Dataset Preview
print("\n" + "="*60)
print("DATASET PREVIEW")
print("="*60)

print("\nFeature Names and Descriptions:")
feature_descriptions = {
    'age': 'Age in years',
    'sex': 'Sex (1=male, 0=female)',
    'cp': 'Chest pain type (1-4)',
    'trestbps': 'Resting blood pressure (mm Hg)',
    'chol': 'Serum cholesterol (mg/dl)',
    'fbs': 'Fasting blood sugar > 120 mg/dl (1=true)',
    'restecg': 'Resting ECG results (0-2)',
    'thalach': 'Maximum heart rate achieved',
    'exang': 'Exercise induced angina (1=yes)',
    'oldpeak': 'ST depression induced by exercise',
    'slope': 'Slope of peak exercise ST segment (1-3)',
    'ca': 'Number of major vessels (0-3)',
    'thal': 'Thalassemia (3=normal, 6=fixed, 7=reversible)'
}

for col in df.columns[:-1]:  # Exclude target
    desc = feature_descriptions.get(col, 'Unknown')
    print(f"  {col:12s} - {desc}")

print("\nFeature Statistics:")
print(df.describe().round(2))

print("\nSample Records (first 5):")
print(df.head())

print("\nClass Distribution:")
print(f"  No Disease (0): {(df['target'] == 0).sum()} patients ({(df['target'] == 0).sum() / len(df) * 100:.1f}%)")
print(f"  Disease (1):    {(df['target'] == 1).sum()} patients ({(df['target'] == 1).sum() / len(df) * 100:.1f}%)")

# Separate features and target
X = df.drop(columns=['target'])
y = df['target']

# One-hot encode categorical features
print("\n" + "="*60)
print("ENCODING CATEGORICAL FEATURES")
print("="*60)

categorical_features = ['cp', 'restecg', 'slope', 'thal']
print(f"\nCategorical features to encode: {categorical_features}")

# Check unique values for each categorical feature
for col in categorical_features:
    if col in X.columns:
        unique_vals = sorted(X[col].unique())
        print(f"  {col}: {unique_vals}")

# One-hot encode
X_encoded = pd.get_dummies(X, columns=categorical_features, prefix=categorical_features, drop_first=False)

print(f"\nOriginal features: {len(X.columns)}")
print(f"After one-hot encoding: {len(X_encoded.columns)}")
print(f"New feature names: {X_encoded.columns.tolist()}")

feature_names = X_encoded.columns.tolist()
X = X_encoded

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
acceptance_threshold = 1.0
weight_penalty_strength = 1e-3

# Update input size based on encoded features
num_features = len(feature_names)
print(f"\n📊 Model will use {num_features} input features (after one-hot encoding)")

from bacon.transformationLayer import IdentityTransformation, NegationTransformation, ValleyTransformation, PeakTransformation, StepUpTransformation, StepDownTransformation
trans = [IdentityTransformation(1), 
         NegationTransformation(1)]

bacon = baconNet(
    input_size=num_features, 
    freeze_loss_threshold=freeze_loss_threshold, 
    aggregator=aggregator, 
    use_transformation_layer=True,
    transformations=trans,
    loss_amplifier=1000,     
    permutation_initial_temperature=5.0,
    permutation_final_temperature=4.0,
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
    hierarchical_epochs_per_attempt=3000,    
    hierarchical_group_size=8,
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

accuracies = [best_score_accuracy]
accuracy_drops = []
feature_contributions = []

# Prune based on actual number of features after one-hot encoding
for i in range(1, num_features):
    func_eval = bacon.prune_features(i)
    kept_indices = bacon.assembler.locked_perm[i:].tolist()
    removed_feature_idx = bacon.assembler.locked_perm[i - 1].item()
    X_all_pruned = X_all[:, kept_indices]
    
    with torch.no_grad():
        pruned_output = func_eval(X_all_pruned)
        pruned_accuracy = (pruned_output.round() == Y_all).float().mean().item()
        accuracies.append(pruned_accuracy)
        drop = accuracies[i - 1] - pruned_accuracy
        accuracy_drops.append(drop)
        feature_contributions.append((removed_feature_idx, drop))
        print(f"✅ Accuracy after pruning {i} feature(s): {pruned_accuracy * 100:.2f}%")

sorted_contributions = sorted(feature_contributions, key=lambda x: x[1], reverse=True)

print("\n📊 Feature contributions (sorted by accuracy drop):")
for idx, drop in sorted_contributions:
    relevance = "❌ Low impact" if drop <= 0 else ("🔥 Critical" if drop > 0.05 else "✓ Important")
    print(f"  {feature_names[idx]}: accuracy drop = {drop * 100:.2f}% {relevance}")

# Plot accuracy vs. pruning
plt.figure(figsize=(10, 5))
plt.plot(range(len(accuracies)), [a * 100 for a in accuracies], marker='o', linewidth=2)
plt.title("Heart Disease: Accuracy vs. Number of Features Pruned")
plt.xlabel("Number of Features Pruned from Left")
plt.ylabel("Accuracy (%)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('heart_disease_pruning.png', dpi=150)
plt.show()

# Visualization
plot_sorted_predictions_with_labels(bacon, X_all, Y_all, threshold=best_threshold_accuracy)
plot_sorted_predictions_with_errors(bacon, X_all, Y_all, threshold=best_threshold_accuracy)

# Uncomment to analyze specific features
# for feature in feature_names:
#     plot_feature_sensitivity_synthetic(bacon, X_all, feature, feature_names)
#     plot_feature_aggregator_response_aligned(bacon.assembler, X_all, feature, feature_names)

print("\n✅ Analysis complete!")
