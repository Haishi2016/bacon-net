# Decision Tree baseline with noise resilience analysis
import sys
sys.path.insert(0, '../../')

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from bacon.utils import SigmoidScaler
import torch
import pandas as pd
import numpy as np
from noise_utils import add_uniform_noise

# Configuration - ADJUST THIS TO TEST DIFFERENT NOISE LEVELS
NOISE_RATIO = 0.5  # 0.0, 0.1, 0.2, 0.3, 0.4, 0.5

print("="*60)
print("DECISION TREE - BREAST CANCER WITH NOISE")
print("="*60)
print(f"Noise ratio: {NOISE_RATIO:.2%}\n")

# Load dataset
breast_cancer = fetch_ucirepo(id=17)
X = breast_cancer.data.features.iloc[:, 0:30]  # mean values only
feature_names = X.columns.tolist()

# Add noise
X_np = X.values
X_tensor = torch.tensor(X_np, dtype=torch.float32)
X_noisy_tensor, corrupted_indices = add_uniform_noise(X_tensor, NOISE_RATIO, seed=42)
X_noisy = X_noisy_tensor.numpy()

y = LabelEncoder().fit_transform(breast_cancer.data.targets.values.ravel())

# Train/test split (SAME RANDOM STATE AS BACON)
X_train, X_test, y_train, y_test = train_test_split(
    X_noisy, y, test_size=0.2, random_state=42
)

# Normalize with SigmoidScaler (SAME AS BACON)
scaler = SigmoidScaler(alpha=4, beta=-1)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# Train Decision Tree with different max_depth to find best
best_accuracy = 0
best_model = None
best_depth = None

for max_depth in [3, 5, 7, 10, 15, None]:
    dt = DecisionTreeClassifier(
        max_depth=max_depth,
        random_state=42,
        min_samples_split=5,
        min_samples_leaf=2
    )
    dt.fit(X_train, y_train)
    
    y_pred = dt.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"max_depth={str(max_depth):5s}: Test Accuracy = {acc:.4f}")
    
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = dt
        best_depth = max_depth

print(f"\n{'='*60}")
print(f"BEST MODEL: max_depth={best_depth}")
print(f"{'='*60}")

# Evaluate best model
y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)

print("\nTraining Set Performance:")
print(f"  Accuracy : {accuracy_score(y_train, y_pred_train):.4f}")
print(f"  Precision: {precision_score(y_train, y_pred_train, zero_division=0):.4f}")
print(f"  Recall   : {recall_score(y_train, y_pred_train, zero_division=0):.4f}")
print(f"  F1-score : {f1_score(y_train, y_pred_train, zero_division=0):.4f}")

print("\nTest Set Performance:")
print(f"  Accuracy : {accuracy_score(y_test, y_pred_test):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred_test, zero_division=0):.4f}")
print(f"  Recall   : {recall_score(y_test, y_pred_test, zero_division=0):.4f}")
print(f"  F1-score : {f1_score(y_test, y_pred_test, zero_division=0):.4f}")

# Extract features used in splits (F_r for structural stability)
def extract_decision_tree_features(dt_model, feature_names):
    """Extract features that appear in decision tree splits."""
    tree = dt_model.tree_
    features_used = set()
    
    def traverse(node_id):
        if tree.feature[node_id] != -2:  # -2 means leaf node
            features_used.add(tree.feature[node_id])
            traverse(tree.children_left[node_id])
            traverse(tree.children_right[node_id])
    
    traverse(0)
    return sorted([feature_names[i] for i in features_used])

features_used = extract_decision_tree_features(best_model, feature_names)

print(f"\n{'='*60}")
print(f"FEATURES USED IN TREE SPLITS (F_{NOISE_RATIO})")
print(f"{'='*60}")
print(f"Total features used: {len(features_used)}/{len(feature_names)}")
print("\nFeatures appearing in splits:")
for i, feat in enumerate(features_used, 1):
    print(f"  {i:2d}. {feat}")

# Feature importance (from tree structure)
feature_importance = best_model.feature_importances_
important_features = [(feature_names[i], importance) 
                      for i, importance in enumerate(feature_importance) 
                      if importance > 0]
important_features.sort(key=lambda x: x[1], reverse=True)

print(f"\n{'='*60}")
print("FEATURE IMPORTANCE (nonzero)")
print(f"{'='*60}")
for feat, imp in important_features:
    print(f"  {feat:30s}: {imp:.4f}")

# Save results for structural stability analysis
results = {
    'noise_ratio': NOISE_RATIO,
    'test_accuracy': best_accuracy,
    'train_accuracy': accuracy_score(y_train, y_pred_train),
    'features_used': features_used,
    'num_features_used': len(features_used),
    'feature_importances': {feature_names[i]: imp for i, imp in enumerate(feature_importance) if imp > 0}
}

# Save to file for comparison
import json
with open(f'dt_results_noise_{NOISE_RATIO:.1f}.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Results saved to dt_results_noise_{NOISE_RATIO:.1f}.json")
print(f"\n{'='*60}")
print(f"FINAL TEST ACCURACY: {best_accuracy:.2%}")
print(f"{'='*60}")
