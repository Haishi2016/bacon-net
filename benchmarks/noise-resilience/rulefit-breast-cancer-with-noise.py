# RuleFit baseline with noise resilience analysis
import sys
sys.path.insert(0, '../../')

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from bacon.utils import SigmoidScaler
import torch
import pandas as pd
import numpy as np
from noise_utils import add_uniform_noise

try:
    from rulefit import RuleFit
except ImportError:
    print("ERROR: RuleFit not installed. Install with:")
    print("  pip install rulefit")
    print("  Or: pip install git+https://github.com/christophM/rulefit.git")
    sys.exit(1)

# Configuration - ADJUST THIS TO TEST DIFFERENT NOISE LEVELS
NOISE_RATIO = 0.5  # 0.0, 0.1, 0.2, 0.3, 0.4, 0.5
WEIGHT_THRESHOLD = 0.01  # Features with rule weights above this are considered "used"

print("="*60)
print("RULEFIT - BREAST CANCER WITH NOISE")
print("="*60)
print(f"Noise ratio: {NOISE_RATIO:.2%}")
print(f"Weight threshold: {WEIGHT_THRESHOLD}\n")

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

# Create DataFrame for RuleFit
df = pd.DataFrame(X_noisy, columns=feature_names)

# Train/test split (SAME RANDOM STATE AS BACON)
X_train_df, X_test_df, y_train, y_test = train_test_split(
    df, y, test_size=0.2, random_state=42
)

# Normalize with SigmoidScaler (SAME AS BACON)
scaler = SigmoidScaler(alpha=4, beta=-1)
X_train = scaler.fit_transform(X_train_df.values)
X_test = scaler.transform(X_test_df.values)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# Train RuleFit
print("\nTraining RuleFit (this may take a few minutes)...")
rf = RuleFit(
    max_rules=2000,
    rfmode='classify',
    tree_size=4,
    random_state=42,
    exp_rand_tree_size=True
)

rf.fit(X_train, y_train, feature_names=feature_names)

# Predictions
y_pred_train = rf.predict(X_train)
y_pred_test = rf.predict(X_test)

# Get probabilities for threshold-based metrics
y_proba_train = rf.predict_proba(X_train)[:, 1] if hasattr(rf, 'predict_proba') else y_pred_train
y_proba_test = rf.predict_proba(X_test)[:, 1] if hasattr(rf, 'predict_proba') else y_pred_test

print("\nTraining Set Performance:")
print(f"  Accuracy : {accuracy_score(y_train, y_pred_train):.4f}")
print(f"  Precision: {precision_score(y_train, y_pred_train, zero_division=0):.4f}")
print(f"  Recall   : {recall_score(y_train, y_pred_train, zero_division=0):.4f}")
print(f"  F1-score : {f1_score(y_train, y_pred_train, zero_division=0):.4f}")

test_accuracy = accuracy_score(y_test, y_pred_test)
print("\nTest Set Performance:")
print(f"  Accuracy : {test_accuracy:.4f}")
print(f"  Precision: {precision_score(y_test, y_pred_test, zero_division=0):.4f}")
print(f"  Recall   : {recall_score(y_test, y_pred_test, zero_division=0):.4f}")
print(f"  F1-score : {f1_score(y_test, y_pred_test, zero_division=0):.4f}")

# Extract features used in rules (F_r for structural stability)
def extract_rulefit_features(rf_model, feature_names, weight_threshold=0.01):
    """Extract features appearing in nonzero-weight rules."""
    features_used = set()
    
    # Get rules from RuleFit
    rules = rf_model.get_rules()
    
    # Filter rules by weight threshold
    for _, row in rules.iterrows():
        if 'coef' in row and abs(row['coef']) > weight_threshold:
            # Extract feature names from rule
            rule_str = row['rule']
            for feat in feature_names:
                if feat in rule_str:
                    features_used.add(feat)
    
    return sorted(list(features_used))

features_used = extract_rulefit_features(rf, feature_names, WEIGHT_THRESHOLD)

print(f"\n{'='*60}")
print(f"FEATURES USED IN RULES (F_{NOISE_RATIO}, weight > {WEIGHT_THRESHOLD})")
print(f"{'='*60}")
print(f"Total features used: {len(features_used)}/{len(feature_names)}")
print("\nFeatures appearing in nonzero-weight rules:")
for i, feat in enumerate(features_used, 1):
    print(f"  {i:2d}. {feat}")

# Get top rules
rules = rf.get_rules()
rules_sorted = rules.sort_values('importance', ascending=False)

print(f"\n{'='*60}")
print("TOP 15 MOST IMPORTANT RULES")
print(f"{'='*60}")
for idx, row in rules_sorted.head(15).iterrows():
    print(f"  Rule {idx:3d}: {row['rule']:50s} | Importance: {row['importance']:.4f}")

# Calculate feature importance from rules (sum of absolute coefficients)
feature_importance = {}
for feat in feature_names:
    feat_rules = rules[rules['rule'].str.contains(feat, regex=False)]
    total_importance = feat_rules['importance'].abs().sum()
    if total_importance > 0:
        feature_importance[feat] = total_importance

print(f"\n{'='*60}")
print("TOP 15 FEATURE IMPORTANCES (from rule contributions)")
print(f"{'='*60}")
sorted_importance = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
for feat, imp in sorted_importance[:15]:
    print(f"  {feat:30s}: {imp:.4f}")

# Save results for structural stability analysis
results = {
    'noise_ratio': NOISE_RATIO,
    'weight_threshold': WEIGHT_THRESHOLD,
    'test_accuracy': test_accuracy,
    'train_accuracy': accuracy_score(y_train, y_pred_train),
    'features_used': features_used,
    'num_features_used': len(features_used),
    'feature_importances': {k: v for k, v in sorted_importance[:20]},
    'num_rules': len(rules)
}

# Save to file for comparison
import json
with open(f'rulefit_results_noise_{NOISE_RATIO:.1f}.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Results saved to rulefit_results_noise_{NOISE_RATIO:.1f}.json")
print(f"\n{'='*60}")
print(f"FINAL TEST ACCURACY: {test_accuracy:.2%}")
print(f"Number of rules generated: {len(rules)}")
print(f"{'='*60}")
