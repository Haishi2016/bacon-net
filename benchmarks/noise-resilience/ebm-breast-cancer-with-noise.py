# EBM (Explainable Boosting Machine) baseline with noise resilience analysis
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
    from interpret.glassbox import ExplainableBoostingClassifier
except ImportError:
    print("ERROR: interpret not installed. Install with:")
    print("  pip install interpret")
    sys.exit(1)

# Configuration - ADJUST THIS TO TEST DIFFERENT NOISE LEVELS
NOISE_RATIO = 0.0  # 0.0, 0.1, 0.2, 0.3, 0.4, 0.5
IMPORTANCE_THRESHOLD = 0.01  # Features with importance above this are considered "used"

print("="*60)
print("EBM (EXPLAINABLE BOOSTING MACHINE) - BREAST CANCER WITH NOISE")
print("="*60)
print(f"Noise ratio: {NOISE_RATIO:.2%}")
print(f"Importance threshold: {IMPORTANCE_THRESHOLD}\n")

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

# Create DataFrame for EBM
df = pd.DataFrame(X_noisy, columns=feature_names)

# Train/test split (SAME RANDOM STATE AS BACON)
X_train_df, X_test_df, y_train, y_test = train_test_split(
    df, y, test_size=0.2, random_state=42
)

# Normalize with SigmoidScaler (SAME AS BACON)
scaler = SigmoidScaler(alpha=4, beta=-1)
X_train = pd.DataFrame(scaler.fit_transform(X_train_df.values), columns=X_train_df.columns)
X_test = pd.DataFrame(scaler.transform(X_test_df.values), columns=X_test_df.columns)

print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# Train EBM
print("\nTraining EBM (Explainable Boosting Machine)...")
ebm = ExplainableBoostingClassifier(
    random_state=42,
    interactions=10,  # Include feature interactions
    max_bins=256,
    max_interaction_bins=32,
    outer_bags=8,
    inner_bags=0
)

ebm.fit(X_train, y_train)

print("✅ EBM training complete!")

# Predictions
y_pred_train = ebm.predict(X_train)
y_pred_test = ebm.predict(X_test)

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

# Extract global explanation
try:
    from interpret import show
    ebm_global = ebm.explain_global()
    
    # Get feature importances
    feature_importances = {}
    for i, name in enumerate(ebm_global.data()['names']):
        importance = ebm_global.data()['scores'][i]
        feature_importances[name] = importance
    
    # Extract features used (F_r for structural stability)
    features_used = sorted([name for name, importance in feature_importances.items() 
                           if abs(importance) > IMPORTANCE_THRESHOLD and ' x ' not in name])  # Exclude interactions
    
    print(f"\n{'='*60}")
    print(f"FEATURES USED (F_{NOISE_RATIO}, importance > {IMPORTANCE_THRESHOLD})")
    print(f"{'='*60}")
    print(f"Total features used: {len(features_used)}/{len(feature_names)}")
    print("\nFeatures with importance above threshold:")
    for i, feat in enumerate(features_used, 1):
        print(f"  {i:2d}. {feat}")
    
    # Show feature importances
    print(f"\n{'='*60}")
    print("FEATURE IMPORTANCES (sorted)")
    print(f"{'='*60}")
    sorted_importances = sorted(feature_importances.items(), key=lambda x: abs(x[1]), reverse=True)
    for name, importance in sorted_importances[:20]:
        feat_type = "interaction" if ' x ' in name else "main effect"
        print(f"  {name:40s}: {importance:+.4f} ({feat_type})")
    
    # Identify important interactions
    interactions = [(name, importance) for name, importance in sorted_importances 
                   if ' x ' in name and abs(importance) > IMPORTANCE_THRESHOLD]
    
    if interactions:
        print(f"\n{'='*60}")
        print(f"IMPORTANT FEATURE INTERACTIONS (importance > {IMPORTANCE_THRESHOLD})")
        print(f"{'='*60}")
        for name, importance in interactions[:10]:
            print(f"  {name}: {importance:+.4f}")
    
except Exception as e:
    print(f"\nWarning: Could not extract detailed explanations: {e}")
    # Fallback: use feature_importances_ if available
    features_used = []
    feature_importances = {}
    if hasattr(ebm, 'feature_importances_'):
        for i, imp in enumerate(ebm.feature_importances_):
            if abs(imp) > IMPORTANCE_THRESHOLD:
                features_used.append(feature_names[i])
                feature_importances[feature_names[i]] = float(imp)

# Save results for structural stability analysis
results = {
    'noise_ratio': NOISE_RATIO,
    'importance_threshold': IMPORTANCE_THRESHOLD,
    'test_accuracy': test_accuracy,
    'train_accuracy': accuracy_score(y_train, y_pred_train),
    'features_used': features_used,
    'num_features_used': len(features_used),
    'feature_importances': {k: float(v) for k, v in list(sorted(feature_importances.items(), 
                                                                  key=lambda x: abs(x[1]), 
                                                                  reverse=True))[:20]},
    'num_interactions': len([k for k in feature_importances.keys() if ' x ' in k])
}

# Save to file for comparison
import json
with open(f'ebm_results_noise_{NOISE_RATIO:.1f}.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Results saved to ebm_results_noise_{NOISE_RATIO:.1f}.json")
print(f"\n{'='*60}")
print(f"FINAL TEST ACCURACY: {test_accuracy:.2%}")
print(f"{'='*60}")
