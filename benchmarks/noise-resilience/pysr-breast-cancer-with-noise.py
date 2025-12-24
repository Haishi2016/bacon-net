# PySR (Symbolic Regression) baseline with noise resilience analysis
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
import re

try:
    from pysr import PySRRegressor
except ImportError:
    print("ERROR: PySR not installed. Install with:")
    print("  pip install pysr")
    print("  python -m pysr install")
    sys.exit(1)

# Configuration - ADJUST THIS TO TEST DIFFERENT NOISE LEVELS
NOISE_RATIO = 0.0  # 0.0, 0.1, 0.2, 0.3, 0.4, 0.5

print("="*60)
print("PYSR (SYMBOLIC REGRESSION) - BREAST CANCER WITH NOISE")
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

y = LabelEncoder().fit_transform(breast_cancer.data.targets.values.ravel()).astype(float)

# Create DataFrame for PySR
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

# Train PySR (Symbolic Regression)
print("\nTraining PySR (this may take several minutes)...")
print("Searching for symbolic expressions...\n")

model = PySRRegressor(
    niterations=40,  # Increase for better results (but slower)
    binary_operators=["+", "*", "/", "-"],
    unary_operators=["exp", "log", "square", "cube"],
    populations=15,
    population_size=33,
    max_complexity=15,
    timeout_in_seconds=300,  # 5 minute timeout
    parsimony=0.001,
    random_state=42,
    verbosity=1,
    progress=True,
    temp_equation_file=True,
    delete_tempfiles=True
)

model.fit(X_train, y_train)

print("\n✅ PySR training complete!")

# Get predictions (convert to binary classification)
y_pred_train_raw = model.predict(X_train)
y_pred_test_raw = model.predict(X_test)

# Convert to binary using threshold
threshold = 0.5
y_pred_train = (y_pred_train_raw > threshold).astype(int)
y_pred_test = (y_pred_test_raw > threshold).astype(int)

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

# Get the best equation
best_equation = str(model.sympy())

print(f"\n{'='*60}")
print("BEST SYMBOLIC EQUATION")
print(f"{'='*60}")
print(best_equation)

# Extract features used in the equation (F_r for structural stability)
def extract_pysr_features(equation_str, feature_names):
    """Extract features (symbols) present in the symbolic expression."""
    features_used = set()
    
    for feat in feature_names:
        # Check if feature name appears in equation
        # Use word boundary to avoid partial matches
        if re.search(r'\b' + re.escape(feat) + r'\b', equation_str):
            features_used.add(feat)
    
    return sorted(list(features_used))

features_used = extract_pysr_features(best_equation, feature_names)

print(f"\n{'='*60}")
print(f"FEATURES IN SYMBOLIC EXPRESSION (F_{NOISE_RATIO})")
print(f"{'='*60}")
print(f"Total features used: {len(features_used)}/{len(feature_names)}")
print("\nFeatures present in equation:")
for i, feat in enumerate(features_used, 1):
    print(f"  {i:2d}. {feat}")

# Show equation complexity
print(f"\n{'='*60}")
print("EQUATION PROPERTIES")
print(f"{'='*60}")
print(f"Complexity: {model.get_best().get('complexity', 'N/A')}")
print(f"Loss: {model.get_best().get('loss', 'N/A')}")
print(f"Score: {model.get_best().get('score', 'N/A')}")

# Show top candidate equations
print(f"\n{'='*60}")
print("TOP 5 CANDIDATE EQUATIONS")
print(f"{'='*60}")
equations_df = model.equations_
if equations_df is not None and len(equations_df) > 0:
    top_eqs = equations_df.nlargest(5, 'score')
    for i, (idx, row) in enumerate(top_eqs.iterrows(), 1):
        print(f"\n{i}. Complexity: {row['complexity']}, Loss: {row['loss']:.4f}, Score: {row['score']:.4f}")
        print(f"   Equation: {row['equation']}")

# Save results for structural stability analysis
results = {
    'noise_ratio': NOISE_RATIO,
    'test_accuracy': test_accuracy,
    'train_accuracy': accuracy_score(y_train, y_pred_train),
    'features_used': features_used,
    'num_features_used': len(features_used),
    'best_equation': best_equation,
    'complexity': float(model.get_best().get('complexity', 0)),
    'loss': float(model.get_best().get('loss', 0))
}

# Save to file for comparison
import json
with open(f'pysr_results_noise_{NOISE_RATIO:.1f}.json', 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✅ Results saved to pysr_results_noise_{NOISE_RATIO:.1f}.json")
print(f"\n{'='*60}")
print(f"FINAL TEST ACCURACY: {test_accuracy:.2%}")
print(f"{'='*60}")
