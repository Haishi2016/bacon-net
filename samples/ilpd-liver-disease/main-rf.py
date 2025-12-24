# Random Forest baseline for ILPD classification
import sys
sys.path.insert(0, '../../')

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from bacon.utils import SigmoidScaler
import pandas as pd
import numpy as np

print("="*60)
print("RANDOM FOREST - ILPD LIVER DISEASE CLASSIFICATION")
print("="*60)

# Fetch ILPD dataset
ilpd = fetch_ucirepo(id=225)

# Extract features and target
X = ilpd.data.features
y = ilpd.data.targets.iloc[:, 0]

print(f"\nDataset shape: {X.shape}")
print(f"Target distribution (original): {y.value_counts().to_dict()}")

# Convert target: 1 (disease) -> 1, 2 (no disease) -> 0
y = (y == 1).astype(int)
print(f"\nConverted to binary: 1=disease, 0=no disease")
print(f"Class distribution: {np.bincount(y)}")  # [no_disease, disease]

# Handle categorical features (Gender) - MUST DO BEFORE FILLING MISSING VALUES
if 'Gender' in X.columns:
    X['Gender'] = X['Gender'].map({'Male': 1, 'Female': 0})

# Handle missing values
if X.isnull().any().any():
    print(f"\nHandling missing values...")
    X = X.fillna(X.median())

df = pd.DataFrame(X)
df['target'] = y

# Separate features and target
X = df.drop(columns=['target'])
y = df['target']

feature_names = X.columns.tolist()

# Train/test split (SAME RANDOM SEED AS BACON)
X_train_df, X_test_df, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Convert DataFrames to numpy arrays for scaling
X_train = X_train_df.values.astype(np.float64)
X_test = X_test_df.values.astype(np.float64)

# Normalize features using SigmoidScaler (SAME AS BACON)
scaler = SigmoidScaler(alpha=4, beta=-1)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print(f"\nTrain samples: {len(X_train)}, Test samples: {len(X_test)}")

# Train Random Forest
print("\nTraining Random Forest...")
rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# Predictions
y_pred_train = rf.predict(X_train)
y_pred_test = rf.predict(X_test)

# Evaluation
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)
test_precision = precision_score(y_test, y_pred_test)
test_recall = recall_score(y_test, y_pred_test)
test_f1 = f1_score(y_test, y_pred_test)

print(f"\n{'='*60}")
print(f"RESULTS")
print(f"{'='*60}")
print(f"Train Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"Test Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"Precision:      {test_precision:.4f}")
print(f"Recall:         {test_recall:.4f}")
print(f"F1-Score:       {test_f1:.4f}")

print(f"\nClassification Report:")
print(classification_report(y_test, y_pred_test, target_names=['No Disease', 'Liver Disease']))

print(f"\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred_test))

# Feature importance
feature_importance = rf.feature_importances_
top_indices = np.argsort(feature_importance)[::-1][:10]

print(f"\n{'='*60}")
print("TOP 10 FEATURE IMPORTANCES")
print(f"{'='*60}")
for i, idx in enumerate(top_indices, 1):
    print(f"  {i:2d}. {feature_names[idx]:30s}: {feature_importance[idx]:.4f}")

print(f"\n{'='*60}")
print(f"✅ FINAL TEST ACCURACY: {test_accuracy:.2%}")
print(f"{'='*60}")
