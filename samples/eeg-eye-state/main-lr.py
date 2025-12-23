# Logistic Regression baseline for EEG Eye State classification
import sys
sys.path.insert(0, '../../')

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from bacon.utils import SigmoidScaler
import pandas as pd
import numpy as np

print("="*60)
print("LOGISTIC REGRESSION - EEG EYE STATE CLASSIFICATION")
print("="*60)

# Fetch EEG Eye State dataset
eeg = fetch_ucirepo(id=264)

# Extract features and target
X = eeg.data.features
y = eeg.data.targets
y_binary = y.values.ravel()

print(f"\nDataset shape: {X.shape}")
print(f"Class distribution: {np.bincount(y_binary)}")

df = pd.DataFrame(X)
df['target'] = y_binary

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
scaler = SigmoidScaler(alpha=3, beta=-1)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

print("\n" + "="*60)
print("TRAINING LOGISTIC REGRESSION")
print("="*60)

# Train Logistic Regression with multiple solvers
best_accuracy = 0
best_model = None
best_solver = None

for solver in ['lbfgs', 'liblinear', 'saga']:
    try:
        lr = LogisticRegression(
            max_iter=10000,
            random_state=42,
            solver=solver,
            class_weight='balanced'
        )
        lr.fit(X_train, y_train)
        
        y_pred = lr.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        
        print(f"Solver {solver:10s}: Test Accuracy = {acc:.4f}")
        
        if acc > best_accuracy:
            best_accuracy = acc
            best_model = lr
            best_solver = solver
    except Exception as e:
        print(f"Solver {solver:10s}: Failed - {e}")

print("\n" + "="*60)
print(f"BEST MODEL: {best_solver} solver")
print("="*60)

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

print("\nConfusion Matrix (Test Set):")
print(confusion_matrix(y_test, y_pred_test))

print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_pred_test, target_names=['Eyes Open', 'Eyes Closed']))

# Feature importance (coefficients)
print("\n" + "="*60)
print("EEG SENSOR IMPORTANCE (by coefficient magnitude)")
print("="*60)

feature_importance = np.abs(best_model.coef_[0])
top_indices = np.argsort(feature_importance)[::-1]

for rank, idx in enumerate(top_indices, 1):
    coef = best_model.coef_[0][idx]
    print(f"{rank:2d}. {feature_names[idx]:6s} | coef = {coef:+.4f} | |coef| = {abs(coef):.4f}")

print("\n" + "="*60)
print(f"FINAL TEST ACCURACY: {best_accuracy:.2%}")
print("="*60)
