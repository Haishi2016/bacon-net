# Logistic Regression baseline for ILPD classification
import sys
sys.path.insert(0, '../../')

from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, average_precision_score
from bacon.utils import SigmoidScaler
import pandas as pd
import numpy as np

print("="*60)
print("LOGISTIC REGRESSION - ILPD LIVER DISEASE CLASSIFICATION")
print("="*60)

# Fetch ILPD dataset
ilpd = fetch_ucirepo(id=225)

# Extract features and target
X = ilpd.data.features.copy()  # Make explicit copy to avoid SettingWithCopyWarning
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

print(f"\nTrain samples (before balancing): {len(X_train)}, Test samples: {len(X_test)}")

# Save original train data for evaluation (before balancing)
y_train_df = y_train.copy()

# Balance training data by upsampling minority class (SAME AS BACON)
def balance_data(X_train, y_train):
    """Balance dataset by upsampling the minority class."""
    y_np = y_train.values if hasattr(y_train, 'values') else y_train
    unique, counts = np.unique(y_np, return_counts=True)
    
    minority_class = unique[np.argmin(counts)]
    majority_class = unique[np.argmax(counts)]
    minority_count = counts.min()
    majority_count = counts.max()
    
    minority_mask = y_np == minority_class
    majority_mask = y_np == majority_class
    
    X_minority = X_train[minority_mask]
    y_minority = y_np[minority_mask]
    X_majority = X_train[majority_mask]
    y_majority = y_np[majority_mask]
    
    # Upsample minority class
    n_samples_needed = majority_count - minority_count
    indices = np.random.RandomState(42).randint(0, len(X_minority), n_samples_needed)
    
    X_minority_upsampled = X_minority[indices]
    y_minority_upsampled = y_minority[indices]
    
    # Combine and shuffle
    X_balanced = np.vstack([X_majority, X_minority, X_minority_upsampled])
    y_balanced = np.concatenate([y_majority, y_minority, y_minority_upsampled])
    
    shuffle_idx = np.random.RandomState(42).permutation(len(X_balanced))
    return X_balanced[shuffle_idx], y_balanced[shuffle_idx]

X_train, y_train = balance_data(X_train, y_train)
print(f"Train samples (after balancing): {len(X_train)}")
print(f"Balanced class distribution: {np.bincount(y_train.astype(int))}")

# Train Logistic Regression
print("\nTraining Logistic Regression...")
lr = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
lr.fit(X_train, y_train)

# Combine train and test for full dataset evaluation (SAME AS BACON post-training analysis)
# Note: Use original (unbalanced) train data for fair evaluation
X_train_orig = X_train_df.values.astype(np.float64)
X_train_orig = scaler.transform(X_train_orig)  # Already fitted
y_train_orig = y_train_df.values if hasattr(y_train_df, 'values') else np.array(y_train_df)

X_all = np.vstack([X_train_orig, X_test])
y_all = np.concatenate([y_train_orig, y_test.values if hasattr(y_test, 'values') else y_test])

print(f"\n📊 Post-training analysis on full dataset ({len(y_all)} samples)")

# Predictions on full dataset
y_pred_all = lr.predict(X_all)
y_prob_all = lr.predict_proba(X_all)[:, 1]

# Evaluation on full dataset
all_accuracy = accuracy_score(y_all, y_pred_all)
all_precision = precision_score(y_all, y_pred_all)
all_recall = recall_score(y_all, y_pred_all)
all_f1 = f1_score(y_all, y_pred_all)
all_auprc = average_precision_score(y_all, y_prob_all)

print(f"\n{'='*60}")
print(f"RESULTS (Full Dataset)")
print(f"{'='*60}")
print(f"Accuracy:  {all_accuracy:.4f} ({all_accuracy*100:.2f}%)")
print(f"Precision: {all_precision:.4f}")
print(f"Recall:    {all_recall:.4f}")
print(f"F1-Score:  {all_f1:.4f}")
print(f"AUPRC:     {all_auprc:.4f}")

print(f"\nClassification Report:")
print(classification_report(y_all, y_pred_all, target_names=['No Disease', 'Liver Disease']))

print(f"\nConfusion Matrix:")
print(confusion_matrix(y_all, y_pred_all))

# Feature importance (absolute coefficients)
feature_importance = np.abs(lr.coef_[0])
top_indices = np.argsort(feature_importance)[::-1][:10]

print(f"\n{'='*60}")
print("TOP 10 FEATURE IMPORTANCES (|coefficients|)")
print(f"{'='*60}")
for i, idx in enumerate(top_indices, 1):
    print(f"  {i:2d}. {feature_names[idx]:30s}: {feature_importance[idx]:.4f}")

print(f"\n{'='*60}")
print(f"✅ FINAL ACCURACY (Full Dataset): {all_accuracy:.2%}")
print(f"{'='*60}")
