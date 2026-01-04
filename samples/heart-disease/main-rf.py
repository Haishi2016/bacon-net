# Random Forest baseline for Heart Disease classification
import sys
sys.path.insert(0, '../../')

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, average_precision_score
import numpy as np
from dataset import prepare_data_sklearn

print("="*60)
print("RANDOM FOREST - HEART DISEASE CLASSIFICATION")
print("="*60)

# Prepare data using common dataset utility
X_train, X_test, y_train, y_test, feature_names = prepare_data_sklearn()

print("\n" + "="*60)
print("TRAINING RANDOM FOREST")
print("="*60)

# Train Random Forest with different configurations
best_accuracy = 0
best_model = None
best_config = None

configs = [
    {'n_estimators': 50, 'max_depth': None},
    {'n_estimators': 100, 'max_depth': None},
    {'n_estimators': 200, 'max_depth': None},
    {'n_estimators': 100, 'max_depth': 10},
    {'n_estimators': 100, 'max_depth': 20},
]

for config in configs:
    rf = RandomForestClassifier(
        n_estimators=config['n_estimators'],
        max_depth=config['max_depth'],
        random_state=42,
        class_weight='balanced',  # Similar to BACON's use_class_weighting
        n_jobs=-1
    )
    rf.fit(X_train, y_train)
    
    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    config_str = f"n_est={config['n_estimators']:3d}, depth={str(config['max_depth']):5s}"
    print(f"{config_str}: Test Accuracy = {acc:.4f}")
    
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = rf
        best_config = config

print("\n" + "="*60)
print(f"BEST MODEL: n_estimators={best_config['n_estimators']}, max_depth={best_config['max_depth']}")
print("="*60)

# Evaluate best model
y_pred_train = best_model.predict(X_train)
y_pred_test = best_model.predict(X_test)

# Get probability scores for AUPRC
y_prob_train = best_model.predict_proba(X_train)[:, 1]
y_prob_test = best_model.predict_proba(X_test)[:, 1]

print("\nTraining Set Performance:")
print(f"  Accuracy : {accuracy_score(y_train, y_pred_train):.4f}")
print(f"  Precision: {precision_score(y_train, y_pred_train, zero_division=0):.4f}")
print(f"  Recall   : {recall_score(y_train, y_pred_train, zero_division=0):.4f}")
print(f"  F1-score : {f1_score(y_train, y_pred_train, zero_division=0):.4f}")
print(f"  AUPRC    : {average_precision_score(y_train, y_prob_train):.4f}")

print("\nTest Set Performance:")
print(f"  Accuracy : {accuracy_score(y_test, y_pred_test):.4f}")
print(f"  Precision: {precision_score(y_test, y_pred_test, zero_division=0):.4f}")
print(f"  Recall   : {recall_score(y_test, y_pred_test, zero_division=0):.4f}")
print(f"  F1-score : {f1_score(y_test, y_pred_test, zero_division=0):.4f}")
print(f"  AUPRC    : {average_precision_score(y_test, y_prob_test):.4f}")

print("\nConfusion Matrix (Test Set):")
print(confusion_matrix(y_test, y_pred_test))

print("\nClassification Report (Test Set):")
print(classification_report(y_test, y_pred_test, target_names=['No Disease', 'Disease']))

# Feature importance
print("\n" + "="*60)
print("TOP 10 MOST IMPORTANT FEATURES")
print("="*60)

feature_importance = best_model.feature_importances_
top_indices = np.argsort(feature_importance)[-10:][::-1]

for rank, idx in enumerate(top_indices, 1):
    importance = feature_importance[idx]
    print(f"{rank:2d}. {feature_names[idx]:25s} | importance = {importance:.4f}")

print("\n" + "="*60)
print(f"FINAL TEST ACCURACY: {best_accuracy:.2%}")
print("="*60)
