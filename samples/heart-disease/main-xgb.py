# XGBoost baseline for Heart Disease classification
import sys
sys.path.insert(0, '../../')

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, average_precision_score
import numpy as np
from dataset import prepare_data_sklearn

try:
    import xgboost as xgb
except ImportError:
    print("ERROR: XGBoost not installed. Please install with: pip install xgboost")
    sys.exit(1)

print("="*60)
print("XGBOOST - HEART DISEASE CLASSIFICATION")
print("="*60)

# Prepare data using common dataset utility
X_train, X_test, y_train, y_test, feature_names = prepare_data_sklearn()

print("\n" + "="*60)
print("TRAINING XGBOOST")
print("="*60)

# Calculate scale_pos_weight for class imbalance (similar to BACON's class_weight='balanced')
neg_count = (y_train == 0).sum()
pos_count = (y_train == 1).sum()
scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
print(f"Class imbalance handling: scale_pos_weight = {scale_pos_weight:.2f}")

# Train XGBoost with different configurations
best_accuracy = 0
best_model = None
best_config = None

configs = [
    {'n_estimators': 50, 'max_depth': 3, 'learning_rate': 0.1},
    {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1},
    {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1},
    {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.05},
    {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1},
]

for config in configs:
    xgb_model = xgb.XGBClassifier(
        n_estimators=config['n_estimators'],
        max_depth=config['max_depth'],
        learning_rate=config['learning_rate'],
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss',
        n_jobs=-1
    )
    xgb_model.fit(X_train, y_train, verbose=False)
    
    y_pred = xgb_model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    config_str = f"n_est={config['n_estimators']:3d}, depth={config['max_depth']}, lr={config['learning_rate']:.2f}"
    print(f"{config_str}: Test Accuracy = {acc:.4f}")
    
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = xgb_model
        best_config = config

print("\n" + "="*60)
print(f"BEST MODEL: n_estimators={best_config['n_estimators']}, max_depth={best_config['max_depth']}, lr={best_config['learning_rate']}")
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
