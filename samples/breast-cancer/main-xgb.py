"""XGBoost baseline for Breast Cancer Wisconsin Dataset"""
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from dataset import prepare_data_sklearn

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not installed. Install with: pip install xgboost")

def evaluate_model(model, X_train, X_test, y_train, y_test, feature_names):
    """Evaluate model and print comprehensive metrics"""
    # Training predictions
    y_train_pred = model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    
    # Test predictions
    y_test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred)
    test_recall = recall_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    
    print(f"\nTraining Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy:     {test_acc:.4f}")
    print(f"Test Precision:    {test_precision:.4f}")
    print(f"Test Recall:       {test_recall:.4f}")
    print(f"Test F1 Score:     {test_f1:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred, target_names=['Benign', 'Malignant']))
    
    # Feature importance
    feature_importance = sorted(zip(feature_names, model.feature_importances_), key=lambda x: x[1], reverse=True)
    
    print("\nTop 15 Features by Importance:")
    for i, (feature, importance) in enumerate(feature_importance[:15], 1):
        print(f"  {i:2d}. {feature:30s}: {importance:.4f}")
    
    return test_acc, test_f1

def main():
    """Test XGBoost with multiple configurations"""
    if not XGBOOST_AVAILABLE:
        print("XGBoost is not available. Please install it to run this script.")
        return
    
    print("="*80)
    print("XGBOOST BASELINE - BREAST CANCER WISCONSIN")
    print("="*80)
    
    # Load data
    X_train, X_test, y_train, y_test, feature_names = prepare_data_sklearn()
    
    # Calculate scale_pos_weight for class imbalance
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos_weight = neg_count / pos_count
    print(f"\nClass distribution: Benign={neg_count}, Malignant={pos_count}")
    print(f"scale_pos_weight: {scale_pos_weight:.4f}")
    
    print("\n" + "="*80)
    print("TRAINING XGBOOST MODELS")
    print("="*80)
    
    # Test different configurations
    configs = [
        {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1, 'name': 'XGB-100-3-0.1'},
        {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.1, 'name': 'XGB-200-3-0.1'},
        {'n_estimators': 100, 'max_depth': 5, 'learning_rate': 0.1, 'name': 'XGB-100-5-0.1'},
        {'n_estimators': 100, 'max_depth': 6, 'learning_rate': 0.1, 'name': 'XGB-100-6-0.1'},
        {'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.05, 'name': 'XGB-100-3-0.05'},
    ]
    
    results = []
    for config in configs:
        print(f"\n{'='*80}")
        print(f"Configuration: {config['name']}")
        print(f"n_estimators={config['n_estimators']}, max_depth={config['max_depth']}, learning_rate={config['learning_rate']}")
        print(f"{'='*80}")
        
        # Train model
        model = xgb.XGBClassifier(
            n_estimators=config['n_estimators'],
            max_depth=config['max_depth'],
            learning_rate=config['learning_rate'],
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )
        
        model.fit(X_train, y_train)
        
        test_acc, test_f1 = evaluate_model(model, X_train, X_test, y_train, y_test, feature_names)
        results.append((config['name'], test_acc, test_f1))
    
    # Summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"{'Configuration':<20} {'Test Accuracy':<15} {'Test F1':<15}")
    print("-" * 80)
    for name, acc, f1 in results:
        print(f"{name:<20} {acc:<15.4f} {f1:<15.4f}")
    
    # Best model
    best_name, best_acc, best_f1 = max(results, key=lambda x: x[2])
    print(f"\nBest Model: {best_name} (F1: {best_f1:.4f})")

if __name__ == '__main__':
    main()
