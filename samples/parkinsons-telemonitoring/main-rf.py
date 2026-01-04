"""Random Forest baseline for Parkinson's Telemonitoring Dataset"""
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, average_precision_score
from dataset import prepare_data_sklearn

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
    
    # Get probability scores for AUPRC
    y_train_prob = model.predict_proba(X_train)[:, 1]
    y_test_prob = model.predict_proba(X_test)[:, 1]
    train_auprc = average_precision_score(y_train, y_train_prob)
    test_auprc = average_precision_score(y_test, y_test_prob)
    
    print(f"\nTraining Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy:     {test_acc:.4f}")
    print(f"Test Precision:    {test_precision:.4f}")
    print(f"Test Recall:       {test_recall:.4f}")
    print(f"Test F1 Score:     {test_f1:.4f}")
    print(f"Test AUPRC:        {test_auprc:.4f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    print("\nConfusion Matrix:")
    print(cm)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred, target_names=['Low severity', 'High severity']))
    
    # Feature importance
    feature_importance = sorted(zip(feature_names, model.feature_importances_), key=lambda x: x[1], reverse=True)
    
    print("\nTop 10 Features by Importance:")
    for i, (feature, importance) in enumerate(feature_importance[:10], 1):
        print(f"  {i:2d}. {feature:20s}: {importance:.4f}")
    
    return test_acc, test_f1, test_auprc

def main():
    """Test Random Forest with multiple configurations"""
    print("="*80)
    print("RANDOM FOREST BASELINE - PARKINSON'S TELEMONITORING")
    print("="*80)
    
    # Load data
    X_train, X_test, y_train, y_test, feature_names = prepare_data_sklearn()
    
    print("\n" + "="*80)
    print("TRAINING RANDOM FOREST MODELS")
    print("="*80)
    
    # Test different configurations
    configs = [
        {'n_estimators': 100, 'max_depth': None, 'name': 'RF-100-None'},
        {'n_estimators': 200, 'max_depth': None, 'name': 'RF-200-None'},
        {'n_estimators': 100, 'max_depth': 10, 'name': 'RF-100-10'},
        {'n_estimators': 100, 'max_depth': 20, 'name': 'RF-100-20'},
        {'n_estimators': 50, 'max_depth': None, 'name': 'RF-50-None'},
    ]
    
    results = []
    for config in configs:
        print(f"\n{'='*80}")
        print(f"Configuration: {config['name']}")
        print(f"n_estimators={config['n_estimators']}, max_depth={config['max_depth']}")
        print(f"{'='*80}")
        
        # Train model with class_weight='balanced' (similar to BACON's use_class_weighting)
        model = RandomForestClassifier(
            n_estimators=config['n_estimators'],
            max_depth=config['max_depth'],
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(X_train, y_train)
        
        test_acc, test_f1, test_auprc = evaluate_model(model, X_train, X_test, y_train, y_test, feature_names)
        results.append((config['name'], test_acc, test_f1, test_auprc))
    
    # Summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"{'Configuration':<20} {'Test Accuracy':<15} {'Test F1':<15} {'Test AUPRC':<15}")
    print("-" * 80)
    for name, acc, f1, auprc in results:
        print(f"{name:<20} {acc:<15.4f} {f1:<15.4f} {auprc:<15.4f}")
    
    # Best model
    best_name, best_acc, best_f1, best_auprc = max(results, key=lambda x: x[2])
    print(f"\nBest Model: {best_name} (F1: {best_f1:.4f}, AUPRC: {best_auprc:.4f})")

if __name__ == '__main__':
    main()
