"""Logistic Regression baseline for AIDS Clinical Trials Dataset"""
import numpy as np
from sklearn.linear_model import LogisticRegression
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
    print(classification_report(y_test, y_test_pred, target_names=['Alive', 'Died']))
    
    # Feature importance (absolute coefficient values)
    if hasattr(model, 'coef_'):
        coef_abs = np.abs(model.coef_[0])
        feature_importance = sorted(zip(feature_names, coef_abs), key=lambda x: x[1], reverse=True)
        
        print("\nTop 15 Features by |Coefficient|:")
        for i, (feature, importance) in enumerate(feature_importance[:15], 1):
            print(f"  {i:2d}. {feature:20s}: {importance:.4f}")
    
    return test_acc, test_f1

def main():
    """Test Logistic Regression with multiple configurations"""
    print("="*80)
    print("LOGISTIC REGRESSION BASELINE - AIDS CLINICAL TRIALS")
    print("="*80)
    
    # Load data
    X_train, X_test, y_train, y_test, feature_names = prepare_data_sklearn()
    
    print("\n" + "="*80)
    print("TRAINING LOGISTIC REGRESSION MODELS")
    print("="*80)
    
    # Test different solvers
    solvers = ['lbfgs', 'liblinear', 'saga']
    
    results = []
    for solver in solvers:
        print(f"\n{'='*80}")
        print(f"Solver: {solver}")
        print(f"{'='*80}")
        
        # Train model with class_weight='balanced'
        model = LogisticRegression(
            solver=solver,
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        test_acc, test_f1 = evaluate_model(model, X_train, X_test, y_train, y_test, feature_names)
        results.append((solver, test_acc, test_f1))
    
    # Summary
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80)
    print(f"{'Solver':<15} {'Test Accuracy':<15} {'Test F1':<15}")
    print("-" * 80)
    for solver, acc, f1 in results:
        print(f"{solver:<15} {acc:<15.4f} {f1:<15.4f}")
    
    # Best model
    best_solver, best_acc, best_f1 = max(results, key=lambda x: x[2])
    print(f"\nBest Model: {best_solver} (F1: {best_f1:.4f})")

if __name__ == '__main__':
    main()
