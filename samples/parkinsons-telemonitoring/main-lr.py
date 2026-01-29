"""Logistic Regression baseline for Parkinson's Telemonitoring Dataset"""
import sys
sys.path.insert(0, '../')

import numpy as np
from sklearn.linear_model import LogisticRegression
from dataset import prepare_data_sklearn
from common import (
    evaluate_sklearn_model, 
    evaluate_sklearn_model_cv,
    print_sklearn_results_summary
)

# Configuration: choose evaluation mode
USE_CROSS_VALIDATION = True  # Set to False for 3-way hold-out split
CV_FOLDS = 5


def main():
    """Test Logistic Regression with multiple configurations"""
    print("="*80)
    print("LOGISTIC REGRESSION BASELINE - PARKINSON'S TELEMONITORING")
    print(f"Evaluation: {'Cross-Validation' if USE_CROSS_VALIDATION else '3-Way Hold-out Split'}")
    print("="*80)
    
    # Load data
    (X_train, X_val, X_test, y_train, y_val, y_test, feature_names, _,
     subject_ids, subjects_train, subjects_val, subjects_test) = prepare_data_sklearn()
    
    if USE_CROSS_VALIDATION:
        # Combine all data for CV
        X_all = np.vstack([X_train, X_val, X_test])
        y_all = np.concatenate([y_train, y_val, y_test])
        # Combine subject IDs for group-based CV
        groups_all = np.concatenate([subjects_train, subjects_val, subjects_test]) if subjects_train is not None else None
        print(f"\n📊 Using {CV_FOLDS}-fold subject-based CV on {len(y_all)} samples ({len(np.unique(groups_all))} subjects)")
    
    print("\n" + "="*80)
    print("TRAINING LOGISTIC REGRESSION MODELS")
    print("="*80)
    
    # Test different solvers
    solvers = ['lbfgs', 'liblinear', 'saga']
    
    results_list = []
    for solver in solvers:
        print(f"\n{'='*80}")
        print(f"Solver: {solver}")
        print(f"{'='*80}")
        
        # Create model with class_weight='balanced'
        model = LogisticRegression(
            solver=solver,
            max_iter=1000,
            class_weight='balanced',
            random_state=42
        )
        
        if USE_CROSS_VALIDATION:
            results = evaluate_sklearn_model_cv(
                model, X_all, y_all, feature_names,
                n_splits=CV_FOLDS,
                class_names=['Low severity', 'High severity'],
                groups=groups_all
            )
        else:
            model.fit(X_train, y_train)
            results = evaluate_sklearn_model(
                model, X_train, y_train, X_val, y_val, X_test, y_test, 
                feature_names, class_names=['Low severity', 'High severity']
            )
        
        results_list.append((f'LR-{solver}', results))
    
    # Print summary
    print_sklearn_results_summary(results_list, metric='f1', use_cv=USE_CROSS_VALIDATION)


if __name__ == '__main__':
    main()
