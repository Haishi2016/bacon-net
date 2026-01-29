"""Random Forest baseline for Parkinson's Telemonitoring Dataset"""
import sys
sys.path.insert(0, '../')

import numpy as np
from sklearn.ensemble import RandomForestClassifier
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
    """Test Random Forest with multiple configurations"""
    print("="*80)
    print("RANDOM FOREST BASELINE - PARKINSON'S TELEMONITORING")
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
    
    results_list = []
    for config in configs:
        print(f"\n{'='*80}")
        print(f"Configuration: {config['name']}")
        print(f"n_estimators={config['n_estimators']}, max_depth={config['max_depth']}")
        print(f"{'='*80}")
        
        # Create model with class_weight='balanced'
        model = RandomForestClassifier(
            n_estimators=config['n_estimators'],
            max_depth=config['max_depth'],
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
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
        
        results_list.append((config['name'], results))
    
    # Print summary
    print_sklearn_results_summary(results_list, metric='f1', use_cv=USE_CROSS_VALIDATION)


if __name__ == '__main__':
    main()
