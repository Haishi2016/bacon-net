"""XGBoost baseline for Parkinson's Telemonitoring Dataset"""
import sys
sys.path.insert(0, '../')

import numpy as np
from dataset import prepare_data_sklearn
from common import (
    evaluate_sklearn_model,
    evaluate_sklearn_model_cv,
    print_sklearn_results_summary
)

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not installed. Install with: pip install xgboost")

# Configuration: choose evaluation mode
USE_CROSS_VALIDATION = True  # Set to False for 3-way hold-out split
CV_FOLDS = 5


def main():
    """Test XGBoost with multiple configurations"""
    if not XGBOOST_AVAILABLE:
        print("XGBoost is not available. Please install it to run this script.")
        return
    
    print("="*80)
    print("XGBOOST BASELINE - PARKINSON'S TELEMONITORING")
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
        # Calculate scale_pos_weight for all data
        neg_count = np.sum(y_all == 0)
        pos_count = np.sum(y_all == 1)
    else:
        # Calculate scale_pos_weight from training data
        neg_count = np.sum(y_train == 0)
        pos_count = np.sum(y_train == 1)
    
    scale_pos_weight = neg_count / pos_count
    print(f"\nClass distribution: Negative={neg_count}, Positive={pos_count}")
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
    
    results_list = []
    for config in configs:
        print(f"\n{'='*80}")
        print(f"Configuration: {config['name']}")
        print(f"n_estimators={config['n_estimators']}, max_depth={config['max_depth']}, learning_rate={config['learning_rate']}")
        print(f"{'='*80}")
        
        # Create model
        model = xgb.XGBClassifier(
            n_estimators=config['n_estimators'],
            max_depth=config['max_depth'],
            learning_rate=config['learning_rate'],
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
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
