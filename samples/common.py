"""Common utilities for bacon-net samples.

This module provides shared functions for setting up models and running
standard analyses to ensure consistency across all sample implementations.
"""

import torch
import logging
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, average_precision_score
)
from bacon.baconNet import baconNet
from bacon.visualization import (
    print_tree_structure,
    visualize_tree_structure,
    plot_sorted_predictions_with_labels,
    print_metrics,
    plot_sorted_predictions_with_errors,
    print_table_structure,
    plot_feature_pruning_analysis
)
from bacon.utils import (
    find_best_threshold,
    analyze_bacon_tree_conjunctive_disjunctive,
    analyze_feature_importance_with_pruning,
    save_tree_structure_to_json
)


# =============================================================================
# SKLEARN MODEL EVALUATION UTILITIES
# =============================================================================

def find_optimal_threshold_sklearn(y_true, y_prob, metric='f1'):
    """Find optimal classification threshold on validation data.
    
    Args:
        y_true: True labels
        y_prob: Predicted probabilities
        metric: Metric to optimize ('f1', 'accuracy', 'balanced_accuracy')
        
    Returns:
        tuple: (best_threshold, best_score)
    """
    from sklearn.metrics import balanced_accuracy_score
    
    best_threshold = 0.5
    best_score = 0.0
    
    for threshold in np.arange(0.1, 0.9, 0.01):
        y_pred = (y_prob >= threshold).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'accuracy':
            score = accuracy_score(y_true, y_pred)
        elif metric == 'balanced_accuracy':
            score = balanced_accuracy_score(y_true, y_pred)
        else:
            score = f1_score(y_true, y_pred, zero_division=0)
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold, best_score


def evaluate_sklearn_model(model, X_train, y_train, X_val, y_val, X_test, y_test, 
                           feature_names, class_names=None, optimize_threshold=True):
    """Evaluate sklearn model with proper train/val/test separation.
    
    Args:
        model: Trained sklearn model with predict and predict_proba methods
        X_train: Training features (numpy array)
        y_train: Training labels (numpy array)
        X_val: Validation features (numpy array)
        y_val: Validation labels (numpy array)
        X_test: Test features (numpy array)
        y_test: Test labels (numpy array)
        feature_names: List of feature names
        class_names: List of class names for classification report (default: ['Class 0', 'Class 1'])
        optimize_threshold: Whether to optimize threshold on validation set (default: True)
        
    Returns:
        dict: Dictionary with all metrics for train, val, and test sets
    """
    if class_names is None:
        class_names = ['Class 0', 'Class 1']
    
    results = {}
    
    # Get probabilities for all sets
    y_train_prob = model.predict_proba(X_train)[:, 1]
    y_val_prob = model.predict_proba(X_val)[:, 1]
    y_test_prob = model.predict_proba(X_test)[:, 1]
    
    # Find optimal threshold on validation set
    if optimize_threshold:
        optimal_threshold, val_best_score = find_optimal_threshold_sklearn(y_val, y_val_prob, metric='f1')
        print(f"\n🎯 Optimal threshold (from validation F1): {optimal_threshold:.2f}")
    else:
        optimal_threshold = 0.5
    
    results['optimal_threshold'] = optimal_threshold
    
    # Training set evaluation (with optimal threshold)
    y_train_pred = (y_train_prob >= optimal_threshold).astype(int)
    results['train'] = {
        'accuracy': accuracy_score(y_train, y_train_pred),
        'precision': precision_score(y_train, y_train_pred, zero_division=0),
        'recall': recall_score(y_train, y_train_pred, zero_division=0),
        'f1': f1_score(y_train, y_train_pred, zero_division=0),
        'auprc': average_precision_score(y_train, y_train_prob)
    }
    
    # Validation set evaluation (with optimal threshold)
    y_val_pred = (y_val_prob >= optimal_threshold).astype(int)
    results['val'] = {
        'accuracy': accuracy_score(y_val, y_val_pred),
        'precision': precision_score(y_val, y_val_pred, zero_division=0),
        'recall': recall_score(y_val, y_val_pred, zero_division=0),
        'f1': f1_score(y_val, y_val_pred, zero_division=0),
        'auprc': average_precision_score(y_val, y_val_prob)
    }
    
    # Test set evaluation (with optimal threshold from validation)
    y_test_pred = (y_test_prob >= optimal_threshold).astype(int)
    results['test'] = {
        'accuracy': accuracy_score(y_test, y_test_pred),
        'precision': precision_score(y_test, y_test_pred, zero_division=0),
        'recall': recall_score(y_test, y_test_pred, zero_division=0),
        'f1': f1_score(y_test, y_test_pred, zero_division=0),
        'auprc': average_precision_score(y_test, y_test_prob),
        'confusion_matrix': confusion_matrix(y_test, y_test_pred),
        'classification_report': classification_report(y_test, y_test_pred, 
                                                        target_names=class_names,
                                                        output_dict=True)
    }
    
    # Print results
    print("\n" + "="*60)
    print(f"MODEL EVALUATION RESULTS (threshold={optimal_threshold:.2f})")
    print("="*60)
    
    print("\nTraining Set Performance:")
    print(f"  Accuracy:  {results['train']['accuracy']:.4f}")
    print(f"  Precision: {results['train']['precision']:.4f}")
    print(f"  Recall:    {results['train']['recall']:.4f}")
    print(f"  F1 Score:  {results['train']['f1']:.4f}")
    print(f"  AUPRC:     {results['train']['auprc']:.4f}")
    
    print("\nValidation Set Performance:")
    print(f"  Accuracy:  {results['val']['accuracy']:.4f}")
    print(f"  Precision: {results['val']['precision']:.4f}")
    print(f"  Recall:    {results['val']['recall']:.4f}")
    print(f"  F1 Score:  {results['val']['f1']:.4f}")
    print(f"  AUPRC:     {results['val']['auprc']:.4f}")
    
    print("\n" + "="*60)
    print("FINAL TEST SET PERFORMANCE")
    print("="*60)
    print(f"  Accuracy:  {results['test']['accuracy']:.4f}")
    print(f"  Precision: {results['test']['precision']:.4f}")
    print(f"  Recall:    {results['test']['recall']:.4f}")
    print(f"  F1 Score:  {results['test']['f1']:.4f}")
    print(f"  AUPRC:     {results['test']['auprc']:.4f}")
    
    print("\nConfusion Matrix:")
    print(results['test']['confusion_matrix'])
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred, target_names=class_names))
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance = sorted(zip(feature_names, importances), 
                                   key=lambda x: x[1], reverse=True)
        print("\nTop 10 Features by Importance:")
        for i, (feature, importance) in enumerate(feature_importance[:10], 1):
            print(f"  {i:2d}. {feature:20s}: {importance:.4f}")
        results['feature_importance'] = feature_importance
    elif hasattr(model, 'coef_'):
        coef_abs = np.abs(model.coef_[0])
        feature_importance = sorted(zip(feature_names, coef_abs), 
                                   key=lambda x: x[1], reverse=True)
        print("\nTop 10 Features by |Coefficient|:")
        for i, (feature, importance) in enumerate(feature_importance[:10], 1):
            print(f"  {i:2d}. {feature:20s}: {importance:.4f}")
        results['feature_importance'] = feature_importance
    
    return results


def evaluate_sklearn_model_cv(model, X, y, feature_names, n_splits=5, class_names=None, groups=None):
    """Evaluate sklearn model using k-fold cross-validation.
    
    Uses GroupKFold when groups are provided (e.g., subject IDs) to prevent data leakage.
    Uses StratifiedKFold otherwise.
    
    Appropriate for small datasets where hold-out test set would be unreliable.
    
    Args:
        model: Sklearn model instance (will be cloned for each fold)
        X: Features (numpy array)
        y: Labels (numpy array)
        feature_names: List of feature names
        n_splits: Number of CV folds (default: 5)
        class_names: List of class names for classification report
        groups: Group labels for GroupKFold (e.g., subject IDs). If provided, uses group-based CV.
        
    Returns:
        dict: Dictionary with CV metrics and optimal threshold
    """
    from sklearn.model_selection import StratifiedKFold, GroupKFold, cross_val_predict
    from sklearn.base import clone
    
    if class_names is None:
        class_names = ['Class 0', 'Class 1']
    
    results = {}
    
    # Use GroupKFold if groups provided (prevents data leakage with repeated measures)
    if groups is not None:
        n_unique_groups = len(np.unique(groups))
        actual_splits = min(n_splits, n_unique_groups)
        if actual_splits < n_splits:
            print(f"⚠️  Only {n_unique_groups} unique groups, using {actual_splits}-fold CV instead of {n_splits}")
        cv = GroupKFold(n_splits=actual_splits)
        print(f"📊 Using GroupKFold with {actual_splits} folds ({n_unique_groups} unique groups)")
        # Get out-of-fold predictions with groups
        y_prob = cross_val_predict(model, X, y, cv=cv, groups=groups, method='predict_proba')[:, 1]
    else:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        print(f"📊 Using StratifiedKFold with {n_splits} folds")
        # Get out-of-fold predictions
        y_prob = cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1]
    
    # Find optimal threshold on CV predictions
    best_threshold = 0.5
    best_f1 = 0.0
    for threshold in np.arange(0.1, 0.9, 0.01):
        y_pred_temp = (y_prob >= threshold).astype(int)
        f1_temp = f1_score(y, y_pred_temp, zero_division=0)
        if f1_temp > best_f1:
            best_f1 = f1_temp
            best_threshold = threshold
    
    results['optimal_threshold'] = best_threshold
    
    # Apply optimal threshold
    y_pred = (y_prob >= best_threshold).astype(int)
    
    # Calculate metrics (these are CV metrics, used for both 'val' and 'test' keys for compatibility)
    cv_metrics = {
        'accuracy': accuracy_score(y, y_pred),
        'precision': precision_score(y, y_pred, zero_division=0),
        'recall': recall_score(y, y_pred, zero_division=0),
        'f1': f1_score(y, y_pred, zero_division=0),
        'auprc': average_precision_score(y, y_prob),
        'confusion_matrix': confusion_matrix(y, y_pred),
        'classification_report': classification_report(y, y_pred, 
                                                        target_names=class_names,
                                                        output_dict=True,
                                                        zero_division=0)
    }
    
    # Store same metrics for both 'val' and 'test' for API compatibility
    results['cv'] = cv_metrics
    results['val'] = cv_metrics  # For compatibility with print_sklearn_results_summary
    results['test'] = cv_metrics  # For compatibility with print_sklearn_results_summary
    
    # Print results
    print(f"\n🎯 Optimal threshold (from {n_splits}-fold CV): {best_threshold:.2f}")
    
    print("\n" + "="*60)
    print(f"CROSS-VALIDATION RESULTS ({n_splits}-Fold)")
    print("="*60)
    print(f"  Accuracy:  {cv_metrics['accuracy']:.4f}")
    print(f"  Precision: {cv_metrics['precision']:.4f}")
    print(f"  Recall:    {cv_metrics['recall']:.4f}")
    print(f"  F1 Score:  {cv_metrics['f1']:.4f}")
    print(f"  AUPRC:     {cv_metrics['auprc']:.4f}")
    
    print("\nConfusion Matrix:")
    print(cv_metrics['confusion_matrix'])
    
    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=class_names, zero_division=0))
    
    # Train final model on all data for feature importance
    model_final = clone(model)
    model_final.fit(X, y)
    
    if hasattr(model_final, 'feature_importances_'):
        importances = model_final.feature_importances_
        feature_importance = sorted(zip(feature_names, importances), 
                                   key=lambda x: x[1], reverse=True)
        print("\nTop 10 Features by Importance:")
        for i, (feature, importance) in enumerate(feature_importance[:10], 1):
            print(f"  {i:2d}. {feature:20s}: {importance:.4f}")
        results['feature_importance'] = feature_importance
    elif hasattr(model_final, 'coef_'):
        coef_abs = np.abs(model_final.coef_[0])
        feature_importance = sorted(zip(feature_names, coef_abs), 
                                   key=lambda x: x[1], reverse=True)
        print("\nTop 10 Features by |Coefficient|:")
        for i, (feature, importance) in enumerate(feature_importance[:10], 1):
            print(f"  {i:2d}. {feature:20s}: {importance:.4f}")
        results['feature_importance'] = feature_importance
    
    return results


def print_sklearn_results_summary(results_list, metric='f1', use_cv=False):
    """Print summary table of multiple sklearn model results.
    
    Args:
        results_list: List of tuples (model_name, results_dict)
        metric: Metric to use for determining best model (default: 'f1')
        use_cv: Whether results are from cross-validation (affects column headers)
    """
    print("\n" + "="*80)
    print("RESULTS SUMMARY" + (" (Cross-Validation)" if use_cv else ""))
    print("="*80)
    
    if use_cv:
        print(f"{'Model':<25} {'Threshold':<10} {'CV Acc':<12} {'CV F1':<12} {'CV AUPRC':<12}")
        print("-" * 70)
        
        for name, results in results_list:
            threshold = results.get('optimal_threshold', 0.5)
            print(f"{name:<25} "
                  f"{threshold:<10.2f} "
                  f"{results['cv']['accuracy']:<12.4f} "
                  f"{results['cv']['f1']:<12.4f} "
                  f"{results['cv']['auprc']:<12.4f}")
        
        # Find best model
        best_name, best_results = max(results_list, key=lambda x: x[1]['cv'][metric])
        print(f"\nBest Model (by CV {metric}): {best_name}")
        print(f"  Optimal threshold: {best_results.get('optimal_threshold', 0.5):.2f}")
        print(f"  CV {metric}: {best_results['cv'][metric]:.4f}")
        print(f"  CV AUPRC: {best_results['cv']['auprc']:.4f}")
    else:
        print(f"{'Model':<25} {'Threshold':<10} {'Val F1':<12} {'Test Acc':<12} {'Test F1':<12} {'Test AUPRC':<12}")
        print("-" * 80)
        
        for name, results in results_list:
            threshold = results.get('optimal_threshold', 0.5)
            print(f"{name:<25} "
                  f"{threshold:<10.2f} "
                  f"{results['val']['f1']:<12.4f} "
                  f"{results['test']['accuracy']:<12.4f} "
                  f"{results['test']['f1']:<12.4f} "
                  f"{results['test']['auprc']:<12.4f}")
        
        # Find best model based on validation metric
        best_name, best_results = max(results_list, key=lambda x: x[1]['val'][metric])
        print(f"\nBest Model (by val {metric}): {best_name}")
        print(f"  Optimal threshold: {best_results.get('optimal_threshold', 0.5):.2f}")
        print(f"  Validation {metric}: {best_results['val'][metric]:.4f}")
        print(f"  Test {metric}: {best_results['test'][metric]:.4f}")
        print(f"  Test AUPRC: {best_results['test']['auprc']:.4f}")
    
    return best_name, best_results


def create_bacon_model(
    input_size,
    aggregator='lsp.half_weight',
    weight_mode='fixed',
    transformations=None,
    use_transformation_layer=True,
    loss_amplifier=1000,
    weight_normalization='softmax',
    use_class_weighting=True,
    weight_penalty_strength=1e-3,
    permutation_initial_temperature=5.0,
    permutation_final_temperature=0.5,
    training_policy=None,
    # Loss trimming controls
    loss_trim_percentile: float = 0.0,
    loss_trim_mode: str = "none",
    loss_trim_start_epoch: int = 0,
    # Tree layout options
    tree_layout: str = "left",
    # Full tree options (only used when tree_layout="full")
    full_tree_depth: int = None,
    full_tree_temperature: float = 3.0,
    full_tree_final_temperature: float = 0.1,
    full_tree_max_egress: int = None,
    full_tree_shape: str = "triangle",
    loss_weight_full_tree_egress: float = 0.0,
    # Learning rates for different parameter groups
    lr_permutation: float = 0.3,
    lr_transformation: float = 0.5,
    lr_aggregator: float = 0.1,
    lr_other: float = 0.1,
    **kwargs
):
    """Create a baconNet model with standard configuration.
    
    Args:
        input_size: Number of input features
        aggregator: Aggregator type (default: 'lsp.half_weight')
        weight_mode: Weight mode (default: 'fixed')
        transformations: List of transformation layers (default: None)
        use_transformation_layer: Whether to use transformations (default: True)
        loss_amplifier: Loss amplification factor (default: 1000)
        weight_normalization: Weight normalization method (default: 'softmax')
        use_class_weighting: Whether to use class weighting (default: True)
        weight_penalty_strength: Weight penalty strength (default: 1e-3)
        permutation_initial_temperature: Initial temperature (default: 5.0)
        permutation_final_temperature: Final temperature (default: 0.5)
        training_policy: Training policy (e.g., FixedAndnessPolicy) (default: None)
        loss_trim_percentile: Fraction of batch to drop by loss percentile (default: 0.0)
        loss_trim_mode: 'drop_high', 'drop_low', or 'none' (default: 'none')
        loss_trim_start_epoch: Epoch to start trimming (warmup) (default: 0)
        tree_layout: Tree structure layout. Options: "left", "balanced", "paired", "full" (default: "left")
        full_tree_depth: Depth of the fully connected tree when tree_layout="full". (default: None = input_size - 1)
        full_tree_temperature: Initial temperature for full tree sigmoid (default: 3.0)
        full_tree_final_temperature: Final temperature after annealing (default: 0.1)
        full_tree_max_egress: Each source concentrates on top-K destinations (default: None = no constraint)
        full_tree_shape: Shape of the fully connected tree - "triangle" or "square" (default: "triangle")
        loss_weight_full_tree_egress: Weight for full tree egress constraint loss (default: 0.0)
        lr_permutation: Learning rate for permutation layer (default: 0.3)
        lr_transformation: Learning rate for transformation layer (default: 0.5)
        lr_aggregator: Learning rate for aggregator weights (default: 0.1)
        lr_other: Learning rate for other parameters (default: 0.1)
        **kwargs: Additional arguments passed to baconNet (advanced)
        
    Returns:
        baconNet model instance
    """
    from bacon.transformationLayer import IdentityTransformation, NegationTransformation
    
    if transformations is None and use_transformation_layer:
        transformations = [
            IdentityTransformation(1),
            NegationTransformation(1)
        ]
    
    model = baconNet(
        input_size=input_size,
        aggregator=aggregator,
        use_transformation_layer=use_transformation_layer,
        transformations=transformations,
        loss_amplifier=loss_amplifier,
        permutation_initial_temperature=permutation_initial_temperature,
        permutation_final_temperature=permutation_final_temperature,
        weight_penalty_strength=weight_penalty_strength,
        weight_normalization=weight_normalization,
        use_class_weighting=use_class_weighting,
        weight_mode=weight_mode,
        training_policy=training_policy,
        loss_trim_percentile=loss_trim_percentile,
        loss_trim_mode=loss_trim_mode,
        loss_trim_start_epoch=loss_trim_start_epoch,
        # Tree layout options
        tree_layout=tree_layout,
        full_tree_depth=full_tree_depth,
        full_tree_temperature=full_tree_temperature,
        full_tree_final_temperature=full_tree_final_temperature,
        full_tree_max_egress=full_tree_max_egress,
        full_tree_shape=full_tree_shape,
        loss_weight_full_tree_egress=loss_weight_full_tree_egress,
        # Learning rates
        lr_permutation=lr_permutation,
        lr_transformation=lr_transformation,
        lr_aggregator=lr_aggregator,
        lr_other=lr_other,
        **kwargs
    )
    
    return model


def train_bacon_model(
    model,
    X_train,
    Y_train,
    X_test,
    Y_test,
    attempts=10,
    acceptance_threshold=0.80,
    use_hierarchical_permutation=True,
    hierarchical_bleed_ratio=0.5,
    hierarchical_epochs_per_attempt=3000,
    hierarchical_group_size=8,
    loss_weight_perm_sparsity=5.0,
    sinkhorn_iters=200,
    freeze_confidence_threshold=0.95,
    freeze_min_confidence=0.85,
    frozen_training_epochs=200,
    max_epochs=5000,
    binary_threshold = 0.5,
    **kwargs
):
    """Train a baconNet model with standard configuration.
    
    Args:
        model: baconNet model instance
        X_train: Training features
        Y_train: Training labels
        X_test: Test features
        Y_test: Test labels
        attempts: Number of training attempts (default: 10)
        acceptance_threshold: Acceptance threshold (default: 0.80)
        use_hierarchical_permutation: Use hierarchical permutation (default: True)
        hierarchical_bleed_ratio: Bleed ratio (default: 0.5)
        hierarchical_epochs_per_attempt: Epochs per attempt (default: 3000)
        hierarchical_group_size: Group size (default: 8)
        loss_weight_perm_sparsity: Permutation sparsity weight (default: 5.0)
        sinkhorn_iters: Sinkhorn iterations (default: 200)
        freeze_confidence_threshold: Confidence threshold for freezing (default: 0.95)
        freeze_min_confidence: Minimum confidence (default: 0.85)
        max_epochs: Maximum epochs (default: 5000)
        **kwargs: Additional arguments passed to find_best_model
        
    Returns:
        tuple: (best_model, best_accuracy)
    """
    (best_model, best_accuracy) = model.find_best_model(
        X_train, Y_train, X_test, Y_test,
        attempts=attempts,
        use_hierarchical_permutation=use_hierarchical_permutation,
        hierarchical_bleed_ratio=hierarchical_bleed_ratio,
        hierarchical_epochs_per_attempt=hierarchical_epochs_per_attempt,
        hierarchical_group_size=hierarchical_group_size,
        acceptance_threshold=acceptance_threshold,
        loss_weight_perm_sparsity=loss_weight_perm_sparsity,
        sinkhorn_iters=sinkhorn_iters,        
        freeze_confidence_threshold=freeze_confidence_threshold,
        freeze_min_confidence=freeze_min_confidence,
        max_epochs=max_epochs,
        frozen_training_epochs=frozen_training_epochs,
        binary_threshold=binary_threshold,
        **kwargs
    )
    
    logging.info(f"🏆 Best accuracy: {best_accuracy * 100:.2f}%")
    return best_model, best_accuracy


def train_bacon_model_cv(
    model_factory,
    X, Y,
    feature_names,
    n_splits=5,
    balance_training=True,
    binary_threshold=0.5,
    train_kwargs=None,
    device=None,
    groups=None
):
    """Train and evaluate BACON model using cross-validation.
    
    Uses GroupKFold when groups are provided (e.g., subject IDs) to prevent data leakage.
    Uses StratifiedKFold otherwise.
    
    Args:
        model_factory: Callable that returns a new baconNet model instance
        X: Features tensor
        Y: Labels tensor
        feature_names: List of feature names
        n_splits: Number of CV folds (default: 5)
        balance_training: Whether to balance training data (default: True)
        binary_threshold: Classification threshold (default: 0.5)
        train_kwargs: Dict of kwargs for train_bacon_model (default: None)
        device: Torch device (default: None, uses X.device)
        groups: Group labels for GroupKFold (e.g., subject IDs). If provided, uses group-based CV.
        
    Returns:
        dict: CV results including predictions and metrics
    """
    from sklearn.model_selection import StratifiedKFold, GroupKFold
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, average_precision_score
    
    if device is None:
        device = X.device
    
    if train_kwargs is None:
        train_kwargs = {}
    
    # Use GroupKFold if groups provided (prevents data leakage with repeated measures)
    if groups is not None:
        n_unique_groups = len(np.unique(groups))
        actual_splits = min(n_splits, n_unique_groups)
        if actual_splits < n_splits:
            print(f"⚠️  Only {n_unique_groups} unique groups, using {actual_splits}-fold CV instead of {n_splits}")
        cv = GroupKFold(n_splits=actual_splits)
        cv_type = f"GroupKFold ({n_unique_groups} unique groups)"
    else:
        actual_splits = n_splits
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        cv_type = "StratifiedKFold"
    
    # Store out-of-fold predictions
    y_prob_oof = torch.zeros_like(Y)
    fold_results = []
    
    Y_np = Y.cpu().numpy().flatten()
    X_np = X.cpu().numpy()
    
    print(f"\n📊 Starting {actual_splits}-fold cross-validation for BACON model")
    print(f"   CV type: {cv_type}")
    print("="*60)
    
    # Generate fold indices
    if groups is not None:
        fold_iterator = list(cv.split(X_np, Y_np, groups))
    else:
        fold_iterator = list(cv.split(X_np, Y_np))
    
    for fold_idx, (train_idx, val_idx) in enumerate(fold_iterator):
        print(f"\n🔄 Fold {fold_idx + 1}/{actual_splits}")
        print("-"*40)
        
        # Split data
        X_train_fold = X[train_idx]
        Y_train_fold = Y[train_idx]
        X_val_fold = X[val_idx]
        Y_val_fold = Y[val_idx]
        
        # Balance training data if requested
        if balance_training:
            from dataset import balance_data
            X_train_fold, Y_train_fold = balance_data(X_train_fold, Y_train_fold, device)
        
        # Create fresh model for this fold
        model = model_factory()
        
        # Train model (don't save to disk during CV)
        train_kwargs_fold = train_kwargs.copy()
        train_kwargs_fold['save_model'] = False
        train_kwargs_fold['save_path'] = f'./assembler_fold{fold_idx}.pth'
        
        _, fold_acc = train_bacon_model(
            model,
            X_train_fold, Y_train_fold,
            X_val_fold, Y_val_fold,
            binary_threshold=binary_threshold,
            **train_kwargs_fold
        )
        
        # Get predictions for this fold
        model.eval()
        with torch.no_grad():
            val_probs = model.inference_raw(X_val_fold)
            y_prob_oof[val_idx] = val_probs
        
        fold_results.append({
            'fold': fold_idx + 1,
            'accuracy': fold_acc
        })
        
        print(f"   Fold {fold_idx + 1} accuracy: {fold_acc:.4f}")
    
    # Calculate CV metrics
    y_prob_np = y_prob_oof.cpu().numpy().flatten()
    
    # Find optimal threshold
    best_threshold = 0.5
    best_f1 = 0.0
    for threshold in np.arange(0.1, 0.9, 0.01):
        y_pred_temp = (y_prob_np >= threshold).astype(int)
        f1_temp = f1_score(Y_np, y_pred_temp, zero_division=0)
        if f1_temp > best_f1:
            best_f1 = f1_temp
            best_threshold = threshold
    
    y_pred = (y_prob_np >= best_threshold).astype(int)
    
    cv_metrics = {
        'accuracy': accuracy_score(Y_np, y_pred),
        'precision': precision_score(Y_np, y_pred, zero_division=0),
        'recall': recall_score(Y_np, y_pred, zero_division=0),
        'f1': f1_score(Y_np, y_pred, zero_division=0),
        'auprc': average_precision_score(Y_np, y_prob_np)
    }
    
    print("\n" + "="*60)
    print(f"CROSS-VALIDATION RESULTS ({n_splits}-Fold)")
    print("="*60)
    print(f"🎯 Optimal threshold: {best_threshold:.2f}")
    print(f"\nCV Performance:")
    print(f"  Accuracy:  {cv_metrics['accuracy']:.4f}")
    print(f"  Precision: {cv_metrics['precision']:.4f}")
    print(f"  Recall:    {cv_metrics['recall']:.4f}")
    print(f"  F1 Score:  {cv_metrics['f1']:.4f}")
    print(f"  AUPRC:     {cv_metrics['auprc']:.4f}")
    
    print(f"\nPer-fold accuracies: {[r['accuracy'] for r in fold_results]}")
    print(f"Mean fold accuracy: {np.mean([r['accuracy'] for r in fold_results]):.4f}")
    
    return {
        'cv_metrics': cv_metrics,
        'optimal_threshold': best_threshold,
        'fold_results': fold_results,
        'y_prob_oof': y_prob_oof,
        'y_pred': y_pred
    }


def analyze_model_structure(model, feature_names):
    """Analyze and visualize the model's tree structure.
    
    Args:
        model: Trained baconNet model
        feature_names: List of feature names
    """
    print("\n" + "="*60)
    print("MODEL STRUCTURE ANALYSIS")
    print("="*60)
    
    analysis = analyze_bacon_tree_conjunctive_disjunctive(model.assembler)
    print(analysis)
    
    print_tree_structure(model.assembler, feature_names)
    print_table_structure(model.assembler, feature_names)
    visualize_tree_structure(model.assembler, feature_names)


def analyze_overfitting(model, X_train, Y_train, X_test, Y_test, threshold=0.5):
    """Check for overfitting by comparing train and test performance.
    
    Args:
        model: Trained baconNet model
        X_train: Training features
        Y_train: Training labels
        X_test: Test features
        Y_test: Test labels
        threshold: Classification threshold (default: 0.5)
    """
    print("\n" + "="*60)
    print("OVERFITTING CHECK")
    print("="*60)
    
    print(f"\nTraining Set Performance (threshold={threshold}):")
    print_metrics(model, X_train, Y_train, threshold=threshold)
    
    print(f"\nTest Set Performance (threshold={threshold}):")
    print_metrics(model, X_test, Y_test, threshold=threshold)


def optimize_thresholds(model, X_all, Y_all):
    """Find optimal thresholds for different metrics.
    
    Args:
        model: Trained baconNet model
        X_all: Features for threshold optimization
        Y_all: Labels for threshold optimization
        
    Returns:
        dict: Dictionary with best thresholds and scores for each metric
    """
    print("\n" + "="*60)
    print("THRESHOLD OPTIMIZATION")
    print("="*60)
    
    results = {}
    
    for metric in ['recall', 'precision', 'accuracy', 'f1']:
        best_threshold, best_score = find_best_threshold(model, X_all, Y_all, metric=metric)
        results[metric] = {'threshold': best_threshold, 'score': best_score}
        
        print(f"\nBest threshold for {metric}: {best_threshold:.3f}, Best score: {best_score:.4f}")
        print_metrics(model, X_all, Y_all, threshold=best_threshold)
    
    # AUPRC threshold optimization
    # Note: AUPRC is threshold-independent, but we find the threshold that maximizes F1
    # and report that alongside the AUPRC score
    best_threshold, auprc_score = find_best_threshold(model, X_all, Y_all, metric='auprc')
    results['auprc'] = {'threshold': best_threshold, 'score': auprc_score}
    
    print(f"\nBest threshold for AUPRC (F1-optimized): {best_threshold:.3f}, AUPRC score: {auprc_score:.4f}")
    print_metrics(model, X_all, Y_all, threshold=best_threshold)
    
    return results


def analyze_feature_importance(
    model,
    X_all,
    Y_all,
    feature_names,
    title_prefix="",
    threshold=0.5,
    baseline_enabled=True,
    device=None
):
    """Analyze feature importance through pruning with baseline detection.
    
    Args:
        model: Trained baconNet model
        X_all: Features to use for analysis (typically test set)
        Y_all: Labels for the data
        feature_names: List of feature names
        title_prefix: Prefix for plot title (e.g., "Heart Disease")
        threshold: Classification threshold (default: 0.5)
        baseline_enabled: Enable baseline detection (default: True)
        baseline_drop_threshold: Threshold for baseline detection (default: 0.05)
        device: torch device (default: None, uses X_all.device)
        
    Returns:
        dict: Pruning results including accuracies and baseline features
    """
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    if device is None:
        device = X_all.device
    
    pruning_results = analyze_feature_importance_with_pruning(
        model,
        X_all,
        Y_all,
        feature_names,
        threshold=threshold,
        baseline_enabled=baseline_enabled,
        device=device
    )
    
    accuracies = pruning_results['accuracies']
    f1_scores = pruning_results.get('f1_scores')
    auprc_scores = pruning_results.get('auprc_scores')
    baseline_features = pruning_results['baseline_features']
    baseline_feature_names = pruning_results['baseline_feature_names']
    
    if len(baseline_features) > 0:
        print(f"\n📌 Baseline features (not pruned): {baseline_feature_names}")
    
    # Plot accuracy vs. pruning
    plot_title = f"{title_prefix}: Accuracy vs. Number of Features Pruned" if title_prefix else "Accuracy vs. Number of Features Pruned"
    filename = f"{title_prefix.lower().replace(' ', '_')}_pruning.png" if title_prefix else "pruning.png"
    
    plot_feature_pruning_analysis(
        accuracies,
        f1_scores=f1_scores,
        auprc_scores=auprc_scores,
        baseline_features=baseline_features,
        title=plot_title,
        filename=filename
    )
    
    return pruning_results


def visualize_predictions(model, X_all, Y_all, threshold):
    """Visualize model predictions.
    
    Args:
        model: Trained baconNet model
        X_all: Combined features (train + test)
        Y_all: Combined labels (train + test)
        threshold: Classification threshold
    """
    print("\n" + "="*60)
    print("PREDICTION VISUALIZATION")
    print("="*60)
    
    plot_sorted_predictions_with_labels(model, X_all, Y_all, threshold=threshold)
    plot_sorted_predictions_with_errors(model, X_all, Y_all, threshold=threshold)


def run_standard_analysis(
    model,
    X_train,
    Y_train,
    X_val,
    Y_val,
    X_test,
    Y_test,
    feature_names,
    title_prefix="",
    device=None,
    save_tree_json=True,
    json_filename=None,
    pruning_threshold=None,
    pruning_threshold_metric='f1'
):
    """Run the complete standard analysis pipeline with proper train/val/test separation.
    
    Args:
        model: Trained baconNet model
        X_train: Training features (original, not balanced)
        Y_train: Training labels (original, not balanced)
        X_val: Validation features (for threshold optimization)
        Y_val: Validation labels (for threshold optimization)
        X_test: Test features (for final evaluation only)
        Y_test: Test labels (for final evaluation only)
        feature_names: List of feature names
        title_prefix: Prefix for plots (e.g., "Heart Disease")
        device: torch device (default: None)
        save_tree_json: Whether to save tree structure as JSON (default: True)
        json_filename: Filename for JSON export (default: "{title_prefix}_tree_structure.json")
        pruning_threshold: Threshold to use for pruning analysis (default: None, uses optimized threshold)
        pruning_threshold_metric: Metric to optimize threshold for if pruning_threshold is None (default: 'f1')
        
    Returns:
        dict: Analysis results including thresholds and pruning results
    """
    if device is None:
        device = X_train.device
    
    # Analyze model structure
    analyze_model_structure(model, feature_names)
    
    # Save tree structure to JSON
    if save_tree_json:
        if json_filename is None:
            # Create filename from title_prefix
            safe_prefix = title_prefix.replace(" ", "_").lower() if title_prefix else "model"
            json_filename = f"{safe_prefix}_tree_structure.json"
        save_tree_structure_to_json(model.assembler, json_filename, feature_names)
    
    # Check for overfitting: compare train vs val performance
    print("\n" + "="*60)
    print("OVERFITTING CHECK (Train vs Validation)")
    print("="*60)
    print("\nTraining Set Performance (threshold=0.5):")
    print_metrics(model, X_train, Y_train, threshold=0.5)
    print("\nValidation Set Performance (threshold=0.5):")
    print_metrics(model, X_val, Y_val, threshold=0.5)
    
    # Optimize thresholds on VALIDATION set only
    print("\n" + "="*60)
    print("THRESHOLD OPTIMIZATION (on Validation Set)")
    print("="*60)
    print(f"📊 Using validation set ({len(Y_val)} samples) for threshold optimization")
    threshold_results = optimize_thresholds(model, X_val, Y_val)
    
    # Determine threshold for pruning
    if pruning_threshold is None:
        pruning_threshold = threshold_results[pruning_threshold_metric]['threshold']
        pruning_score = threshold_results[pruning_threshold_metric]['score']
        print(f"\n🎯 Using optimized {pruning_threshold_metric.upper()} threshold: {pruning_threshold:.3f} (val score: {pruning_score:.4f})")
    else:
        print(f"\n🎯 Using custom threshold: {pruning_threshold:.3f}")
    
    # FINAL TEST SET EVALUATION - using optimized threshold
    print("\n" + "="*60)
    print("FINAL TEST SET EVALUATION")
    print("="*60)
    print(f"📊 Final evaluation on held-out test set ({len(Y_test)} samples)")
    print(f"   Using threshold optimized on validation set: {pruning_threshold:.3f}")
    print_metrics(model, X_test, Y_test, threshold=pruning_threshold)
    
    # Plot Precision-Recall Curve on test set
    print("\n" + "="*60)
    print("PRECISION-RECALL CURVE (Test Set)")
    print("="*60)
    from bacon.visualization import plot_precision_recall_curve, plot_roc_curve
    pr_title = f"{title_prefix}: Precision-Recall Curve (Test Set)" if title_prefix else "Precision-Recall Curve (Test Set)"
    pr_filename = f"{title_prefix.lower().replace(' ', '_')}_pr_curve.png" if title_prefix else "pr_curve.png"
    plot_precision_recall_curve(
        model, 
        X_test, 
        Y_test, 
        threshold=pruning_threshold,
        title=pr_title,
        filename=pr_filename
    )
    
    # Plot ROC Curve on test set
    print("\n" + "="*60)
    print("ROC CURVE (Test Set)")
    print("="*60)
    roc_title = f"{title_prefix}: ROC Curve (Test Set)" if title_prefix else "ROC Curve (Test Set)"
    roc_filename = f"{title_prefix.lower().replace(' ', '_')}_roc_curve.png" if title_prefix else "roc_curve.png"
    plot_roc_curve(
        model, 
        X_test, 
        Y_test, 
        threshold=pruning_threshold,
        title=roc_title,
        filename=roc_filename
    )
    
    # Analyze feature importance on VALIDATION set (not test)
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS (Validation Set)")
    print("="*60)
    print("📊 Using validation set for pruning analysis")
    pruning_results = analyze_feature_importance(
        model, X_val, Y_val, feature_names,
        title_prefix=title_prefix,
        threshold=pruning_threshold,
        device=device,
        baseline_enabled=True
    )
    
    # Perform growing analysis on validation set
    print("\n" + "="*60)
    print("FEATURE GROWING ANALYSIS (Validation Set)")
    print("="*60)
    print("📊 Using validation set for growing analysis")
    from bacon.utils import analyze_feature_importance_with_growing
    growing_results = analyze_feature_importance_with_growing(
        model,
        X_val,
        Y_val,
        feature_names,
        threshold=pruning_threshold,
        device=device
    )
    
    # Plot growing analysis
    if len(growing_results['accuracies']) > 0:
        plot_title = f"{title_prefix}: Accuracy vs. Number of Features (Growing)" if title_prefix else "Accuracy vs. Number of Features (Growing)"
        filename = f"{title_prefix.lower().replace(' ', '_')}_growing.png" if title_prefix else "growing.png"
        
        from bacon.visualization import plot_feature_growing_analysis
        plot_feature_growing_analysis(
            growing_results['accuracies'],
            f1_scores=growing_results.get('f1_scores'),
            auprc_scores=growing_results.get('auprc_scores'),
            title=plot_title,
            filename=filename
        )
    
    # Visualize predictions on test set
    visualize_predictions(model, X_test, Y_test, pruning_threshold)
    
    print("\n✅ Analysis complete!")
    
    return {
        'thresholds': threshold_results,
        'pruning': pruning_results,
        'growing': growing_results,
        'optimized_threshold': pruning_threshold
    }
