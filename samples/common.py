"""Common utilities for bacon-net samples.

This module provides shared functions for setting up models and running
standard analyses to ensure consistency across all sample implementations.
"""

import torch
import logging
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
        **kwargs: Additional arguments passed to baconNet
        
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
        **kwargs
    )
    
    logging.info(f"🏆 Best accuracy: {best_accuracy * 100:.2f}%")
    return best_model, best_accuracy


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
        X_all: Combined features (train + test)
        Y_all: Combined labels (train + test)
        
    Returns:
        dict: Dictionary with best thresholds and scores for each metric
    """
    print("\n" + "="*60)
    print("THRESHOLD OPTIMIZATION (on combined data)")
    print("="*60)
    
    results = {}
    
    for metric in ['recall', 'precision', 'accuracy', 'f1']:
        best_threshold, best_score = find_best_threshold(model, X_all, Y_all, metric=metric)
        results[metric] = {'threshold': best_threshold, 'score': best_score}
        
        print(f"\nBest threshold for {metric}: {best_threshold:.3f}, Best score: {best_score:.4f}")
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
    baseline_features = pruning_results['baseline_features']
    baseline_feature_names = pruning_results['baseline_feature_names']
    
    if len(baseline_features) > 0:
        print(f"\n📌 Baseline features (not pruned): {baseline_feature_names}")
    
    # Plot accuracy vs. pruning
    plot_title = f"{title_prefix}: Accuracy vs. Number of Features Pruned" if title_prefix else "Accuracy vs. Number of Features Pruned"
    filename = f"{title_prefix.lower().replace(' ', '_')}_pruning.png" if title_prefix else "pruning.png"
    
    plot_feature_pruning_analysis(
        accuracies,
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
    """Run the complete standard analysis pipeline.
    
    Args:
        model: Trained baconNet model
        X_train: Training features
        Y_train: Training labels
        X_test: Test features
        Y_test: Test labels
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
    
    # Combine train and test for comprehensive analysis
    X_all = torch.cat([X_train, X_test], dim=0)
    Y_all = torch.cat([Y_train, Y_test], dim=0)
    
    # Analyze model structure
    analyze_model_structure(model, feature_names)
    
    # Save tree structure to JSON
    if save_tree_json:
        if json_filename is None:
            # Create filename from title_prefix
            safe_prefix = title_prefix.replace(" ", "_").lower() if title_prefix else "model"
            json_filename = f"{safe_prefix}_tree_structure.json"
        save_tree_structure_to_json(model.assembler, json_filename, feature_names)
    
    # Check for overfitting
    analyze_overfitting(model, X_train, Y_train, X_test, Y_test)
    
    # Optimize thresholds
    threshold_results = optimize_thresholds(model, X_all, Y_all)
    
    # Determine threshold for pruning
    if pruning_threshold is None:
        pruning_threshold = threshold_results[pruning_threshold_metric]['threshold']
        pruning_score = threshold_results[pruning_threshold_metric]['score']
        print(f"\n🎯 Using optimized {pruning_threshold_metric.upper()} threshold for pruning: {pruning_threshold:.3f} (score: {pruning_score:.4f})")
    else:
        print(f"\n🎯 Using custom threshold for pruning: {pruning_threshold:.3f}")
    
    # Analyze feature importance (use TEST data for consistency with reported accuracy)
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    print("📊 Using TEST data for pruning analysis (consistent with reported accuracy)")
    pruning_results = analyze_feature_importance(
        model, X_test, Y_test, feature_names,
        title_prefix=title_prefix,
        threshold=pruning_threshold,
        device=device
    )
    
    # Perform growing analysis (incremental feature addition)
    print("\n" + "="*60)
    print("FEATURE GROWING ANALYSIS")
    print("="*60)
    print("📊 Using TEST data for growing analysis")
    from bacon.utils import analyze_feature_importance_with_growing
    growing_results = analyze_feature_importance_with_growing(
        model,
        X_test,
        Y_test,
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
            title=plot_title,
            filename=filename
        )
    
    # Visualize predictions
    best_threshold = threshold_results['accuracy']['threshold']
    visualize_predictions(model, X_all, Y_all, best_threshold)
    
    print("\n✅ Analysis complete!")
    
    return {
        'thresholds': threshold_results,
        'pruning': pruning_results,
        'growing': growing_results
    }
