# Structural Stability Analysis - Compare feature consistency across noise levels
import json
import os
import numpy as np
import matplotlib.pyplot as plt

def compute_feature_set_consistency(F_0, F_r):
    """
    Compute S_F(r) = |F_0 ∩ F_r| / |F_0 ∪ F_r|
    
    Args:
        F_0: Set of features at noise level 0
        F_r: Set of features at noise level r
    
    Returns:
        Consistency score between 0 and 1
    """
    if len(F_0) == 0 and len(F_r) == 0:
        return 1.0  # Both empty, perfect consistency
    
    intersection = len(F_0.intersection(F_r))
    union = len(F_0.union(F_r))
    
    if union == 0:
        return 1.0
    
    return intersection / union

def load_results(model_type, noise_levels):
    """Load results for a specific model across noise levels."""
    results = {}
    for noise in noise_levels:
        filename = f'{model_type}_results_noise_{noise:.1f}.json'
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                results[noise] = json.load(f)
        else:
            print(f"Warning: {filename} not found")
    return results

def analyze_structural_stability(model_type, noise_levels):
    """Analyze structural stability for a model type."""
    results = load_results(model_type, noise_levels)
    
    if 0.0 not in results:
        print(f"Error: No baseline (noise=0.0) results found for {model_type}")
        return None
    
    F_0 = set(results[0.0]['features_used'])
    print(f"\n{'='*60}")
    print(f"{model_type.upper()} - Structural Stability Analysis")
    print(f"{'='*60}")
    print(f"Baseline features (r=0): {len(F_0)} features")
    print(f"  {sorted(F_0)}")
    
    consistency_scores = []
    accuracies = []
    
    for noise in sorted(results.keys()):
        F_r = set(results[noise]['features_used'])
        consistency = compute_feature_set_consistency(F_0, F_r)
        consistency_scores.append(consistency)
        accuracies.append(results[noise]['test_accuracy'])
        
        print(f"\nNoise level r={noise:.1f}:")
        print(f"  Features used: {len(F_r)}")
        print(f"  S_F({noise:.1f}) = {consistency:.4f}")
        print(f"  Test accuracy: {results[noise]['test_accuracy']:.4f}")
        print(f"  Added features: {sorted(F_r - F_0)}")
        print(f"  Removed features: {sorted(F_0 - F_r)}")
    
    return {
        'noise_levels': sorted(results.keys()),
        'consistency_scores': consistency_scores,
        'accuracies': accuracies,
        'num_features': [len(results[r]['features_used']) for r in sorted(results.keys())]
    }

def plot_comparison(all_results, model_types):
    """Plot structural stability comparison across models."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Structural Consistency vs Noise
    ax1 = axes[0]
    for model_type in model_types:
        if model_type in all_results and all_results[model_type] is not None:
            data = all_results[model_type]
            ax1.plot(data['noise_levels'], data['consistency_scores'], 
                    marker='o', linewidth=2, label=model_type.upper())
    
    ax1.set_xlabel('Noise Level (r)', fontsize=12)
    ax1.set_ylabel('Feature Set Consistency S_F(r)', fontsize=12)
    ax1.set_title('Structural Stability: Feature Consistency vs Noise', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_ylim([0, 1.05])
    
    # Plot 2: Test Accuracy vs Noise
    ax2 = axes[1]
    for model_type in model_types:
        if model_type in all_results and all_results[model_type] is not None:
            data = all_results[model_type]
            ax2.plot(data['noise_levels'], data['accuracies'], 
                    marker='s', linewidth=2, label=model_type.upper())
    
    ax2.set_xlabel('Noise Level (r)', fontsize=12)
    ax2.set_ylabel('Test Accuracy', fontsize=12)
    ax2.set_title('Prediction Performance vs Noise', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Number of Features vs Noise
    ax3 = axes[2]
    for model_type in model_types:
        if model_type in all_results and all_results[model_type] is not None:
            data = all_results[model_type]
            ax3.plot(data['noise_levels'], data['num_features'], 
                    marker='^', linewidth=2, label=model_type.upper())
    
    ax3.set_xlabel('Noise Level (r)', fontsize=12)
    ax3.set_ylabel('Number of Features Used', fontsize=12)
    ax3.set_title('Feature Count vs Noise', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('structural_stability_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\n✅ Comparison plot saved to structural_stability_comparison.png")

def generate_summary_table(all_results, model_types):
    """Generate summary table of results."""
    print(f"\n{'='*80}")
    print("SUMMARY TABLE: Structural Stability Metrics")
    print(f"{'='*80}")
    print(f"{'Model':<10} | {'Noise':<6} | {'S_F(r)':<8} | {'Accuracy':<9} | {'# Features':<12}")
    print(f"{'-'*80}")
    
    for model_type in model_types:
        if model_type in all_results and all_results[model_type] is not None:
            data = all_results[model_type]
            for i, noise in enumerate(data['noise_levels']):
                print(f"{model_type.upper():<10} | {noise:<6.1f} | "
                      f"{data['consistency_scores'][i]:<8.4f} | "
                      f"{data['accuracies'][i]:<9.4f} | "
                      f"{data['num_features'][i]:<12}")

def main():
    # Configuration
    noise_levels = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]
    model_types = ['bacon', 'dt', 'rulefit', 'pysr', 'ebm']
    
    print("="*80)
    print("STRUCTURAL STABILITY ANALYSIS")
    print("Comparing feature consistency across noise levels")
    print("="*80)
    
    # Analyze each model
    all_results = {}
    for model_type in model_types:
        try:
            all_results[model_type] = analyze_structural_stability(model_type, noise_levels)
        except Exception as e:
            print(f"\nError analyzing {model_type}: {e}")
            all_results[model_type] = None
    
    # Generate visualizations and summary
    available_models = [m for m in model_types if m in all_results and all_results[m] is not None]
    
    if available_models:
        generate_summary_table(all_results, available_models)
        plot_comparison(all_results, available_models)
        
        # Compute average consistency across noise levels for each model
        print(f"\n{'='*80}")
        print("AVERAGE CONSISTENCY (across all noise levels)")
        print(f"{'='*80}")
        for model_type in available_models:
            data = all_results[model_type]
            avg_consistency = np.mean(data['consistency_scores'])
            std_consistency = np.std(data['consistency_scores'])
            print(f"{model_type.upper():<10}: {avg_consistency:.4f} ± {std_consistency:.4f}")
    else:
        print("\nNo results found. Please run the individual model scripts first.")
        print("Example: python dt-breast-cancer-with-noise.py")

if __name__ == '__main__':
    main()
