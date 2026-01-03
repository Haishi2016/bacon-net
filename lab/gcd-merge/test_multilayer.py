"""
Test multi-layer GCD merging with error accumulation analysis.

This tests merging n-layer cascaded GCD operations into a single GCD,
quantifying how errors accumulate across multiple layers.
"""

import torch
import numpy as np
from scipy.optimize import minimize
from datetime import datetime
import json

from bacon.aggregators.lsp import FullWeightAggregator


def GCD_2(x, y, w, a):
    """Wrapper for GCD_2 aggregation using FullWeightAggregator."""
    aggregator = FullWeightAggregator()
    w0 = 1 - w
    w1 = w
    return aggregator.aggregate_tensor(x, y, a, w0, w1)


def n_layer_cascade(x, y, w_list, a_list):
    """
    Compute n-layer cascaded GCD.
    
    Args:
        x, y: Input tensors
        w_list: List of n weights [w1, w2, ..., wn]
        a_list: List of n+1 alphas [a1, a2, ..., a_{n+1}]
                First n for inner GCDs, last one for final outer GCD
    
    Returns:
        Result of GCD(GCD(...GCD(GCD(x,y,w1,a1), 1, a2)...), 1, a_{n+1})
    """
    assert len(w_list) == len(a_list) - 1, f"Need n weights and n+1 alphas, got {len(w_list)} weights and {len(a_list)} alphas"
    
    result = x
    result_y = y
    
    # Apply cascaded GCDs
    for i, (w, a) in enumerate(zip(w_list, a_list[:-1])):
        result = GCD_2(result, result_y, w, a)
        result_y = torch.ones_like(result)  # Subsequent operations are with 1
    
    # Final outer GCD with 1
    result = GCD_2(result, torch.ones_like(result), 0.5, a_list[-1])
    
    return result


def single_gcd(x, y, w, a):
    """Single GCD to approximate n-layer cascade."""
    return GCD_2(x, y, w, a)


def compute_error(x, y, w_list, a_list, w_merged, a_merged):
    """Compute error between n-layer cascade and single merged GCD."""
    cascade = n_layer_cascade(x, y, w_list, a_list)
    merged = single_gcd(x, y, w_merged, a_merged)
    
    errors = torch.abs(cascade - merged)
    return float(errors.mean())


def classify_alpha_branch(a):
    """Classify which branch of GCD_2 piecewise formula α falls into."""
    if a >= 1.995:
        return "hard_and"
    elif a >= 0.75:
        return "power_mean"
    elif a > 0.51:
        return "mixed"
    elif abs(a - 0.5) < 0.01:
        return "weighted"
    else:
        return "dual"


def find_best_merge(x, y, w_list, a_list, 
                    grid_density='medium',
                    optimization_iters=300,
                    optimization_tol=1e-7):
    """
    Find best single GCD to approximate n-layer cascade.
    
    Returns:
        dict with w_merged, a_merged, error, and metadata
    """
    # Configure grid density
    density_configs = {
        'coarse': {'power_mean': 10, 'mixed': 5, 'dual': 8, 'w_points': 9},
        'medium': {'power_mean': 20, 'mixed': 10, 'dual': 15, 'w_points': 13},
        'fine': {'power_mean': 30, 'mixed': 15, 'dual': 20, 'w_points': 17},
    }
    
    config = density_configs[grid_density]
    
    # Grid search
    best_error = float('inf')
    best_w = None
    best_a = None
    
    alpha_ranges = [
        ("power_mean", np.linspace(0.75, 1.99, config['power_mean'])),
        ("mixed", np.linspace(0.51, 0.74, config['mixed'])),
        ("weighted", [0.5]),
        ("dual", np.linspace(-0.99, 0.49, config['dual'])),
    ]
    
    for branch_name, alpha_vals in alpha_ranges:
        for a in alpha_vals:
            for w in np.linspace(0.1, 0.9, config['w_points']):
                error = compute_error(x, y, w_list, a_list, w, a)
                
                if error < best_error:
                    best_error = error
                    best_w = w
                    best_a = a
    
    # Refine with optimization
    def objective(params):
        w, a = params
        w = np.clip(w, 0.01, 0.99)
        a = np.clip(a, -0.99, 1.99)
        return compute_error(x, y, w_list, a_list, w, a)
    
    result = minimize(
        objective,
        x0=[best_w, best_a],
        method='Nelder-Mead',
        options={'maxiter': optimization_iters, 'xatol': 1e-5, 'fatol': optimization_tol}
    )
    
    w_merged = np.clip(result.x[0], 0.01, 0.99)
    a_merged = np.clip(result.x[1], -0.99, 1.99)
    error = result.fun
    
    # Compute additional metrics
    cascade = n_layer_cascade(x, y, w_list, a_list)
    merged = single_gcd(x, y, w_merged, a_merged)
    errors = torch.abs(cascade - merged)
    
    return {
        'n_layers': len(w_list),
        'w_list': w_list,
        'a_list': a_list,
        'w_merged': float(w_merged),
        'a_merged': float(a_merged),
        'branch_merged': classify_alpha_branch(a_merged),
        'error_mean': float(errors.mean()),
        'error_max': float(errors.max()),
        'error_std': float(errors.std()),
        'error_median': float(torch.median(errors))
    }


def run_multilayer_test(n_layers_list=[2, 3, 4, 5],
                       n_samples_per_config=10,
                       grid_density='medium',
                       optimization_iters=300,
                       seed=42):
    """
    Test error accumulation across different numbers of layers.
    
    Args:
        n_layers_list: List of layer counts to test (e.g., [2, 3, 4, 5])
        n_samples_per_config: Number of random parameter combinations per layer count
        grid_density: Grid search density
        optimization_iters: Max optimization iterations
        seed: Random seed for reproducibility
    """
    print("="*80)
    print("MULTI-LAYER GCD MERGE TEST")
    print("="*80)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"\nConfiguration:")
    print(f"  Layer counts:        {n_layers_list}")
    print(f"  Samples per config:  {n_samples_per_config}")
    print(f"  Grid density:        {grid_density}")
    print(f"  Optimization iters:  {optimization_iters}")
    print(f"  Random seed:         {seed}")
    
    # Generate test data
    n_points = 500
    torch.manual_seed(seed)
    np.random.seed(seed)
    x = torch.rand(n_points)
    y = torch.rand(n_points)
    
    print(f"\nTest data: {n_points} random input samples")
    
    results = []
    
    for n_layers in n_layers_list:
        print(f"\n" + "="*80)
        print(f"TESTING {n_layers}-LAYER CASCADES")
        print("="*80)
        
        layer_results = []
        
        for sample_idx in range(n_samples_per_config):
            # Generate random parameters
            # Use good branches (power_mean) for better composition
            w_list = [np.random.uniform(0.4, 0.8) for _ in range(n_layers)]
            a_list = [np.random.uniform(1.0, 1.8) for _ in range(n_layers + 1)]
            
            print(f"\nSample {sample_idx + 1}/{n_samples_per_config}:")
            print(f"  Weights: {[f'{w:.2f}' for w in w_list]}")
            print(f"  Alphas:  {[f'{a:.2f}' for a in a_list]}")
            
            try:
                result = find_best_merge(
                    x, y, w_list, a_list,
                    grid_density=grid_density,
                    optimization_iters=optimization_iters
                )
                
                layer_results.append(result)
                results.append(result)
                
                print(f"  → Merged: w={result['w_merged']:.4f}, α={result['a_merged']:.4f}")
                print(f"  → Error:  {result['error_mean']:.8f} (max: {result['error_max']:.6f})")
                
            except Exception as e:
                print(f"  ⚠️  Error: {e}")
                continue
        
        # Summary for this layer count
        if layer_results:
            errors = np.array([r['error_mean'] for r in layer_results])
            print(f"\n{n_layers}-Layer Summary:")
            print(f"  Mean error:   {errors.mean():.8f} ± {errors.std():.8f}")
            print(f"  Median error: {np.median(errors):.8f}")
            print(f"  Min error:    {errors.min():.8f}")
            print(f"  Max error:    {errors.max():.8f}")
    
    # Overall analysis
    print(f"\n" + "="*80)
    print("OVERALL ANALYSIS")
    print("="*80)
    
    # Group by layer count
    by_layers = {}
    for r in results:
        n = r['n_layers']
        if n not in by_layers:
            by_layers[n] = []
        by_layers[n].append(r['error_mean'])
    
    print(f"\nError vs Number of Layers:")
    print(f"{'Layers':<8} {'Mean Error':<15} {'Std Dev':<15} {'Max Error':<15}")
    print("-" * 60)
    
    for n in sorted(by_layers.keys()):
        errors = np.array(by_layers[n])
        print(f"{n:<8} {errors.mean():<15.8f} {errors.std():<15.8f} {errors.max():<15.8f}")
    
    # Check if error grows with layers
    if len(by_layers) > 1:
        layer_counts = sorted(by_layers.keys())
        mean_errors = [np.mean(by_layers[n]) for n in layer_counts]
        
        print(f"\nError Growth Analysis:")
        for i in range(1, len(layer_counts)):
            prev_error = mean_errors[i-1]
            curr_error = mean_errors[i]
            growth_factor = curr_error / prev_error if prev_error > 0 else float('inf')
            additive_growth = curr_error - prev_error
            
            print(f"  {layer_counts[i-1]}→{layer_counts[i]} layers:")
            print(f"    Additive:      +{additive_growth:.8f}")
            print(f"    Multiplicative: ×{growth_factor:.4f}")
    
    # Save results
    output_file = f"multilayer_merge_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump({
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'n_layers_list': n_layers_list,
                'n_samples_per_config': n_samples_per_config,
                'n_points': n_points,
                'seed': seed
            },
            'by_layers': {str(k): v for k, v in by_layers.items()},
            'results': results
        }, f, indent=2)
    
    print(f"\n" + "="*80)
    print(f"Results saved to: {output_file}")
    print("="*80)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-layer GCD merge test with error accumulation')
    parser.add_argument('--layers', nargs='+', type=int, default=[2, 3, 4, 5],
                       help='Layer counts to test (e.g., --layers 2 3 4 5)')
    parser.add_argument('--samples', type=int, default=10,
                       help='Number of random parameter samples per layer count')
    parser.add_argument('--density', choices=['coarse', 'medium', 'fine'],
                       default='medium', help='Grid search density')
    parser.add_argument('--iters', type=int, default=300,
                       help='Max optimization iterations')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    results = run_multilayer_test(
        n_layers_list=args.layers,
        n_samples_per_config=args.samples,
        grid_density=args.density,
        optimization_iters=args.iters,
        seed=args.seed
    )
