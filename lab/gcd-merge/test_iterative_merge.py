"""
Test iterative GCD merging to measure true error accumulation.

This test starts with an n-layer cascade and progressively merges layers
from the bottom up, tracking cumulative error at each step.
"""

import torch
import numpy as np
from scipy.optimize import minimize
from datetime import datetime
import json
import copy

from bacon.aggregators.lsp import FullWeightAggregator


def classify_operator_type(w, a):
    """
    Classify GCD operator as conjunctive, neutral, or disjunctive.
    
    Based on andness = w * (1 + α) / 2:
    - Conjunctive: andness > 0.6 (AND-like, error amplifying)
    - Neutral: 0.4 ≤ andness ≤ 0.6 (balanced)
    - Disjunctive: andness < 0.4 (OR-like, error dampening)
    """
    andness = w * (1 + a) / 2
    
    if andness > 0.6:
        return "conjunctive"
    elif andness >= 0.4:
        return "neutral"
    else:
        return "disjunctive"


def GCD_2(x, y, w, a):
    """Wrapper for GCD_2 aggregation using FullWeightAggregator."""
    aggregator = FullWeightAggregator()
    w0 = 1 - w
    w1 = w
    return aggregator.aggregate_tensor(x, y, a, w0, w1)


def evaluate_cascade(x, y, w_list, a_list):
    """
    Evaluate a cascade represented by lists of parameters.
    
    Args:
        x, y: Input tensors
        w_list: List of weights [w1, w2, ..., wn]
        a_list: List of alphas [a1, a2, ..., a_{n+1}]
    
    Returns:
        Result tensor
    """
    if len(w_list) == 0:
        # Base case: just return x (shouldn't happen)
        return x
    
    if len(w_list) == 1:
        # Single GCD
        return GCD_2(x, y, w_list[0], a_list[0])
    
    # Compute inner GCD
    result = GCD_2(x, y, w_list[0], a_list[0])
    
    # Apply remaining layers with 1
    for i in range(1, len(w_list)):
        result = GCD_2(result, torch.ones_like(result), w_list[i], a_list[i])
    
    # Final outer alpha (if more alphas than weights)
    if len(a_list) > len(w_list):
        result = GCD_2(result, torch.ones_like(result), 0.5, a_list[-1])
    
    return result


def merge_bottom_two_layers(x, y, w_list, a_list, 
                            grid_density='medium',
                            optimization_iters=300):
    """
    Merge the bottom two layers of a cascade into a single layer.
    
    Args:
        w_list: [w1, w2, ..., wn] - weights for n layers
        a_list: [a1, a2, ..., a_{n+1}] - alphas for cascade
    
    Returns:
        New (w_list, a_list) with bottom two layers merged
        Error metrics dict
    """
    if len(w_list) < 2:
        raise ValueError("Need at least 2 layers to merge")
    
    # Original cascade output
    original = evaluate_cascade(x, y, w_list, a_list)
    
    # Grid search for best (w_merged, a_merged)
    density_configs = {
        'coarse': {'power_mean': 10, 'mixed': 5, 'dual': 8, 'w_points': 9},
        'medium': {'power_mean': 20, 'mixed': 10, 'dual': 15, 'w_points': 13},
        'fine': {'power_mean': 30, 'mixed': 15, 'dual': 20, 'w_points': 17},
    }
    config = density_configs[grid_density]
    
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
        for a_merged in alpha_vals:
            for w_merged in np.linspace(0.1, 0.9, config['w_points']):
                # Create merged cascade
                new_w_list = [w_merged] + w_list[2:]
                new_a_list = [a_merged] + a_list[2:]
                
                # Evaluate
                merged = evaluate_cascade(x, y, new_w_list, new_a_list)
                error = float(torch.abs(original - merged).mean())
                
                if error < best_error:
                    best_error = error
                    best_w = w_merged
                    best_a = a_merged
    
    # Refine with optimization
    def objective(params):
        w, a = params
        w = np.clip(w, 0.01, 0.99)
        a = np.clip(a, -0.99, 1.99)
        
        new_w_list = [w] + w_list[2:]
        new_a_list = [a] + a_list[2:]
        merged = evaluate_cascade(x, y, new_w_list, new_a_list)
        return float(torch.abs(original - merged).mean())
    
    result = minimize(
        objective,
        x0=[best_w, best_a],
        method='Nelder-Mead',
        options={'maxiter': optimization_iters, 'xatol': 1e-5, 'fatol': 1e-7}
    )
    
    w_final = np.clip(result.x[0], 0.01, 0.99)
    a_final = np.clip(result.x[1], -0.99, 1.99)
    
    # Build final merged cascade
    merged_w_list = [w_final] + w_list[2:]
    merged_a_list = [a_final] + a_list[2:]
    
    # Compute error metrics
    merged_output = evaluate_cascade(x, y, merged_w_list, merged_a_list)
    errors = torch.abs(original - merged_output)
    
    error_metrics = {
        'error_mean': float(errors.mean()),
        'error_max': float(errors.max()),
        'error_std': float(errors.std()),
        'error_median': float(torch.median(errors)),
        'w_merged': float(w_final),
        'a_merged': float(a_final),
        'original_w1': float(w_list[0]),
        'original_w2': float(w_list[1]),
        'original_a1': float(a_list[0]),
        'original_a2': float(a_list[1]),
        'type_layer1': classify_operator_type(w_list[0], a_list[0]),
        'type_layer2': classify_operator_type(w_list[1], a_list[1]) if len(w_list) > 1 else None,
        'type_merged': classify_operator_type(w_final, a_final),
    }
    
    return merged_w_list, merged_a_list, error_metrics


def iterative_merge_test(initial_w_list, initial_a_list, x, y,
                         grid_density='medium',
                         optimization_iters=300):
    """
    Progressively merge layers from bottom up, tracking cumulative error.
    
    Args:
        initial_w_list: Starting weights [w1, w2, ..., wn]
        initial_a_list: Starting alphas [a1, a2, ..., a_{n+1}]
        x, y: Input tensors
    
    Returns:
        List of merge step results with cumulative errors
    """
    n_initial_layers = len(initial_w_list)
    
    print(f"\n{'='*80}")
    print(f"ITERATIVE MERGE: {n_initial_layers}-Layer Cascade")
    print(f"{'='*80}")
    print(f"Initial configuration:")
    print(f"  Weights: {[f'{w:.3f}' for w in initial_w_list]}")
    print(f"  Alphas:  {[f'{a:.3f}' for a in initial_a_list]}")
    
    # Evaluate original cascade
    original_output = evaluate_cascade(x, y, initial_w_list, initial_a_list)
    
    # Track current state
    current_w_list = initial_w_list.copy()
    current_a_list = initial_a_list.copy()
    
    merge_steps = []
    cumulative_errors = []
    
    step = 1
    while len(current_w_list) >= 2:
        print(f"\n{'-'*80}")
        print(f"MERGE STEP {step}: {len(current_w_list)} → {len(current_w_list)-1} layers")
        print(f"{'-'*80}")
        print(f"Current: w={[f'{w:.3f}' for w in current_w_list]}, a={[f'{a:.3f}' for a in current_a_list]}")
        
        # Merge bottom two layers
        new_w_list, new_a_list, error_metrics = merge_bottom_two_layers(
            x, y, current_w_list, current_a_list,
            grid_density=grid_density,
            optimization_iters=optimization_iters
        )
        
        # Evaluate merged cascade against original
        merged_output = evaluate_cascade(x, y, new_w_list, new_a_list)
        cumulative_error = float(torch.abs(original_output - merged_output).mean())
        cumulative_errors.append(cumulative_error)
        
        print(f"Merged:  w={[f'{w:.3f}' for w in new_w_list]}, a={[f'{a:.3f}' for a in new_a_list]}")
        print(f"Operator types: {error_metrics['type_layer1']}+{error_metrics['type_layer2']} → {error_metrics['type_merged']}")
        print(f"Step error (vs prev):     {error_metrics['error_mean']:.8f}")
        print(f"Cumulative error (vs orig): {cumulative_error:.8f}")
        
        # Store step info
        step_info = {
            'step': step,
            'layers_before': len(current_w_list),
            'layers_after': len(new_w_list),
            'w_before': current_w_list.copy(),
            'a_before': current_a_list.copy(),
            'w_after': new_w_list.copy(),
            'a_after': new_a_list.copy(),
            'step_error': error_metrics['error_mean'],
            'cumulative_error': cumulative_error,
            'error_metrics': error_metrics
        }
        merge_steps.append(step_info)
        
        # Update current state
        current_w_list = new_w_list
        current_a_list = new_a_list
        step += 1
    
    print(f"\n{'='*80}")
    print(f"MERGE COMPLETE: {n_initial_layers} layers → 1 layer")
    print(f"{'='*80}")
    print(f"Final configuration:")
    print(f"  w={current_w_list[0]:.6f}, a={current_a_list[0]:.6f}")
    print(f"  Total cumulative error: {cumulative_errors[-1]:.8f}")
    
    return {
        'initial_layers': n_initial_layers,
        'initial_w': initial_w_list,
        'initial_a': initial_a_list,
        'final_w': current_w_list[0],
        'final_a': current_a_list[0],
        'merge_steps': merge_steps,
        'cumulative_errors': cumulative_errors
    }


def run_multiple_iterative_tests(n_layers_list=[3, 4, 5, 6],
                                 n_samples_per_layer=5,
                                 grid_density='medium',
                                 optimization_iters=300,
                                 seed=42):
    """
    Run iterative merge tests on multiple random configurations.
    
    Args:
        n_layers_list: List of initial layer counts to test
        n_samples_per_layer: Number of random configs per layer count
        grid_density: Grid search density
        optimization_iters: Max optimization iterations per merge
        seed: Random seed
    """
    print("="*80)
    print("ITERATIVE MERGE ACCUMULATION TEST")
    print("="*80)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"\nConfiguration:")
    print(f"  Initial layer counts: {n_layers_list}")
    print(f"  Samples per count:    {n_samples_per_layer}")
    print(f"  Grid density:         {grid_density}")
    print(f"  Optimization iters:   {optimization_iters}")
    
    # Generate test data
    n_points = 500
    torch.manual_seed(seed)
    np.random.seed(seed)
    x = torch.rand(n_points)
    y = torch.rand(n_points)
    
    print(f"\nTest data: {n_points} random input samples")
    
    all_results = []
    
    for n_layers in n_layers_list:
        print(f"\n{'='*80}")
        print(f"TESTING {n_layers}-LAYER CASCADES")
        print(f"{'='*80}")
        
        for sample_idx in range(n_samples_per_layer):
            print(f"\nSample {sample_idx + 1}/{n_samples_per_layer}")
            
            # Generate random parameters (use good range)
            w_list = [np.random.uniform(0.4, 0.8) for _ in range(n_layers)]
            a_list = [np.random.uniform(1.0, 1.8) for _ in range(n_layers + 1)]
            
            try:
                result = iterative_merge_test(
                    w_list, a_list, x, y,
                    grid_density=grid_density,
                    optimization_iters=optimization_iters
                )
                all_results.append(result)
                
            except Exception as e:
                print(f"  ⚠️  Error: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Analysis
    print(f"\n{'='*80}")
    print("OVERALL ANALYSIS")
    print(f"{'='*80}")
    
    # Group by initial layer count
    by_layers = {}
    for r in all_results:
        n = r['initial_layers']
        if n not in by_layers:
            by_layers[n] = []
        by_layers[n].append(r['cumulative_errors'][-1])  # Final error
    
    print(f"\nFinal cumulative error by initial layer count:")
    print(f"{'Layers':<8} {'Mean Error':<15} {'Std Dev':<15} {'Min':<15} {'Max':<15}")
    print("-" * 75)
    
    for n in sorted(by_layers.keys()):
        errors = np.array(by_layers[n])
        print(f"{n:<8} {errors.mean():<15.8f} {errors.std():<15.8f} {errors.min():<15.8f} {errors.max():<15.8f}")
    
    # Check growth pattern
    if len(by_layers) > 1:
        print(f"\nError growth vs initial layer count:")
        layer_counts = sorted(by_layers.keys())
        mean_errors = [np.mean(by_layers[n]) for n in layer_counts]
        
        for i in range(1, len(layer_counts)):
            prev = mean_errors[i-1]
            curr = mean_errors[i]
            if prev > 0:
                mult_factor = curr / prev
                add_growth = curr - prev
                print(f"  {layer_counts[i-1]}→{layer_counts[i]} layers: +{add_growth:.8f} (×{mult_factor:.4f})")
    
    # Analyze by operator type transitions
    print(f"\n{'='*80}")
    print("ERROR ACCUMULATION BY OPERATOR TYPE TRANSITIONS")
    print(f"{'='*80}")
    
    # Collect all merge steps across all results
    all_merge_steps = []
    for r in all_results:
        all_merge_steps.extend(r['merge_steps'])
    
    # Group by operator type transition
    transition_errors = {}
    for step in all_merge_steps:
        metrics = step['error_metrics']
        transition = f"{metrics['type_layer1']}+{metrics['type_layer2']}"
        
        if transition not in transition_errors:
            transition_errors[transition] = []
        transition_errors[transition].append(step['step_error'])
    
    print(f"\nStep error by operator transition pattern:")
    print(f"{'Transition':<25} {'Mean Error':<15} {'Std Dev':<15} {'Count':<8}")
    print("-" * 70)
    
    for transition in sorted(transition_errors.keys()):
        errors = np.array(transition_errors[transition])
        print(f"{transition:<25} {errors.mean():<15.8f} {errors.std():<15.8f} {len(errors):<8}")
    
    # Analyze step-by-step error accumulation
    print(f"\n{'='*80}")
    print("STEP-BY-STEP ERROR ACCUMULATION")
    print(f"{'='*80}")
    
    # Average cumulative error at each merge step
    max_steps = max(len(r['cumulative_errors']) for r in all_results)
    
    # Detailed operator type analysis
    print(f"\n{'='*80}")
    print("OPERATOR TYPE STATISTICS")
    print(f"{'='*80}")
    
    # Count operator types in original cascades
    all_operator_types = []
    for r in all_results:
        for w, a in zip(r['initial_w'], r['initial_a'][:-1]):  # Exclude final outer alpha
            all_operator_types.append(classify_operator_type(w, a))
    
    type_counts = {}
    for op_type in all_operator_types:
        type_counts[op_type] = type_counts.get(op_type, 0) + 1
    
    print(f"\nDistribution of operator types in test cases:")
    for op_type in ['conjunctive', 'neutral', 'disjunctive']:
        count = type_counts.get(op_type, 0)
        pct = 100 * count / len(all_operator_types) if all_operator_types else 0
        print(f"  {op_type:<15}: {count:>3} ({pct:>5.1f}%)")
    
    # Analyze which transitions are most/least error-prone
    if transition_errors:
        print(f"\nMost error-prone transitions (worst 3):")
        sorted_transitions = sorted(transition_errors.items(), 
                                    key=lambda x: np.mean(x[1]), 
                                    reverse=True)[:3]
        for transition, errors in sorted_transitions:
            print(f"  {transition:<25}: {np.mean(errors):.8f} ± {np.std(errors):.8f}")
        
        print(f"\nMost robust transitions (best 3):")
        sorted_transitions = sorted(transition_errors.items(), 
                                    key=lambda x: np.mean(x[1]))[:3]
        for transition, errors in sorted_transitions:
            print(f"  {transition:<25}: {np.mean(errors):.8f} ± {np.std(errors):.8f}")
    
    print(f"\nAverage cumulative error after each merge step:")
    print(f"{'Step':<8} {'Avg Error':<15} {'Count':<8}")
    print("-" * 35)
    
    for step in range(max_steps):
        errors_at_step = []
        for r in all_results:
            if step < len(r['cumulative_errors']):
                errors_at_step.append(r['cumulative_errors'][step])
        
        if errors_at_step:
            avg_error = np.mean(errors_at_step)
            print(f"{step+1:<8} {avg_error:<15.8f} {len(errors_at_step):<8}")
    
    # Save results
    output_file = f"iterative_merge_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump({
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'n_layers_list': n_layers_list,
                'n_samples_per_layer': n_samples_per_layer,
                'n_points': n_points,
                'seed': seed
            },
            'results': all_results
        }, f, indent=2)
    
    print(f"\n{'='*80}")
    print(f"Results saved to: {output_file}")
    print(f"{'='*80}")
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Iterative GCD merge with true error accumulation')
    parser.add_argument('--layers', nargs='+', type=int, default=[3, 4, 5, 6],
                       help='Initial layer counts to test (e.g., --layers 3 4 5 6)')
    parser.add_argument('--samples', type=int, default=5,
                       help='Number of random parameter samples per layer count')
    parser.add_argument('--density', choices=['coarse', 'medium', 'fine'],
                       default='medium', help='Grid search density')
    parser.add_argument('--iters', type=int, default=300,
                       help='Max optimization iterations per merge step')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    results = run_multiple_iterative_tests(
        n_layers_list=args.layers,
        n_samples_per_layer=args.samples,
        grid_density=args.density,
        optimization_iters=args.iters,
        seed=args.seed
    )
