"""
Comprehensive GCD Merge Test: Systematic evaluation across parameter space.

Tests GCD merge approximation quality across multiple combinations of:
- Inner GCD parameters (w, α)
- Outer GCD parameters (α1)
- Different α branches

Generates statistics and identifies which configurations can be merged well.
"""

import sys
sys.path.insert(0, '../../')

import numpy as np
import torch
from scipy.optimize import minimize
from bacon.aggregators.lsp import FullWeightAggregator
import json
from datetime import datetime

# Create aggregator instance
aggregator = FullWeightAggregator()

def GCD_2(x, y, w, a):
    """Wrapper for GCD_2 using FullWeightAggregator."""
    w0 = 1.0 - w
    w1 = w
    return aggregator.aggregate_tensor(x, y, a, w0, w1)

def cascaded_gcd(x, y, w, a, a1):
    """Compute: GCD(GCD(x, y, w, a), 1, a1)"""
    inner = GCD_2(x, y, w, a)
    outer = GCD_2(inner, torch.ones_like(inner), 0.5, a1)
    return outer

def single_gcd(x, y, w2, a2):
    """Compute: GCD(x, y, w2, a2)"""
    return GCD_2(x, y, w2, a2)

def compute_error(x, y, w, a, a1, w2, a2):
    """Compute difference between cascaded and single GCD."""
    cascaded = cascaded_gcd(x, y, w, a, a1)
    single = single_gcd(x, y, w2, a2)
    error = torch.abs(cascaded - single).mean().item()
    return error

def classify_alpha_branch(a):
    """Classify which branch of GCD_2 formula is used."""
    if abs(a - 2.0) < 1e-6:
        return "hard_and"
    elif 0.75 <= a < 2.0:
        return "power_mean"
    elif 0.5 < a < 0.75:
        return "mixed"
    elif abs(a - 0.5) < 1e-6:
        return "weighted"
    elif -1.0 <= a < 0.5:
        return "dual"
    else:
        return "out_of_range"

def find_best_approximation(x, y, w, a, a1, 
                           grid_density='medium',
                           optimization_iters=300,
                           optimization_tol=1e-7,
                           multi_start=False):
    """
    Find best (w2, α2) to approximate GCD(GCD(x,y,w,a), 1, a1).
    
    Args:
        x, y: Input tensors
        w, a, a1: Cascaded GCD parameters
        grid_density: 'coarse', 'medium', 'fine', or 'ultra_fine'
        optimization_iters: Max iterations for local optimization
        optimization_tol: Convergence tolerance for optimization
        multi_start: If True, try optimization from top 3 grid points
    
    Returns:
        dict with w2, a2, error, branch, and other metrics
    """
    # Configure grid density
    density_configs = {
        'coarse': {'power_mean': 10, 'mixed': 5, 'dual': 8, 'w_points': 9},
        'medium': {'power_mean': 20, 'mixed': 10, 'dual': 15, 'w_points': 13},
        'fine': {'power_mean': 30, 'mixed': 15, 'dual': 20, 'w_points': 17},
        'ultra_fine': {'power_mean': 50, 'mixed': 25, 'dual': 30, 'w_points': 25}
    }
    
    config = density_configs[grid_density]
    
    # Grid search across α ranges
    best_error = float('inf')
    best_w2 = None
    best_a2 = None
    best_branch = None
    
    # Track top N candidates for multi-start
    top_candidates = []
    
    alpha_ranges = [
        ("power_mean", np.linspace(0.75, 1.99, config['power_mean'])),
        ("mixed", np.linspace(0.51, 0.74, config['mixed'])),
        ("weighted", [0.5]),
        ("dual", np.linspace(-0.99, 0.49, config['dual'])),
    ]
    
    for branch_name, alpha_vals in alpha_ranges:
        for a2 in alpha_vals:
            for w2 in np.linspace(0.1, 0.9, config['w_points']):
                error = compute_error(x, y, w, a, a1, w2, a2)
                
                if error < best_error:
                    best_error = error
                    best_w2 = w2
                    best_a2 = a2
                    best_branch = branch_name
                
                # Track top 3 for multi-start
                if multi_start:
                    top_candidates.append((error, w2, a2))
                    top_candidates.sort(key=lambda x: x[0])
                    if len(top_candidates) > 3:
                        top_candidates.pop()
    
    # Refine with optimization
    def objective(params):
        w2, a2 = params
        w2 = np.clip(w2, 0.01, 0.99)
        a2 = np.clip(a2, -0.99, 1.99)
        return compute_error(x, y, w, a, a1, w2, a2)
    
    # Try optimization from best grid point
    result = minimize(
        objective,
        x0=[best_w2, best_a2],
        method='Nelder-Mead',
        options={'maxiter': optimization_iters, 'xatol': 1e-5, 'fatol': optimization_tol}
    )
    
    best_result = result
    
    # Multi-start: try from other good grid points
    if multi_start and len(top_candidates) > 1:
        for _, w2_start, a2_start in top_candidates[1:]:
            result_alt = minimize(
                objective,
                x0=[w2_start, a2_start],
                method='Nelder-Mead',
                options={'maxiter': optimization_iters, 'xatol': 1e-5, 'fatol': optimization_tol}
            )
            if result_alt.fun < best_result.fun:
                best_result = result_alt
    
    result = best_result
    
    w2_opt = np.clip(result.x[0], 0.01, 0.99)
    a2_opt = np.clip(result.x[1], -0.99, 1.99)
    error_opt = result.fun
    
    # Compute detailed metrics
    cascaded = cascaded_gcd(x, y, w, a, a1)
    single = single_gcd(x, y, w2_opt, a2_opt)
    diff = torch.abs(cascaded - single)
    
    return {
        'w': w,
        'a': a,
        'a1': a1,
        'w2': float(w2_opt),
        'a2': float(a2_opt),
        'branch_inner': classify_alpha_branch(a),
        'branch_outer': classify_alpha_branch(a1),
        'branch_merged': classify_alpha_branch(a2_opt),
        'error_mean': float(diff.mean().item()),
        'error_max': float(diff.max().item()),
        'error_median': float(diff.median().item()),
        'error_95th': float(torch.quantile(diff, 0.95).item()),
        'optimization_converged': result.success,
        'optimization_iterations': result.nfev
    }

def run_comprehensive_test(grid_density='medium', 
                          optimization_iters=300,
                          optimization_tol=1e-7,
                          multi_start=False):
    """
    Run comprehensive test across parameter space.
    
    Args:
        grid_density: 'coarse', 'medium', 'fine', 'ultra_fine'
                      Controls grid search granularity (coarse=fastest, ultra_fine=slowest)
        optimization_iters: Max iterations for local optimization (default: 300)
        optimization_tol: Convergence tolerance (default: 1e-7)
        multi_start: Try optimization from multiple starting points (slower but more robust)
    """
    print("="*80)
    print("COMPREHENSIVE GCD MERGE TEST")
    print("="*80)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"\nSearch Configuration:")
    print(f"  Grid density:        {grid_density}")
    print(f"  Optimization iters:  {optimization_iters}")
    print(f"  Optimization tol:    {optimization_tol}")
    print(f"  Multi-start:         {multi_start}")
    
    # Generate test data (fixed across all tests)
    n_samples = 500
    torch.manual_seed(42)
    x = torch.rand(n_samples)
    y = torch.rand(n_samples)
    
    print(f"\nTest data: {n_samples} random samples (seed=42)")
    
    # Define parameter grid
    w_values = [0.3, 0.5, 0.7, 0.9]  # Different andness levels
    
    # Inner α values (covering different branches)
    a_values = [
        -0.5,  # dual
        0.5,   # weighted
        0.6,   # mixed
        1.0,   # power mean
        1.5,   # power mean
        1.9,   # power mean (near hard AND)
    ]
    
    # Outer α values
    a1_values = [
        -0.3,  # dual
        0.5,   # weighted
        0.65,  # mixed
        1.2,   # power mean
        1.8,   # power mean
    ]
    
    total_tests = len(w_values) * len(a_values) * len(a1_values)
    
    print(f"\nParameter grid:")
    print(f"  w (andness):  {w_values}")
    print(f"  α (inner):    {a_values}")
    print(f"  α1 (outer):   {a1_values}")
    print(f"  Total tests:  {total_tests}")
    
    print(f"\n" + "="*80)
    print("RUNNING TESTS...")
    print("="*80)
    
    results = []
    test_count = 0
    
    for w in w_values:
        for a in a_values:
            for a1 in a1_values:
                test_count += 1
                
                # Progress indicator
                if test_count % 10 == 0 or test_count == 1:
                    print(f"\nTest {test_count}/{total_tests}: w={w:.1f}, α={a:.1f}, α1={a1:.1f}")
                
                try:
                    result = find_best_approximation(x, y, w, a, a1)
                    results.append(result)
                    
                    # Show result for this test
                    if test_count % 10 == 0 or test_count == 1:
                        print(f"  → Best: w2={result['w2']:.4f}, α2={result['a2']:.4f}, error={result['error_mean']:.6f}")
                        print(f"     Branches: {result['branch_inner']}+{result['branch_outer']} → {result['branch_merged']}")
                except Exception as e:
                    print(f"  ⚠️  Error: {e}")
                    continue
    
    print(f"\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    
    # Convert to numpy arrays for analysis
    errors = np.array([r['error_mean'] for r in results])
    errors_max = np.array([r['error_max'] for r in results])
    
    print(f"\nOverall Statistics:")
    print(f"  Tests completed: {len(results)}/{total_tests}")
    print(f"  Mean error:      {errors.mean():.6f}")
    print(f"  Median error:    {np.median(errors):.6f}")
    print(f"  Min error:       {errors.min():.6f}")
    print(f"  Max error:       {errors.max():.6f}")
    print(f"  Std dev:         {errors.std():.6f}")
    
    # Quality categories
    excellent = sum(errors < 0.001)
    good = sum((errors >= 0.001) & (errors < 0.01))
    acceptable = sum((errors >= 0.01) & (errors < 0.05))
    poor = sum(errors >= 0.05)
    
    print(f"\nQuality Distribution:")
    print(f"  Excellent (< 0.001):  {excellent:3d} ({100*excellent/len(results):.1f}%)")
    print(f"  Good (< 0.01):        {good:3d} ({100*good/len(results):.1f}%)")
    print(f"  Acceptable (< 0.05):  {acceptable:3d} ({100*acceptable/len(results):.1f}%)")
    print(f"  Poor (≥ 0.05):        {poor:3d} ({100*poor/len(results):.1f}%)")
    
    # Best and worst cases
    print(f"\n" + "="*80)
    print("BEST CASES (Top 5)")
    print("="*80)
    
    sorted_results = sorted(results, key=lambda r: r['error_mean'])
    for i, r in enumerate(sorted_results[:5]):
        print(f"\n{i+1}. Error: {r['error_mean']:.8f}")
        print(f"   Inner:  GCD(x,y, w={r['w']:.1f}, α={r['a']:.2f}) [{r['branch_inner']}]")
        print(f"   Outer:  GCD(inner, 1, α1={r['a1']:.2f}) [{r['branch_outer']}]")
        print(f"   Merged: GCD(x,y, w={r['w2']:.4f}, α={r['a2']:.4f}) [{r['branch_merged']}]")
    
    print(f"\n" + "="*80)
    print("WORST CASES (Top 5)")
    print("="*80)
    
    for i, r in enumerate(sorted_results[-5:][::-1]):
        print(f"\n{i+1}. Error: {r['error_mean']:.8f} (max: {r['error_max']:.6f})")
        print(f"   Inner:  GCD(x,y, w={r['w']:.1f}, α={r['a']:.2f}) [{r['branch_inner']}]")
        print(f"   Outer:  GCD(inner, 1, α1={r['a1']:.2f}) [{r['branch_outer']}]")
        print(f"   Merged: GCD(x,y, w={r['w2']:.4f}, α={r['a2']:.4f}) [{r['branch_merged']}]")
    
    # Analyze by branch combinations
    print(f"\n" + "="*80)
    print("ANALYSIS BY BRANCH COMBINATION")
    print("="*80)
    
    branch_stats = {}
    for r in results:
        key = f"{r['branch_inner']}+{r['branch_outer']}"
        if key not in branch_stats:
            branch_stats[key] = []
        branch_stats[key].append(r['error_mean'])
    
    print(f"\nMean error by inner+outer branch combination:")
    for key in sorted(branch_stats.keys()):
        errors = branch_stats[key]
        print(f"  {key:20s}: {np.mean(errors):.6f} ± {np.std(errors):.6f} (n={len(errors)})")
    
    # Save results
    output_file = f"gcd_merge_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump({
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'n_samples': n_samples,
                'total_tests': total_tests,
                'completed_tests': len(results)
            },
            'summary': {
                'mean_error': float(np.mean(errors)),
                'median_error': float(np.median(errors)),
                'min_error': float(np.min(errors)),
                'max_error': float(np.max(errors)),
                'std_error': float(np.std(errors)),
                'quality_distribution': {
                    'excellent': int(excellent),
                    'good': int(good),
                    'acceptable': int(acceptable),
                    'poor': int(poor)
                }
            },
            'results': results
        }, f, indent=2)
    
    print(f"\n" + "="*80)
    print(f"Results saved to: {output_file}")
    print("="*80)
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Comprehensive GCD merge approximation test')
    parser.add_argument('--density', choices=['coarse', 'medium', 'fine', 'ultra_fine'],
                       default='medium', help='Grid search density')
    parser.add_argument('--iters', type=int, default=300,
                       help='Max optimization iterations')
    parser.add_argument('--tol', type=float, default=1e-7,
                       help='Optimization convergence tolerance')
    parser.add_argument('--multi-start', action='store_true',
                       help='Enable multi-start optimization (slower but more robust)')
    
    args = parser.parse_args()
    
    results = run_comprehensive_test(
        grid_density=args.density,
        optimization_iters=args.iters,
        optimization_tol=args.tol,
        multi_start=args.multi_start
    )
