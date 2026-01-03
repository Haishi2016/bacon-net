"""
Test GCD Merge: Can we flatten cascaded GCD operations?

Goal: Find w2, a2 such that:
    GCD(x, y, w2, a2) ≈ GCD(GCD(x, y, w, a), 1, a1)

Since GCD is piecewise-defined with different branches for different α ranges:
  - α = 2: Hard AND
  - 0.75 ≤ α < 2: Power mean variant
  - 0.5 < α < 0.75: Mixed weighted + power mean
  - α = 0.5: Weighted average
  - -1 ≤ α < 0.5: Dual (via complement)

There's no symbolic solution - we approximate by finding the best (w2, α2) match.
"""

import sys
sys.path.insert(0, '../../')

import numpy as np
import torch
from scipy.optimize import minimize
from bacon.aggregators.lsp import FullWeightAggregator

# Create aggregator instance
aggregator = FullWeightAggregator()

def GCD_2(x, y, w, a):
    """
    Wrapper for GCD_2 using FullWeightAggregator.
    
    Args:
        x, y: Input tensors [0,1]
        w: Weight (andness) in [0,1]
        a: Aggregation parameter (alpha)
    
    Returns:
        GCD result tensor
    """
    # Convert w to weight values: w is andness (0=OR, 1=AND)
    # For FullWeight, we need w0, w1 where w is the andness weight
    w0 = 1.0 - w  # weight for OR side
    w1 = w        # weight for AND side
    
    return aggregator.aggregate_tensor(x, y, a, w0, w1)

def cascaded_gcd(x, y, w, a, a1):
    """
    Compute: GCD(GCD(x, y, w, a), 1, a1)
    
    Args:
        x, y: Input values [0,1]
        w: Weight for inner GCD (andness)
        a: Aggregation parameter for inner GCD
        a1: Aggregation parameter for outer GCD
    
    Returns:
        Result of cascaded GCD operation
    """
    # Inner GCD
    inner = GCD_2(x, y, w, a)
    
    # Outer GCD: GCD(inner, 1, a1)
    # Note: second input is 1 (neutral element)
    outer = GCD_2(inner, torch.ones_like(inner), 0.5, a1)  # w=0.5 for balanced
    
    return outer


def single_gcd(x, y, w2, a2):
    """
    Compute: GCD(x, y, w2, a2)
    
    This is what we want to match the cascaded version.
    """
    return GCD_2(x, y, w2, a2)


def compute_error(x, y, w, a, a1, w2, a2):
    """
    Compute difference between cascaded and single GCD.
    """
    cascaded = cascaded_gcd(x, y, w, a, a1)
    single = single_gcd(x, y, w2, a2)
    
    error = torch.abs(cascaded - single).mean().item()
    return error


def classify_alpha_branch(a):
    """Classify which branch of GCD_2 formula is used."""
    if abs(a - 2.0) < 1e-6:
        return "α=2 (hard AND)"
    elif 0.75 <= a < 2.0:
        return f"0.75≤α<2 (power mean, α={a:.3f})"
    elif 0.5 < a < 0.75:
        return f"0.5<α<0.75 (mixed, α={a:.3f})"
    elif abs(a - 0.5) < 1e-6:
        return "α=0.5 (weighted avg)"
    elif -1.0 <= a < 0.5:
        return f"-1≤α<0.5 (dual, α={a:.3f})"
    else:
        return f"α={a:.3f} (out of range)"


def test_hypothesis():
    """
    Test if we can find w2, a2 that approximate cascaded GCD.
    """
    print("="*80)
    print("GCD MERGE APPROXIMATION TEST")
    print("="*80)
    
    # Test parameters for inner GCD
    w = 0.7  # Andness (more AND-like)
    a = 1.8  # LSP aggregation parameter (must be in [-1, 2])
    
    # Test parameter for outer GCD
    a1 = 1.2  # Must be in [-1, 2]
    
    print(f"\nCascaded GCD parameters:")
    print(f"  Inner: w={w}, α={a} [{classify_alpha_branch(a)}]")
    print(f"  Outer: α1={a1} [{classify_alpha_branch(a1)}] (second input = 1)")
    
    # Generate test data
    n_samples = 1000
    x = torch.rand(n_samples)
    y = torch.rand(n_samples)
    
    print(f"\nTest data: {n_samples} random samples in [0,1]")
    
    # Compute cascaded GCD (ground truth)
    cascaded = cascaded_gcd(x, y, w, a, a1)
    
    print(f"\nCascaded GCD output:")
    print(f"  Mean: {cascaded.mean():.4f}")
    print(f"  Std:  {cascaded.std():.4f}")
    print(f"  Range: [{cascaded.min():.4f}, {cascaded.max():.4f}]")
    
    # Try to match with single GCD
    print(f"\n" + "="*80)
    print("METHOD 1: COARSE GRID SEARCH (all α branches)")
    print("="*80)
    
    # Grid search across different α ranges
    best_error = float('inf')
    best_w2 = None
    best_a2 = None
    best_branch = None
    
    # Test different α ranges
    alpha_ranges = [
        ("0.75≤α<2", np.linspace(0.75, 1.99, 25)),
        ("0.5<α<0.75", np.linspace(0.51, 0.74, 12)),
        ("α=0.5", [0.5]),
        ("-1≤α<0.5", np.linspace(-0.99, 0.49, 15)),
    ]
    
    print("\nSearching across α branches:")
    for branch_name, alpha_vals in alpha_ranges:
        for a2 in alpha_vals:
            for w2 in np.linspace(0.1, 0.9, 17):
                error = compute_error(x, y, w, a, a1, w2, a2)
                
                if error < best_error:
                    best_error = error
                    best_w2 = w2
                    best_a2 = a2
                    best_branch = branch_name
    
    print(f"\nBest match from grid search:")
    print(f"  Branch: {best_branch}")
    print(f"  w2 = {best_w2:.4f}")
    print(f"  α2 = {best_a2:.4f}")
    print(f"  Mean absolute error: {best_error:.6f}")
    
    # Refine with optimization
    print(f"\n" + "="*80)
    print("METHOD 2: LOCAL OPTIMIZATION (refinement)")
    print("="*80)
    
    def objective(params):
        w2, a2 = params
        # Constrain to valid ranges
        w2 = np.clip(w2, 0.01, 0.99)
        a2 = np.clip(a2, -0.99, 1.99)
        return compute_error(x, y, w, a, a1, w2, a2)
    
    # Start from grid search result
    result = minimize(
        objective,
        x0=[best_w2, best_a2],
        method='Nelder-Mead',
        options={'maxiter': 500, 'xatol': 1e-6, 'fatol': 1e-8}
    )
    
    w2_opt, a2_opt = result.x
    w2_opt = np.clip(w2_opt, 0.01, 0.99)
    a2_opt = np.clip(a2_opt, -0.99, 1.99)
    error_opt = result.fun
    
    print(f"\nOptimized parameters:")
    print(f"  w2 = {w2_opt:.6f}")
    print(f"  α2 = {a2_opt:.6f} [{classify_alpha_branch(a2_opt)}]")
    print(f"  Mean absolute error: {error_opt:.8f}")
    print(f"  Optimization converged: {result.success}")
    print(f"  Function evaluations: {result.nfev}")
    
    # Use optimized parameters
    best_w2 = w2_opt
    best_a2 = a2_opt
    best_error = error_opt
    
    # Detailed comparison
    print(f"\n" + "="*80)
    print("DETAILED COMPARISON")
    print("="*80)
    
    single = single_gcd(x, y, best_w2, best_a2)
    
    print(f"\nSingle GCD output (with optimized params):")
    print(f"  Mean: {single.mean():.4f}")
    print(f"  Std:  {single.std():.4f}")
    print(f"  Range: [{single.min():.4f}, {single.max():.4f}]")
    
    print(f"\nPointwise error analysis:")
    diff = torch.abs(cascaded - single)
    print(f"  Max error:    {diff.max():.6f}")
    print(f"  Mean error:   {diff.mean():.6f}")
    print(f"  Median error: {diff.median():.6f}")
    print(f"  95th percentile: {torch.quantile(diff, 0.95):.6f}")
    
    # Show a few examples
    print(f"\nSample comparisons (first 5):")
    for i in range(min(5, len(x))):
        print(f"  x={x[i]:.3f}, y={y[i]:.3f}: cascaded={cascaded[i]:.6f}, single={single[i]:.6f}, diff={diff[i]:.6f}")
    
    # Final verdict
    print(f"\n" + "="*80)
    if best_error < 1e-3:
        print(f"✅ EXCELLENT: Found near-equivalent parameters (error < 0.001)")
    elif best_error < 1e-2:
        print(f"⚠️  GOOD: Close approximation (error < 0.01)")
    elif best_error < 0.05:
        print(f"⚠️  ACCEPTABLE: Reasonable approximation (error < 0.05)")
    else:
        print(f"❌ POOR: Approximation quality insufficient (error = {best_error:.4f})")
    
    print(f"\n💡 To use this approximation:")
    print(f"   Replace: GCD(GCD(x,y,{w},{a}), 1, {a1})")
    print(f"   With:    GCD(x, y, {best_w2:.4f}, {best_a2:.4f})")
    print(f"   Error:   ~{best_error:.4f} mean absolute difference")
    
    print("\n" + "="*80)
    return best_w2, best_a2, best_error


if __name__ == "__main__":
    test_hypothesis()
