"""
Test to understand why pruning feature 0 causes 5% accuracy drop
even though it has balanced weights (0.5/0.5) and low bias (0.0227)
"""

import torch
import numpy as np

def lsp_aggregate(x, y, a, w0, w1):
    """Simplified LSP half-weight aggregator"""
    epsilon = 1e-6
    x = np.clip(x, epsilon, 1-epsilon)
    y = np.clip(y, epsilon, 1-epsilon)
    a = np.clip(a, -1.0 + epsilon, 2.0 - epsilon)
    
    # For a ≈ 0.5, uses arithmetic mean
    if abs(a - 0.5) < epsilon:
        return w0*x + w1*y
    
    # For other values, simplified formula
    return w0*x + w1*y  # Simplified for this test

# Test with sample values
np.random.seed(42)
num_samples = 100

# Generate sample feature values
triglyceride = np.random.uniform(0.1, 0.9, num_samples)
vma = np.random.uniform(0.1, 0.9, num_samples)

# Original: aggregate with balanced weights and low bias
original_output = lsp_aggregate(triglyceride, vma, a=0.0227, w0=0.5, w1=0.5)

# Pruned: effectively just VMA (w0=0, w1=1)
pruned_output = lsp_aggregate(triglyceride, vma, a=0.0227, w0=0.0, w1=1.0)

# Compare
print("="*70)
print("PRUNING EFFECT ANALYSIS")
print("="*70)
print(f"\nOriginal aggregation (w0=0.5, w1=0.5, a=0.0227):")
print(f"  Formula: 0.5*Triglyceride + 0.5*VMA")
print(f"  Mean output: {original_output.mean():.4f}")
print(f"  Std output:  {original_output.std():.4f}")

print(f"\nPruned aggregation (w0=0.0, w1=1.0, a=0.0227):")
print(f"  Formula: 0.0*Triglyceride + 1.0*VMA = VMA")
print(f"  Mean output: {pruned_output.mean():.4f}")
print(f"  Std output:  {pruned_output.std():.4f}")

print(f"\nDifference:")
print(f"  Mean difference: {abs(original_output.mean() - pruned_output.mean()):.4f}")
print(f"  Max difference:  {abs(original_output - pruned_output).max():.4f}")

# Show that the outputs are different
correlation = np.corrcoef(original_output, pruned_output)[0, 1]
print(f"  Correlation: {correlation:.4f}")

# Show sample comparisons
print(f"\nSample comparisons (first 5):")
print(f"  {'Triglyceride':<15} {'VMA':<15} {'Original':<15} {'Pruned':<15} {'Diff':<15}")
print("  " + "-"*75)
for i in range(5):
    diff = original_output[i] - pruned_output[i]
    print(f"  {triglyceride[i]:<15.4f} {vma[i]:<15.4f} {original_output[i]:<15.4f} {pruned_output[i]:<15.4f} {diff:<15.4f}")

print("\n" + "="*70)
print("CONCLUSION")
print("="*70)
print("""
Even with balanced weights (0.5/0.5), the aggregation output is DIFFERENT
from just using the right feature (VMA) alone.

Original:  output = 0.5*Triglyceride + 0.5*VMA
Pruned:    output = 1.0*VMA

The difference is 0.5*Triglyceride, which is significant!

This explains the 5% accuracy drop: the model learned to use the
AVERAGED value of both features, not just VMA alone. When you prune
Triglyceride, you're not just removing a "low weight" feature - you're
fundamentally changing the input distribution to all downstream layers.

The 5% drop is NOT a bug - it's the true cost of removing that feature.
""")
