# GCD Merge Experiments

## Goal
Find algebraic simplification to merge cascaded GCD_2 operators.

## Problem Statement
Given:
- Inner: `z = GCD(x, y, w, a)`
- Outer: `result = GCD(z, 1, a1)`

Find: `w2, a2` such that:
```
GCD(x, y, w2, a2) = GCD(GCD(x, y, w, a), 1, a1)
```

## Why This Matters
- **Model simplification**: Reduce tree depth
- **Computational efficiency**: Fewer operations
- **Interpretability**: Simpler logical structure

## Approach
1. Algebraic analysis of GCD composition
2. Numerical search for w2, a2 parameters
3. Validation across input ranges
