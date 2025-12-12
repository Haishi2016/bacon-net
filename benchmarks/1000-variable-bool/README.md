# 1000-Variable Boolean Expression Inference Benchmark

This benchmark compares BACON vs Decision Tree on inferring a randomly generated 1000-variable boolean expression.

## Setup

- **Input**: 1000 boolean variables (A, B, C, ...) 
- **Expression**: Randomly generated with AND/OR operations (e.g., `((A and B) or (C and D)) and ...`)
- **Training**: 10,000 randomly sampled boolean assignments
- **Testing**: 5,000 randomly sampled boolean assignments

## Models

### BACON (`bacon.py`)
- Binary tree structure with hierarchical permutation search
- `bool.min_max` aggregator (classic boolean logic)
- Hierarchical grouping: 1000 vars → ~20 coarse groups (group_size=50)
- Temperature annealing for permutation discovery
- Freeze threshold: 0.18 (relaxed for large input)

### Decision Tree (`decision_tree.py`)
- sklearn DecisionTreeClassifier
- Max depth: 15 (comparable to binary tree depth ~log2(1000) ≈ 10)
- Min samples split: 20, Min samples leaf: 10

## Running the Benchmark

```bash
# Run BACON
python benchmarks/1000-variable-bool/bacon.py

# Run Decision Tree
python benchmarks/1000-variable-bool/decision_tree.py
```

## Expected Comparison

- **Accuracy**: Both should achieve >95% if expression is learnable
- **Training Time**: Decision Tree likely much faster
- **Interpretability**: BACON provides explicit tree structure matching boolean logic
- **Feature Usage**: Decision Tree may use fewer features (feature selection), BACON uses hierarchical structure
