# Noise Resilience Benchmarks

This folder contains tests for evaluating how well BACON models maintain performance and structural consistency when input features are corrupted with noise.

## Metrics

### Performance Stability: nAUDC

**Normalized Area Under Degradation Curve** measures how gracefully model accuracy degrades as noise increases:

```
nAUDC = (1 / r_max * Acc(0)) * ∫₀^r_max Acc(r) dr
```

- **Range**: [0, 1]
- **Higher is better**: nAUDC = 1.0 means no performance degradation
- **Interpretation**: Models with higher nAUDC are more robust to noise

### Structural Stability: SF(r)

**Structural Fidelity** measures consistency of learned explanations across noise levels:

```
SF(r) = |F₀ ∩ Fᵣ| / |F₀ ∪ Fᵣ|
```

Where:
- F₀ = features used by model trained on clean data
- Fᵣ = features used by model trained on data with noise ratio r
- Measured as Jaccard similarity

- **Range**: [0, 1]
- **Higher is better**: SF = 1.0 means perfect structural consistency
- **Interpretation**: Models with high SF maintain stable explanations despite noise

## Noise Model

Uniform noise is applied across all input dimensions:

1. For each sample, randomly select ⌊r·d⌋ features (where r = noise ratio, d = dimension)
2. Replace selected features with independent draws from Uniform(0, 1)
3. Noise is applied to both relevant and irrelevant features

This tests whether the model has truly learned which features matter vs. just memorizing patterns.

## Tests

### `boolean.py`

Tests noise resilience on boolean expression: `y = (x0 ∧ x1) ∨ (x2 ∧ x3)`

- **Input size**: 10 features (only first 4 relevant)
- **Expected behavior**: Model should primarily use features [0, 1, 2, 3]
- **Noise ratios**: 0.0 to 0.8 in steps of 0.1
- **Outputs**: 
  - Degradation curve plot
  - Structural fidelity plot
  - nAUDC score
  - JSON results file

**Run test:**
```bash
cd benchmarks/noise-resilience
python boolean.py
```

## Utilities

`noise_utils.py` provides reusable functions:

- `add_uniform_noise()`: Add uniform noise to inputs
- `compute_nAUDC()`: Calculate normalized area under degradation curve
- `extract_active_features()`: Get features used by trained model
- `compute_structural_fidelity()`: Calculate Jaccard similarity between feature sets
- `evaluate_noise_resilience()`: Complete evaluation pipeline

These utilities can be imported and used in other benchmark scenarios.

## Future Extensions

Potential additional tests:

1. **Adversarial Noise**: Targeted corruption of specific features
2. **Gaussian Noise**: Continuous noise instead of replacement
3. **Missing Features**: Test with completely absent features
4. **Cross-Dataset**: Train on clean, test on noisy (and vice versa)
5. **Feature Importance**: Corrupt relevant vs irrelevant features separately

## References

Based on evaluation methodology from:
- Noise robustness in neural-symbolic reasoning
- Explanation stability under perturbations
- Feature attribution consistency metrics
