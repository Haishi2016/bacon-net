# Ablation Studies

This folder contains ablation studies to understand the impact of various hyperparameters on BACON's performance.

## Sparsity Hardening Study

**File**: `test_sparsity_hardening.py`

**Purpose**: Investigate how the `loss_weight_perm_sparsity` parameter affects the model's ability to maintain accuracy when the soft permutation matrix is hardened (converted to discrete assignments via argmax).

### Hypothesis

- **Low sparsity weight**: Model learns good soft assignments but struggles when hardened
- **High sparsity weight**: Model learns near-discrete assignments that harden well
- **Trade-off**: There may be a sweet spot that balances soft accuracy with hardening robustness

### Methodology

1. Use 100-variable boolean expressions (faster than 1000-variable benchmark)
2. Train models with different `loss_weight_perm_sparsity` values: [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 20.0, 50.0]
3. For each model:
   - Measure **soft accuracy**: accuracy with continuous permutation during inference
   - Measure **hard accuracy**: accuracy after locking permutation to argmax
   - Calculate **accuracy drop**: difference between soft and hard
   - Record **confidence**: average max probability per position
   - Record **uniqueness**: number of unique columns assigned

### Key Metrics

- **Soft Accuracy**: Test accuracy with Gumbel-Sinkhorn soft permutation
- **Hard Accuracy**: Test accuracy after `lock_permutation()` converts to discrete
- **Accuracy Drop**: Absolute difference (soft - hard)
- **Relative Drop**: Percentage drop relative to soft accuracy
- **Confidence**: Average max probability in permutation matrix (higher = more peaked)
- **Uniqueness**: Fraction of unique column assignments (1.0 = perfect permutation)

### Expected Results

**Low Sparsity (0.01 - 0.1)**:
- High soft accuracy
- Low confidence
- Large accuracy drop when hardened
- Poor explainability

**Medium Sparsity (0.5 - 5.0)**:
- Moderate soft accuracy
- Moderate confidence
- Moderate accuracy drop
- Balanced trade-off

**High Sparsity (10.0 - 50.0)**:
- Slightly lower soft accuracy
- High confidence
- Small accuracy drop
- Good explainability

### Configuration

```python
input_size = 100
num_epochs = 3000
annealing_epochs = 1500
freeze_aggregation_epochs = 500
acceptance_threshold = 0.95
```

### Usage

```bash
cd benchmarks/ablation
python test_sparsity_hardening.py
```

### Output

1. **JSON Results**: `sparsity_ablation_results_[timestamp].json`
   - Detailed metrics for each sparsity value
   
2. **Visualization**: `sparsity_ablation_plot_[timestamp].png`
   - 4-panel plot showing:
     - Soft vs Hard accuracy
     - Accuracy drop (absolute and relative)
     - Confidence vs sparsity weight
     - Uniqueness vs sparsity weight

3. **Summary Table**: Console output with all metrics

4. **Recommendations**: Optimal sparsity weight that minimizes relative drop while maintaining >95% soft accuracy

### Implications

This study helps answer:
- What sparsity weight should be used for production models?
- Is there a trade-off between soft accuracy and hardening robustness?
- How does confidence correlate with hardening performance?
- Can we predict accuracy drop from confidence metrics?

### Future Extensions

Potential follow-up studies:
- **Dynamic sparsity scheduling**: Test different (initial, final) combinations
- **Aggregation freezing duration**: Vary `freeze_aggregation_epochs`
- **Temperature schedules**: Different annealing strategies
- **Dataset complexity**: Compare simple vs complex boolean expressions
- **Scaling effects**: Repeat with 1000 variables
