# Structural Stability Comparison

This directory contains scripts to compare structural stability of different interpretable models under noise.

## Motivation

We evaluate **structural consistency** by measuring how the learned explanation of the same model changes as noise increases. We compare the explanation learned at noise level $r$ with the explanation learned on clean data ($r=0$) using **feature-set consistency**:

$$S_F(r) = \frac{|F_0 \cap F_r|}{|F_0 \cup F_r|}$$

where $F_0$ and $F_r$ denote the sets of features actively used by the model at noise levels $r=0$ and $r$, respectively.

## Models Compared

Each model has a different definition of "features used":

1. **BACON** (`bacon-breast-cancer-with-noise.py`): Features in the learned tree structure (ordered by `locked_perm`)
2. **Decision Tree** (`dt-breast-cancer-with-noise.py`): Features appearing in tree splits
3. **RuleFit** (`rulefit-breast-cancer-with-noise.py`): Features appearing in nonzero-weight rules (above threshold)
4. **PySR** (`pysr-breast-cancer-with-noise.py`): Symbols (features) present in the final symbolic expression
5. **EBM** (`ebm-breast-cancer-with-noise.py`): Features with importance above threshold

## Usage

### Step 1: Run experiments at different noise levels

For each model, edit the `NOISE_RATIO` variable at the top of the file and run:

```bash
# For BACON
# Edit noise_ratio in bacon-breast-cancer-with-noise.py (line 42)
python bacon-breast-cancer-with-noise.py

# For Decision Tree  
# Edit NOISE_RATIO in dt-breast-cancer-with-noise.py (line 13)
python dt-breast-cancer-with-noise.py

# For RuleFit
# Edit NOISE_RATIO in rulefit-breast-cancer-with-noise.py (line 19)
python rulefit-breast-cancer-with-noise.py

# For PySR (requires Julia)
# Edit NOISE_RATIO in pysr-breast-cancer-with-noise.py (line 17)
python pysr-breast-cancer-with-noise.py

# For EBM
# Edit NOISE_RATIO in ebm-breast-cancer-with-noise.py (line 17)
python ebm-breast-cancer-with-noise.py
```

**Recommended noise levels**: 0.0, 0.1, 0.2, 0.3, 0.4, 0.5

Each script will save results to a JSON file (e.g., `bacon_results_noise_0.0.json`).

### Step 2: Analyze structural stability

After running experiments for all noise levels and models of interest:

```bash
python analyze_structural_stability.py
```

This will:
- Compute $S_F(r)$ for each model and noise level
- Generate comparison plots (saved to `structural_stability_comparison.png`)
- Print summary tables showing:
  - Feature set consistency vs noise
  - Test accuracy vs noise  
  - Number of features used vs noise

## Example Workflow

```bash
# Run BACON at all noise levels
for noise in 0.0 0.1 0.2 0.3 0.4 0.5; do
    # Edit bacon-breast-cancer-with-noise.py to set noise_ratio = $noise
    python bacon-breast-cancer-with-noise.py
done

# Run Decision Tree at all noise levels
for noise in 0.0 0.1 0.2 0.3 0.4 0.5; do
    # Edit dt-breast-cancer-with-noise.py to set NOISE_RATIO = $noise
    python dt-breast-cancer-with-noise.py
done

# (Repeat for other models...)

# Analyze results
python analyze_structural_stability.py
```

## Output Files

Each model script generates:
- `{model}_results_noise_{r}.json`: Results at noise level r
  - `noise_ratio`: Noise level
  - `test_accuracy`: Test set accuracy
  - `features_used`: List of features used by the model
  - `num_features_used`: Count of features
  - Additional model-specific metrics

The analysis script generates:
- `structural_stability_comparison.png`: Visualization comparing all models

## Installation Requirements

```bash
# Core requirements
pip install scikit-learn torch pandas numpy matplotlib ucimlrepo

# RuleFit (optional)
pip install rulefit
# OR
pip install git+https://github.com/christophM/rulefit.git

# PySR (optional, requires Julia)
pip install pysr
python -m pysr install

# EBM (optional)
pip install interpret
```

## Key Metrics

- **$S_F(r)$**: Higher values indicate more stable feature selection across noise levels
- **Test Accuracy**: Model performance under noise
- **Feature Count**: Number of features actively used by the model

## Notes

- All models use the same random seed (42) for reproducibility
- All models use the same train/test split
- Noise is applied uniformly: each sample has `noise_ratio * num_features` features corrupted
- Feature values are replaced with uniform random samples from [min, max] of that feature
