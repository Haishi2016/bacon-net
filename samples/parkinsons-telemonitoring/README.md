# Parkinson's Telemonitoring Dataset

This sample demonstrates BACON on the Parkinson's Telemonitoring dataset from UCI Machine Learning Repository.

## Dataset Description

The dataset contains biomedical voice measurements from people with early-stage Parkinson's disease. The task is to predict the severity of Parkinson's disease symptoms using voice measurements.

**Target Variable:**
- `motor_UPDRS`: Motor function score (binary classification: high vs low severity)
  - High severity: >= 75th percentile
  - Low severity: < 75th percentile

**Features (20):**
- `subject#`: Subject identifier
- `age`: Age of the patient
- `sex`: Sex (0=male, 1=female)
- `test_time`: Time since baseline recording
- Voice measurements: Jitter, Shimmer, NHR, HNR, RPDE, DFA, PPE (various acoustic features)

**Dataset Stats:**
- Total samples: ~5,875 voice recordings
- 42 Parkinson's disease patients
- Multiple recordings per patient over 6 months

## Usage

```bash
python main.py
```

This will:
1. Load and preprocess the Parkinson's Telemonitoring dataset
2. Train a BACON model with identity, negation, and peak transformations
3. Analyze the learned tree structure
4. Optimize classification thresholds
5. Perform feature importance analysis through pruning
6. Generate visualizations

## Model Configuration

- **Aggregator**: Half-weight LSP aggregator
- **Transformations**: Identity, Negation, Peak
- **Training**: Hierarchical permutation learning with 15 attempts
- **Class weighting**: Enabled (balanced learning)

## Expected Results

The model learns to predict Parkinson's disease severity (motor function) using interpretable combinations of voice measurements. Peak transformations capture optimal ranges for acoustic features that indicate disease progression.

## Reference

Dataset: Parkinson's Telemonitoring (UCI ML Repository #189)
- A. Tsanas, M.A. Little, P.E. McSharry, L.O. Ramig (2009)
- "Accurate telemonitoring of Parkinson's disease progression by noninvasive speech tests"
