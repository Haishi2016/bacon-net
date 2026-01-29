# Parkinson's Telemonitoring Dataset

This sample demonstrates BACON on the Parkinson's [Telemonitoring dataset from UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/189/parkinsons+telemonitoring).

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

**Notes:**

- This dataset is challenging in two ways: 1) it's very small, containing only 42 patients. 2) it has multiple rows per patient. This is not a typical input of a decision making model.
- We use cross-validation instead of 60/20/20 split in this case.
- Other thinkings: 1) treat this as a regression instead of classification. 2) use one sample per subject (such as using averaged telemetries).
## Runs

### LR
| Run | Optimal threshold | Accuracy | F1 | ACPRC | Model |
|-----|-------------------|----------|----|-------|--------|
| 1 | 0.48 | 53.26% | 0.4493 | 0.3584 | LR-lbfgs |
| 2 | 0.48 | 53.26% | 0.4493 | 0.3584 | LR-lbfgs |
| 3 | 0.48 | 53.26% | 0.4493 | 0.3584 | LR-lbfgs |
| 4 | 0.48 | 53.26% | 0.4493 | 0.3584 | LR-lbfgs |
| 5 | 0.48 | 53.26% | 0.4493 | 0.3584 | LR-lbfgs |

### RF
| Run | Optimal threshold | Accuracy | F1 | ACPRC | Model |
|-----|-------------------|----------|----|-------|-------|
| 1 | 0.37 | 95.00% | 0.9005 | 0.9645 | RF-200-None |
| 2 | 0.37 | 95.00% | 0.9005 | 0.9645 | RF-200-None |
| 3 | 0.37 | 95.00% | 0.9005 | 0.9645 | RF-200-None |
| 4 | 0.37 | 95.00% | 0.9005 | 0.9645 | RF-200-None |
| 5 | 0.37 | 95.00% | 0.9005 | 0.9645 | RF-200-None |

### XGB
| Run | Optimal threshold | Accuracy | F1 | ACPRC | Model |
|-----|-------------------|----------|----|-------|-------|
| 1 | 0.58 | 98.47% | 0.9692 | 0.9952 |
| 2 | 0.58 | 98.47% | 0.9692 | 0.9952 |
| 3 | 0.58 | 98.47% | 0.9692 | 0.9952 |
| 4 | 0.58 | 98.47% | 0.9692 | 0.9952 |
| 5 | 0.58 | 98.47% | 0.9692 | 0.9952 |

### BACON

| Binary threshold | Weight penalty |  Transformers | Accuracy | F1 | AUPRC | File name | 
|------------------|----------------|---------------|----------|----|-------|-----------|
| 0.5 | 1e-4 | I, N | 68.32% | 0.6889 | 0.5899 | assembler-0.5.pth |