# Maternal Health Risk Dataset

This sample demonstrates BACON on the Maternal Health Risk dataset from UCI Machine Learning Repository.

## Dataset Description

The dataset contains health data from pregnant women with the goal of predicting high-risk pregnancies.

**Target Variable:**
- `RiskLevel`: Maternal health risk level
  - 0 = Not High Risk (low risk or mid risk)
  - 1 = High Risk

**Features (6):**
- `Age`: Age of pregnant woman (years)
- `SystolicBP`: Systolic blood pressure (mmHg)
- `DiastolicBP`: Diastolic blood pressure (mmHg)
- `BS`: Blood sugar level (mmol/L)
- `BodyTemp`: Body temperature (°F)
- `HeartRate`: Heart rate (beats per minute)

**Dataset Stats:**
- Total samples: ~1,014 pregnant women
- Original risk levels: low risk, mid risk, high risk
- Binary classification: high risk vs not high risk

## Usage

### BACON Model
```bash
python main.py
```

### Baseline Models
```bash
python main-lr.py   # Logistic Regression
python main-rf.py   # Random Forest
python main-xgb.py  # XGBoost
```

## Model Configuration

- **Aggregator**: Half-weight LSP aggregator
- **Transformations**: Identity, Negation, Peak
- **Training**: Standard permutation learning with 10 attempts
- **Class weighting**: Enabled (balanced learning for imbalanced data)

## Expected Results

The model learns to predict high-risk pregnancies using interpretable combinations of vital signs and health measurements. Peak transformations can capture optimal ranges for clinical measurements (e.g., safe blood pressure ranges).

## Reference

Dataset: Maternal Health Risk (UCI ML Repository #863)
- Marzia Ahmed et al. (2020)
- IoT-based risk level prediction model for maternal healthcare

## Runs

| Binary threshold | Accuracy | F1 | AUPRC | File name | Weight penalty | Transformers |
|------------------|----------|----|-------|-----------|----------------|--------------|
| 0.3 | 68.17% | 0.8882 | 0.8270 | assembler-0.3.pth | 1e-4 | I,N|
| 0.4 | 78.38% | 0.7661 | 0.8450 | assembler-0.4.pth | 1e-4 |I,N|
| 0.45 | 72.68% | 0.6893 | 0.8178 | assembler-0.45.pth | 1e-4 | I,N|
| 0.5 | 78.64% | 0.7729 | 0.8577 | assembler-0.5.pth | 1e-4 |I,N|
| 0.5 | 78.64% | 0.7712 | 0.8538 | assembler-0.5-d.pth | 1e-4 | Disabled |
| 0.6 | 72.68% | 0.6922 | 0.8058 | assembler-0.6.pth | 1e-4 |I,N|
| 0.7 | 67.15% | 0.7297 | 0.8381 | assembler-0.7.pth | 1e-4 |I,N|