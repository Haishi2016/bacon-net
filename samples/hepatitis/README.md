# Hepatitis Dataset

This sample demonstrates BACON on the Hepatitis dataset from UCI Machine Learning Repository.

## Dataset Description

The dataset contains data from hepatitis patients with the goal of predicting survival outcome.

**Target Variable:**
- `Class`: Patient outcome
  - 0 = Live (survived)
  - 1 = Die (deceased)

**Features (19):**
- `AGE`: Age of patient (years)
- `SEX`: Sex (1=male, 2=female)
- `STEROID`: Steroid treatment (1=no, 2=yes)
- `ANTIVIRALS`: Antiviral treatment (1=no, 2=yes)
- `FATIGUE`: Fatigue symptom (1=no, 2=yes)
- `MALAISE`: Malaise symptom (1=no, 2=yes)
- `ANOREXIA`: Anorexia symptom (1=no, 2=yes)
- `LIVER BIG`: Liver big (1=no, 2=yes)
- `LIVER FIRM`: Liver firm (1=no, 2=yes)
- `SPLEEN PALPABLE`: Spleen palpable (1=no, 2=yes)
- `SPIDERS`: Spider angiomas (1=no, 2=yes)
- `ASCITES`: Ascites (1=no, 2=yes)
- `VARICES`: Esophageal varices (1=no, 2=yes)
- `BILIRUBIN`: Serum bilirubin (mg/dL)
- `ALK PHOSPHATE`: Alkaline phosphatase (U/L)
- `SGOT`: Serum glutamic-oxaloacetic transaminase (U/L)
- `ALBUMIN`: Serum albumin (g/dL)
- `PROTIME`: Prothrombin time (seconds)
- `HISTOLOGY`: Liver histology (1=no, 2=yes)

**Dataset Stats:**
- Total samples: 155 patients
- Class imbalance: Majority class (Live) vs minority class (Die)
- Missing values: Handled by median imputation

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
- **Transformations**: Identity, Negation
- **Training**: Standard permutation learning with 10 attempts
- **Class weighting**: Enabled (balanced learning for imbalanced data)

## Expected Results

The model learns to predict patient survival using interpretable combinations of clinical symptoms and laboratory measurements. Class weighting helps handle the class imbalance in the dataset.

## Reference

Dataset: Hepatitis (UCI ML Repository #46)
- G. Gong (1988)
- Carnegie-Mellon University hepatitis database

