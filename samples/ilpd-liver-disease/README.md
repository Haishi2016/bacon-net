# ILPD (Indian Liver Patient Dataset)

## Dataset Overview

- **Source**: UCI Machine Learning Repository (ID: 225)
- **Instances**: 583 patients
- **Features**: 10 (biochemical markers + demographic)
- **Target**: Selector (1 = liver disease, 2 = no liver disease)
- **Task**: Binary classification (predict presence of liver disease)

## Dataset Description

This dataset contains patient records from the North East of Andhra Pradesh, India. It includes 416 patients diagnosed with liver disease and 167 patients without liver disease. The dataset is used to study early detection of liver pathology and gender-based disparities in diagnosis.

Patient demographics:
- 441 male patients
- 142 female patients
- Age range: capped at 90 years

## Features

1. **Age** - Age of the patient (years)
2. **Gender** - Male/Female
3. **Total Bilirubin** - Bilirubin level (mg/dL)
4. **Direct Bilirubin** - Direct bilirubin level (mg/dL)
5. **Total Proteins** - Total protein level (g/dL)
6. **Albumin** - Albumin level (g/dL)
7. **A/G Ratio** - Albumin/Globulin ratio
8. **SGPT** - Serum Glutamic Pyruvic Transaminase (IU/L)
9. **SGOT** - Serum Glutamic Oxaloacetic Transaminase (IU/L)
10. **Alkphos** - Alkaline Phosphatase (IU/L)

## Target Variable

- **Selector**: 
  - 1 = Patient has liver disease
  - 2 = Patient does not have liver disease
  - *Note: Scripts convert this to binary (1=disease, 0=no disease)*

## Files

- `main.py` - BACON neural-symbolic classifier
- `main-lr.py` - Logistic Regression baseline
- `main-rf.py` - Random Forest baseline
- `main-xgb.py` - XGBoost baseline

## Usage

```bash
# Run BACON model
python main.py

# Run baseline comparisons
python main-lr.py
python main-rf.py
python main-xgb.py
```

## Preprocessing

- Missing values are filled with median values
- Gender is encoded as: Male=1, Female=0
- Features are normalized using `SigmoidScaler(alpha=3, beta=-1)`
- Target is converted to binary: 1=disease, 0=no disease
- Train/test split: 80/20 with stratification

## Research Context

The dataset has been used to investigate:
- Differences in liver disease between US and Indian patients
- Gender-based disparities in predicting liver disease
- Effectiveness of biochemical markers for male vs female patients

## References

- Straw, I., & Wu, H. (2022). Investigating for bias in healthcare algorithms: a sex-stratified analysis of supervised machine learning models in liver disease prediction. *BMJ Health & Care Informatics*.
- DOI: [10.24432/C5D02C](https://doi.org/10.24432/C5D02C)
