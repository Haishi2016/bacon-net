# AIDS Clinical Trials Group Study 175 Dataset

This sample demonstrates BACON on the AIDS Clinical Trials Group Study 175 dataset from UCI Machine Learning Repository.

## Dataset Description

The dataset contains data from a randomized clinical trial comparing monotherapy with zidovudine (ZDV) to combination therapy with ZDV and other antiretroviral drugs. The goal is to predict patient survival.

**Target Variable:**
- `cid`: Censoring indicator for death
  - 0 = Alive (censored)
  - 1 = Died (failure event)

**Features (27 after one-hot encoding):**

**Demographics:**
- `age`: Age at baseline (years)
- `wtkg`: Weight at baseline (kg)
- `race`: Race (0=white, 1=non-white)
- `gender`: Gender (0=female, 1=male)

**Risk Factors:**
- `hemo`: Hemophilia (0=no, 1=yes)
- `homo`: Homosexual activity (0=no, 1=yes)
- `drugs`: History of IV drug use (0=no, 1=yes)

**Treatment Information (one-hot encoded):**
- `trt_0`: Treatment arm 0 (ZDV only)
- `trt_1`: Treatment arm 1 (ZDV + ddI)
- `trt_2`: Treatment arm 2 (ZDV + Zal)
- `trt_3`: Treatment arm 3 (ddI only)

**Clinical Measurements:**
- `karnof`: Karnofsky score (scale of 0-100, functional impairment)
- `cd40`: CD4 count at baseline
- `cd420`: CD4 count at 20 weeks
- `cd496`: CD4 count at 96 weeks
- `cd80`: CD8 count at baseline
- `cd820`: CD8 count at 20 weeks

**Treatment History:**
- `oprior`: Non-ZDV antiretroviral therapy pre-175 (0=no, 1=yes)
- `z30`: ZDV in the 30 days prior to 175 (0=no, 1=yes)
- `zprior`: Days of prior ZDV therapy
- `preanti`: Days of prior antiretroviral therapy
- `str2`: Antiretroviral history (0=naive, 1=experienced)
- `strat`: Antiretroviral history stratification
- `symptom`: Symptomatic status (0=asymptomatic, 1=symptomatic)
- `treat`: Treatment indicator (0=ZDV only, 1=others)
- `offtrt`: Off-treatment indicator (0=no, 1=yes)

**Other:**
- `r`: Missing CD4 indicator

**Dataset Stats:**
- Total samples: ~2,139 patients
- Class imbalance: Majority alive vs minority died
- Treatment arms: 4 different antiretroviral therapy combinations

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
- **Preprocessing**: Treatment variable (trt) one-hot encoded into 4 features

## Expected Results

The model learns to predict patient survival using interpretable combinations of demographic factors, treatment assignments, clinical measurements (CD4/CD8 counts), and treatment history. The one-hot encoding of treatment arms allows the model to learn differential effects of various antiretroviral therapy combinations.

## Reference

Dataset: AIDS Clinical Trials Group Study 175 (UCI ML Repository #890)
- Hammer et al. (1996)
- "A controlled trial of two nucleoside analogues plus indinavir in persons with human immunodeficiency virus infection and CD4 cell counts of 200 per cubic millimeter or less"
- New England Journal of Medicine
