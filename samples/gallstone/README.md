# Gallstone Disease Prediction with BACON

This sample demonstrates using BACON for gallstone disease prediction using bioimpedance and laboratory data.

## Dataset

**Source:** [UCI Machine Learning Repository - Gallstone](https://archive-beta.ics.uci.edu/dataset/1150/gallstone-1)

**Citation:** Esen, I., Arslan, H., Aktürk Esen, S., Gülşen, M., Kültekin, N., & Özdemir, O. (2024). Early prediction of gallstone disease with a machine learning-based method from bioimpedance and laboratory data. *Medicine*, 103(5), e37258. https://doi.org/10.1097/md.0000000000037258

### Dataset Information

- **Instances:** 320 individuals (161 with gallstone disease, 159 without)
- **Features:** 37 clinical measures (demographic, bioimpedance, laboratory)
- **Task:** Binary classification (gallstone disease vs. healthy)
- **Missing Values:** None
- **Class Balance:** Balanced (50.3% positive, 49.7% negative)

### Features

**Demographic (5 features):**
- Age, Gender, Height, Weight, BMI

**Bioimpedance Data (8 features):**
- Total body water (TBW)
- Extracellular water (ECW)
- Intracellular water (ICW)
- Muscle mass
- Fat mass
- Protein
- Visceral fat area (VFA)
- Hepatic fat

**Laboratory Measures (12 features):**
- Glucose
- Total cholesterol, HDL, LDL, Triglycerides
- AST, ALT, ALP (liver enzymes)
- Creatinine, GFR (kidney function)
- CRP (inflammation marker)
- Hemoglobin
- Vitamin D

### Target Variable

**Gallstone** - Binary gallstone disease diagnosis
- 0 = No gallstone disease
- 1 = Gallstone disease

## Running the Sample

```bash
cd samples/gallstone
python main.py
```

## Measurements
|Run | Binary threshold | Aggregator | Weights| Weights penalty| Optimum threshold | Accuracy | AUPRC|
|----|-----------|------------|--------|----------------|-------------------|----------|------|
| 1 | 0.8 | half_weight | trainable | 1e-3 | 0.465 | 75.00% | 0.8438 |
| 2 | 0.3 | half_weight | trainable | 1e-3 | 0.676 | 75.00% | 0.8176 |
| 3 | 0.5 | half_weight | trainable | 1e-3 | 0.5 | 76.56% | 0.8328 | 