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

## Model Configuration

The sample uses the following BACON configuration:

- **Aggregator:** `lsp.half_weight` (Logic Scoring of Preference)
- **Weight Mode:** `trainable`
- **Weight Normalization:** `softmax`
- **Loss Amplifier:** 1000
- **Sinkhorn Iterations:** 200
- **Class Weighting:** Enabled
- **Acceptance Threshold:** 90%

## Data Preprocessing

**Important:**
- All features are continuous numeric values (no categorical encoding needed)
- SigmoidScaler used with alpha=3, beta=-1 for feature normalization
- Balanced dataset (no resampling needed)
- Train/test split: 80/20 with random_state=42

## Expected Results

The model typically achieves:
- **Test Accuracy:** 85-95%
- **Interpretability:** Clear decision tree showing which biomarkers and lab values predict gallstone risk
- **Feature Importance:** Ranking of clinical measures by contribution

## Analysis Features

The script provides:

1. **Tree Visualization:** Visual representation of the learned clinical risk assessment structure
2. **Threshold Optimization:** Finds optimal classification thresholds for different metrics
3. **Feature Importance:** Analyzes each clinical measure's contribution through pruning
4. **Overfitting Check:** Compares train vs test performance to detect data leakage
5. **Performance Metrics:** Comprehensive evaluation including precision, recall, F1-score
6. **Prediction Analysis:** Visualizations of model predictions and errors

## Key Findings

The analysis can reveal:
- Which bioimpedance measures are most predictive of gallstone disease
- How laboratory values (liver enzymes, cholesterol, etc.) combine in risk assessment
- The logical structure behind gallstone prediction
- Which modifiable factors (BMI, hepatic fat, cholesterol) have greatest impact

## Clinical Relevance

This interpretable model can help:
- Identify high-risk individuals for gallstone screening using non-imaging features
- Understand which bioimpedance and lab markers are most important
- Develop targeted intervention strategies for modifiable risk factors
- Provide explainable predictions for individual risk assessments
- Reduce unnecessary imaging procedures through accurate pre-screening

## Comparison with Baselines

To compare with traditional ML methods, you can create baseline scripts:
- `main-lr.py` - Logistic Regression baseline
- `main-rf.py` - Random Forest baseline
- `main-xgb.py` - XGBoost baseline

All should use identical data preprocessing and splits for fair comparison.

## Notes

- Dataset collected from Ankara VM Medical Park Hospital (June 2022–June 2023)
- Ethically approved by Ankara City Hospital Ethics Committee (E2-23-4632)
- Complete dataset with no missing values simplifies preprocessing
- Balanced classes eliminate need for special handling of class imbalance
- Features combine demographic, body composition, and laboratory data

## References

Esen, I., Arslan, H., Aktürk Esen, S., Gülşen, M., Kültekin, N., & Özdemir, O. (2024). Early prediction of gallstone disease with a machine learning-based method from bioimpedance and laboratory data. *Medicine*, 103(5), e37258.
