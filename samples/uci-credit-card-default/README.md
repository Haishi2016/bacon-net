# UCI Credit Card Default (Taiwan) - BACON Benchmark

This sample uses the UCI "Default of Credit Card Clients" dataset to demonstrate BACON's interpretable credit risk prediction capabilities and provides direct comparison with published baseline methods.

## Why This Dataset?

The UCI Credit Card Default dataset is an **ideal benchmark** for comparing BACON against both traditional ML and modern XAI methods because:

✅ **Single clean table** - No complex multi-table joins or feature engineering required  
✅ **Well-established baselines** - 15+ years of published results (Yeh & Lien 2009 onwards)  
✅ **Modern XAI comparisons** - Recent papers use XGBoost, LightGBM with SHAP/LIME  
✅ **Interpretable features** - Clear semantics (payment history, bill amounts, demographics)  
✅ **Standard benchmark** - Widely used in ML and explainable AI research  

## Dataset Overview

**30,000 credit card clients** from Taiwan (April-September 2005)

### Features (24 total):
- **LIMIT_BAL**: Credit limit
- **SEX**: Gender (1=male, 2=female)
- **EDUCATION**: Education level (1=graduate, 2=university, 3=high school, 4=others)
- **MARRIAGE**: Marital status (1=married, 2=single, 3=others)
- **AGE**: Age in years
- **PAY_0 to PAY_6**: Payment status for past 6 months (-1=pay duly, 1=delay 1 month, 2=delay 2 months, etc.)
- **BILL_AMT1 to BILL_AMT6**: Bill statement amounts for past 6 months
- **PAY_AMT1 to PAY_AMT6**: Payment amounts for past 6 months

### Target:
- **default.payment.next.month**: 1=default, 0=no default
- **Class distribution**: ~22% default rate (imbalanced)

## Download Instructions

1. Go to [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients)
2. Download the dataset (Excel file: `default of credit card clients.xls`)
3. Place it in this directory: `samples/uci-credit-card-default/`

**OR** use the direct download link:
```bash
cd samples/uci-credit-card-default
curl -o "default of credit card clients.xls" "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"
```

## Published Baselines to Compare

### Traditional ML (Yeh & Lien 2009 and follow-up studies):
- **Logistic Regression**: ~0.77-0.78 AUC
- **Decision Trees**: ~0.65-0.70 AUC
- **Random Forest**: ~0.76-0.78 AUC
- **SVM**: ~0.75-0.77 AUC
- **k-NN**: ~0.73-0.75 AUC

### Modern ML & Ensemble Methods (2015-2025):
- **XGBoost**: ~0.77-0.80 AUC
- **LightGBM**: ~0.77-0.79 AUC
- **CatBoost**: ~0.77-0.79 AUC
- **Deep Neural Networks**: ~0.76-0.79 AUC

### XAI Methods (2020-2025):
- **XGBoost + SHAP**: ~0.78-0.80 AUC (with feature importance)
- **Interpretable Boosting Machine (InterpretML)**: ~0.77-0.79 AUC
- **LIME with various models**: Similar accuracy with local explanations
- **Transparent Neural Networks**: ~0.76-0.78 AUC

### Key Papers to Cite:
1. **Yeh & Lien (2009)** - Original dataset paper: "The comparisons of data mining techniques for the predictive accuracy of probability of default of credit card clients"
2. **Recent XAI studies** (2020-2025) using this dataset with XGBoost, SHAP, LIME

## Running the Sample

```bash
# Basic run
python main.py

# With benchmarking against traditional ML
python benchmark_comparison.py
```

## What BACON Provides

Unlike black-box models, BACON learns:
- **Logical tree structure** showing feature combination rules
- **Graded truth values** [0,1] for each feature
- **Transformation layer** (identity vs negation) for feature interpretation
- **Hierarchical feature selection** through permutation search
- **Fully interpretable** decision logic

## Expected Results

Based on BACON's performance on similar datasets:
- **Target AUC**: 0.75-0.78 (competitive with Random Forest, XGBoost)
- **Accuracy**: ~0.78-0.82
- **Advantage**: Fully interpretable logical rules + graded truth semantics
- **Comparison**: Similar predictive power with superior interpretability

## Benchmark Metrics

All experiments report:
- **AUC-ROC** (primary metric - handles class imbalance)
- **Accuracy**
- **Precision/Recall**
- **F1-Score**
- **Interpretability**: Rule complexity, tree depth, number of features used

## Sample Output

```
🏆 Best Test AUC: 0.7745

🧠 Logical Aggregation Tree:

  [PAY_0]─0.62────┐
  [PAY_2]─0.38────[a=1.15]─0.55────┐
  [PAY_3]─0.45────────────[a=0.92]─0.48────┐
  [LIMIT_BAL]─0.52──────────────[a=1.28]──OUTPUT

🔄 Transformation Analysis:
   PAY_0, PAY_2, PAY_3: Identity (higher payment delay = higher default risk)
   LIMIT_BAL: Negation (higher limit = lower default risk)
```

## References

Original dataset:
```
Yeh, I. C., & Lien, C. H. (2009). The comparisons of data mining techniques for 
the predictive accuracy of probability of default of credit card clients. 
Expert Systems with Applications, 36(2), 2473-2480.
```

UCI Repository:
```
Dua, D. and Graff, C. (2019). UCI Machine Learning Repository 
[http://archive.ics.uci.edu/ml]. Irvine, CA: University of California, 
School of Information and Computer Science.
```
