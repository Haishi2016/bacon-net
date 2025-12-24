# Heart Disease Classification with BACON

This sample demonstrates using BACON for heart disease diagnosis prediction using the UCI Heart Disease dataset (Cleveland database).

## Dataset

**Source:** [UCI Machine Learning Repository - Heart Disease Dataset](https://archive.ics.uci.edu/dataset/45/heart+disease)

**Citation:** Janosi, A., Steinbrunn, W., Pfisterer, M., & Detrano, R. (1989). Heart Disease [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C52P4X

### Dataset Information

- **Instances:** 303 patients (Cleveland database)
- **Features:** 13 clinical measurements
- **Task:** Binary classification (presence vs. absence of heart disease)
- **Missing Values:** Yes (handled by dropping incomplete records)

### Features

1. **age** - Age in years
2. **sex** - Sex (1 = male; 0 = female)
3. **cp** - Chest pain type (1-4)
   - Value 1: typical angina
   - Value 2: atypical angina
   - Value 3: non-anginal pain
   - Value 4: asymptomatic
4. **trestbps** - Resting blood pressure (mm Hg)
5. **chol** - Serum cholesterol (mg/dl)
6. **fbs** - Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)
7. **restecg** - Resting electrocardiographic results (0-2)
   - Value 0: normal
   - Value 1: ST-T wave abnormality
   - Value 2: left ventricular hypertrophy
8. **thalach** - Maximum heart rate achieved
9. **exang** - Exercise induced angina (1 = yes; 0 = no)
10. **oldpeak** - ST depression induced by exercise relative to rest
11. **slope** - Slope of the peak exercise ST segment (1-3)
    - Value 1: upsloping
    - Value 2: flat
    - Value 3: downsloping
12. **ca** - Number of major vessels (0-3) colored by fluoroscopy
13. **thal** - Thalassemia (3 = normal; 6 = fixed defect; 7 = reversible defect)

### Target Variable

**num** - Diagnosis of heart disease (0-4)
- Value 0: < 50% diameter narrowing (no disease)
- Values 1-4: > 50% diameter narrowing (disease present)

For binary classification, we convert this to:
- 0 = no disease
- 1 = disease present

## Running the Sample

```bash
cd samples/heart-disease
python main.py
```

## Model Configuration

The sample uses the following BACON configuration:

- **Aggregator:** `lsp.half_weight` (Logic Scoring of Preference - Half Weight)
- **Weight Mode:** `fixed` 
- **Loss Amplifier:** 1000
- **Sinkhorn Iterations:** 200 (for permutation convergence)
- **Class Weighting:** Enabled (handles potential class imbalance)
- **Acceptance Threshold:** 90% (model must achieve 90% test accuracy)
- **Transformation:** Enabled (Identity, Negation)

## Key Findings

In the heart disease task, BACON identifies **asymptomatic chest pain**(`cp`=4) as the most decisive feature, followed by sex (`sex`) as a conditional modifier. Notably, the model does not induce a global baseline condition. This reflects the heterogeneity of cardiac risk presentation in the dataset, where diagnosis is driven by high-signal symptom patterns rather than a shared background state. The absence of an artificial baseline highlights BACON’s ability to adapt its symbolic structure to the intrinsic causal geometry of the domain.

## Clinical Relevance

Heart disease exists primarily through decisive symptom manifestation, rather than gradual accumulation from a shared baseline. This matches how cardiology often works:
* Symptoms trump demographics
* Risk factors modulate, but don’t replace, symptoms

## Saved Models

| Model file | Threshold | Accuracy | Precision | Recall |
|--------|--------|--------|--------|--------|
| assembler.pth | 0.513 | 86.53% | 92.17% | 77.37% |

## Notes

- The dataset contains missing values (marked as '?'), which are handled by dropping incomplete records
- The original dataset has 76 attributes, but clinical ML research typically uses only these 14 attributes
- This is a well-studied benchmark dataset for interpretable medical diagnosis
- Results may vary between runs due to random initialization and data splitting
