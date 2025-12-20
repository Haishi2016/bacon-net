# CDC Diabetes Health Indicators Classification with BACON

This sample demonstrates using BACON for diabetes risk prediction using the CDC Diabetes Health Indicators dataset.

## Dataset

**Source:** [UCI Machine Learning Repository - CDC Diabetes Health Indicators](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators)

**Citation:** CDC Diabetes Health Indicators [Dataset]. (2017). UCI Machine Learning Repository. https://doi.org/10.24432/C53919

### Dataset Information

- **Instances:** 253,680 survey respondents
- **Features:** 21 health and lifestyle indicators
- **Task:** Binary classification (diabetes/prediabetes vs. healthy)
- **Missing Values:** None

### Features

All features are binary (0/1) or ordinal integers:

1. **HighBP** - High blood pressure
2. **HighChol** - High cholesterol
3. **CholCheck** - Cholesterol check in past 5 years
4. **BMI** - Body Mass Index (integer)
5. **Smoker** - Smoked at least 100 cigarettes lifetime
6. **Stroke** - Ever had a stroke
7. **HeartDiseaseorAttack** - Coronary heart disease or MI
8. **PhysActivity** - Physical activity in past 30 days
9. **Fruits** - Consume fruit 1+ times per day
10. **Veggies** - Consume vegetables 1+ times per day
11. **HvyAlcoholConsump** - Heavy alcohol consumption
12. **AnyHealthcare** - Have any health care coverage
13. **NoDocbcCost** - Could not see doctor due to cost
14. **GenHlth** - General health (1=excellent to 5=poor)
15. **MentHlth** - Days of poor mental health (0-30)
16. **PhysHlth** - Days of poor physical health (0-30)
17. **DiffWalk** - Difficulty walking or climbing stairs
18. **Sex** - Sex (0=female, 1=male)
19. **Age** - Age category (1-13, binned)
20. **Education** - Education level (1-6)
21. **Income** - Income level (1-8)

### Target Variable

**Diabetes_binary** - Binary diabetes diagnosis
- 0 = No diabetes
- 1 = Prediabetes or diabetes

## Running the Sample

```bash
cd samples/diabetes
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
- **Sample Size:** 10,000 (from 253k total, for computational efficiency)
- **Acceptance Threshold:** 85%

## Data Preprocessing

**Important:** Unlike the heart disease dataset:
- No one-hot encoding needed (all features already numeric)
- Uses full 253,680 instances for training
- Same train/test split (80/20) and random seed (42) as heart disease

## Expected Results

The model typically achieves:
- **Test Accuracy:** 75-85%
- **Interpretability:** Clear decision tree showing which health factors predict diabetes
- **Feature Importance:** Ranking of health indicators by contribution

## Analysis Features

The script provides:

1. **Tree Visualization:** Visual representation of the learned health risk assessment structure
2. **Threshold Optimization:** Finds optimal classification thresholds for different metrics
3. **Feature Importance:** Analyzes each health indicator's contribution through pruning
4. **Performance Metrics:** Comprehensive evaluation including precision, recall, F1-score
5. **Prediction Analysis:** Visualizations of model predictions and errors

## Key Findings

The analysis reveals:
- Which health indicators are most predictive of diabetes risk
- How lifestyle and health factors combine in risk assessment
- The logical structure behind diabetes prediction
- Which factors could be prioritized for prevention

## Clinical Relevance

This interpretable model can help:
- Identify high-risk populations for diabetes screening
- Understand which modifiable risk factors (e.g., BMI, physical activity) have greatest impact
- Develop targeted intervention strategies
- Provide explainable predictions for individual risk assessments

## Comparison with Baselines

Additional scripts are provided for comparison:
- `main-lr.py` - Logistic Regression baseline
- `main-rf.py` - Random Forest baseline
- `main-xgb.py` - XGBoost baseline

All use identical data preprocessing and splits for fair comparison.

## Notes

- Uses full dataset with 253,680 instances
- All features are already numeric, no encoding required
- Dataset is from CDC's Behavioral Risk Factor Surveillance System (BRFSS) 2014
- Class distribution is imbalanced (~15% diabetes, handled via class weighting)

## References

Burrows, N. R., Hora, I., Geiss, L. S., Gregg, E. W., & Albright, A. (2017). Incidence of End-Stage Renal Disease Attributed to Diabetes Among Persons with Diagnosed Diabetes. *Morbidity and Mortality Weekly Report*, 66(43).
