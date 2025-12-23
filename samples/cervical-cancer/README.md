# Cervical Cancer Behavior Risk Prediction

This sample demonstrates using BACON for predicting cervical cancer risk based on behavioral and psychological factors.

## Dataset

- **Source**: UCI Machine Learning Repository (ID: 537)
- **Instances**: 72 respondents
- **Features**: 19 behavioral/psychological measures (all integer scores)
- **Target**: `ca_cervix` (0 = no cervical cancer, 1 = cervical cancer)
- **Task**: Binary classification

## Features

The dataset includes 19 behavioral and psychological factors grouped into categories:

### Behavioral Factors (3 features)
- **behavior_eating**: Eating behavior score
- **behavior_personalHygiene**: Personal hygiene behavior score  
- **behavior_sexualRisk**: Sexual risk behavior score

### Intention (2 features)
- **intention_aggregation**: Aggregated intention score
- **intention_commitment**: Commitment intention score

### Attitude (2 features)
- **attitude_consistency**: Attitude consistency score
- **attitude_spontaneity**: Attitude spontaneity score

### Social Norms (2 features)
- **norm_significantPerson**: Significant person norm score
- **norm_fulfillment**: Norm fulfillment score

### Risk Perception (2 features)
- **perception_vulnerability**: Perceived vulnerability score
- **perception_severity**: Perceived severity score

### Motivation (2 features)
- **motivation_strength**: Motivation strength score
- **motivation_willingness**: Motivation willingness score

### Social Support (3 features)
- **socialSupport_emotionality**: Emotional social support score
- **socialSupport_appreciation**: Appreciation social support score
- **socialSupport_instrumental**: Instrumental social support score

### Empowerment (3 features)
- **empowerment_knowledge**: Empowerment knowledge score
- **empowerment_abilities**: Empowerment abilities score
- **empowerment_desires**: Empowerment desires score

## Model Configuration

Given the small dataset size (72 instances), the model uses adapted parameters:

- **Scaler**: `SigmoidScaler(alpha=2, beta=-1)` for gentler normalization
- **Weight mode**: Trainable (allows more flexibility for small data)
- **Training attempts**: 20 (more attempts to find good initialization)
- **Hierarchical group size**: 8 (smaller groups for 19 features)
- **Epochs**: 3000 per attempt (shorter for small dataset)
- **Test split**: 25% (maintains reasonable train/test sizes)
- **Acceptance threshold**: 85% (adjusted for small sample difficulty)

## Expected Results

Due to the very small dataset size (72 instances with 19 features), results may vary:

- **Test Accuracy**: 60-85% (high variance expected)
- **Important Feature Groups**: Likely behavioral factors, risk perception, and empowerment
- **Overfitting Risk**: High due to small sample size - monitor train vs test performance carefully

## Running the Sample

```bash
cd samples/cervical-cancer
python main.py
```

Compare with baselines:
```bash
python main-lr.py
python main-rf.py  
python main-xgb.py
```

## Key Findings

The BACON model can identify:

1. **Critical behavioral predictors**: Which behaviors most strongly associate with cancer risk
2. **Psychological factors**: Attitudes, perceptions, and motivations that matter most
3. **Social influences**: Role of social support and norms
4. **Empowerment dimensions**: Knowledge, abilities, or desires most protective

The interpretable tree structure reveals how these factors combine in risk assessment logic.

## Clinical Relevance

Understanding behavioral and psychological risk factors can inform:
- **Prevention programs**: Target modifiable behaviors
- **Education campaigns**: Address knowledge gaps and misconceptions
- **Support interventions**: Strengthen protective social factors
- **Risk screening**: Identify high-risk individuals for closer monitoring

## Notes

- **Small Sample Size**: Results should be interpreted with caution due to limited data
- **Cross-validation**: Consider using k-fold CV for more stable estimates
- **Feature Engineering**: Original scores are integers - consider if normalization/scaling is appropriate
- **Class Balance**: Check if dataset is balanced or imbalanced
- **Stratification**: Train/test split is stratified to maintain class proportions

## Citation

Cervical Cancer Behavior Risk. (2019). UCI Machine Learning Repository. https://doi.org/10.24432/C5402W
