# Home Credit Default Risk - Loan Approval Decision

This sample uses the Home Credit Default Risk dataset to predict loan approval decisions using BACON. The model learns the underlying logic for determining whether an applicant is likely to repay their loan based on various applicant features.

**NEW**: This sample demonstrates BACON's **transformation layer**, which can learn to apply transformations (identity or negation) to features before aggregation, improving the model's expressiveness.

## Dataset

This example uses the Home Credit Default Risk dataset from Kaggle, which contains information about loan applications including:
- Applicant demographics (age, gender, family status)
- Income and employment information
- Credit history and existing loans
- External data sources
- And more...

### Download Instructions

⚠️ Before running this sample, you need to download the dataset:

1. Go to [Home Credit Default Risk Competition](https://www.kaggle.com/c/home-credit-default-risk/data)
2. Accept the competition rules (you'll need a Kaggle account)
3. Download `application_train.csv`
4. Place the file in this directory (`samples/home-credit-default-risk/`)

## Running the Sample

```bash
python main.py
```

## What the Model Learns

The BACON model attempts to discover the logical rules that determine loan approval based on features such as:

- **AMT_INCOME_TOTAL**: Total income of the applicant
- **AMT_CREDIT**: Credit amount of the loan
- **AMT_ANNUITY**: Loan annuity (regular payment amount)
- **DAYS_BIRTH**: Age of the applicant (in days, negative value)
- **DAYS_EMPLOYED**: How long the applicant has been employed (in days, negative value)
- **EXT_SOURCE_1/2/3**: Normalized scores from external data sources

The model outputs a decision tree structure showing which features are most important and how they combine to make the approval decision.

## Transformation Layer

This sample enables the **transformation layer** feature, which learns whether each feature should be:
- Used as-is (identity: `x`)
- Negated (negation: `1-x`)

This is particularly useful for features where the interpretation might be reversed. For example:
- A feature representing "risk" might be more useful as "1-risk" for approval decisions
- External scores might need inversion based on what they measure

The transformation layer uses a softmax mechanism to learn the best transformation for each feature during training.

## Sample Output

```bash
🏆 Best accuracy: 68.45%

🧠 Logical Aggregation Tree (Left-Associative):

  [DAYS_BIRTH]─0.50────┐
  [EXT_SOURCE_2]─0.50──[a=1.23]─0.50────┐
  [AMT_CREDIT]─0.50─────────────[a=0.98]─0.50────┐
  [AMT_INCOME_TOTAL]─0.50───────────────[a=1.45]──OUTPUT

🔄 Transformation Layer Analysis
======================================================================

📊 Transformation Statistics:
   Identity (x):        25 features (75.8%)
   Negation (1-x):      8 features (24.2%)

🎯 Top Features Using Negation (1-x):
   EXT_SOURCE_2                   (confidence: 98.5%)
   EXT_SOURCE_3                   (confidence: 95.2%)
   REGION_RATING_CLIENT           (confidence: 87.3%)
   
🎯 Top Features Using Identity (x):
   AMT_INCOME_TOTAL               (confidence: 99.1%)
   DAYS_BIRTH                     (confidence: 97.8%)
   AMT_CREDIT                     (confidence: 96.4%)

======================================================================

✅ Accuracy after pruning 1 feature(s): 68.45%
✅ Accuracy after pruning 2 feature(s): 68.23%
✅ Accuracy after pruning 3 feature(s): 67.89%
```

The transformation analysis shows which features the model learned to negate. This provides insight into how the model interprets each feature's contribution to the decision.

## Feature Importance

After training, the model performs progressive feature pruning to identify which features are most critical for the decision. Features that can be removed without significantly impacting accuracy are considered less important.

## Feature Preprocessing - "Level of Truth" Normalization

This sample implements Microsoft Research's preprocessing methodology where all features are normalized to [0,1] "level of truth" values:
- Higher values mean "more true" or "more positive" for the prediction
- Examples:
  - Young age → high value (1.0)
  - Long employment → high value (1.0)
  - High income → high value (1.0)
  - Missing values → neutral (0.5)

This normalization makes features more interpretable as graded logic inputs.

## Notes

- The model uses the `lsp.half_weight` aggregator which implements Logical Scoring of Preference (LSP) aggregation
- **Transformation layer** is enabled to learn feature negations
- The model uses Microsoft Research's exact feature set (33+ features)
- You may need to adjust the `acceptance_threshold`, `max_epochs`, and `attempts` parameters based on your dataset size and complexity
