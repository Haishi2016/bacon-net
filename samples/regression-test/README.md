# Regression Test Sample

A simple test case for BACON's regression capabilities using synthetic continuous data.

## Dataset

- **Samples**: 20 rows
- **Features**: 5 continuous input features
- **Target**: 1 continuous output (range: 0.001 to 0.999)
- **Task**: Function approximation / regression

## Purpose

This sample demonstrates BACON's ability to:
1. Learn continuous mappings (not just binary classification)
2. Approximate complex functions with small datasets
3. Provide interpretable tree structures for regression

## Data Structure

The synthetic data consists of:
- **Feature_1** to **Feature_5**: Input features with values in [0, 1]
- **Target**: Continuous output to approximate

Since this is a small controlled dataset (20 samples), we use all data for both training and testing to assess pure approximation capability rather than generalization.

## Usage

```bash
cd samples/regression-test
python main.py
```

## Model Configuration

- **Aggregator**: `math.avg` (suitable for continuous outputs)
- **Weight Mode**: `learnable` (allows flexible function learning)
- **Normalize Andness**: `True`
- **Loss Amplifier**: 100

## Metrics

The script reports:
- **MSE** (Mean Squared Error)
- **MAE** (Mean Absolute Error)
- **RMSE** (Root Mean Squared Error)
- Sample predictions with actual vs predicted values

## Expected Behavior

With the small dataset and proper configuration, BACON should be able to:
- Achieve high approximation accuracy (low MSE/MAE)
- Learn an interpretable tree structure
- Demonstrate that the aggregator-based approach works for regression
