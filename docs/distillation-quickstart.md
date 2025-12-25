# BACON Distillation - Quick Start Guide

## Installation

BACON distillation is included in the bacon-net framework. No additional installation needed.

## Basic Usage

### Step 1: Train and Export Your Model

```python
from bacon.baconNet import baconNet
from bacon.utils import export_tree_structure_to_json

# Train your model
model = baconNet(...)
# ... training code ...

# Export to JSON
export_tree_structure_to_json(
    model.assembler,
    'my_model.json',
    feature_names=['feature1', 'feature2', ...],
    aggregator_type='lsp.half_weight'
)
```

### Step 2: Distill to Standalone Python

**Option A: Instance Mode (Zero Dependencies)**
```bash
python bacon-distill.py my_model.json my_model.py --mode instance
```

**Option B: Batch Mode (NumPy Required)**
```bash
python bacon-distill.py my_model.json my_model.py --mode batch
```

### Step 3: Use the Distilled Model

**Instance mode - single sample:**
```python
from my_model import predict

sample = [0.5, 0.7, 0.3, ...]
result = predict(sample)
print(f"Prediction: {result}")
```

**Batch mode - multiple samples:**
```python
import numpy as np
from my_model import predict

samples = np.array([
    [0.5, 0.7, 0.3, ...],
    [0.6, 0.8, 0.4, ...],
    [0.4, 0.6, 0.2, ...]
])
results = predict(samples)
print(f"Predictions: {results}")
```

**Batch mode - single sample:**
```python
import numpy as np
from my_model import predict

sample = np.array([0.5, 0.7, 0.3, ...])
result = predict(sample)  # Returns scalar
print(f"Prediction: {result}")
```

## CLI Options

```bash
python bacon-distill.py <json_file> <output_file> [options]

Required Arguments:
  json_file       Path to the JSON model structure file
  output_file     Where to save the generated Python code

Optional Arguments:
  --aggregator, -a    Type of aggregator (default: lsp.half_weight)
                      Choices: lsp.half_weight, lsp.full_weight, 
                               math.arithmetic, math.geometric
  
  --mode, -m          Generation mode (default: instance)
                      Choices: instance, batch
  
  --verbose, -v       Enable verbose output
```

## Mode Selection Guide

### Choose Instance Mode When:
- 🎯 **Deploying to edge devices** (IoT, Raspberry Pi, etc.)
- 🎯 **Serverless functions** (AWS Lambda, Azure Functions)
- 🎯 **Zero dependencies required** (no NumPy, no PyTorch)
- 🎯 **Processing one sample at a time**
- 🎯 **Maximum portability needed**

### Choose Batch Mode When:
- 🎯 **Processing large datasets** (1000+ samples)
- 🎯 **NumPy already in your environment**
- 🎯 **Running on servers** with good resources
- 🎯 **Data science pipelines** (Jupyter, pandas workflows)
- 🎯 **Need vectorized operations**

## Examples

### Example 1: Heart Disease Prediction (Instance Mode)

```bash
# Generate instance mode model
python bacon-distill.py heart_disease.json heart_predict.py --mode instance

# Use it
python -c "from heart_predict import predict; print(predict([63, 1, 1, 145, ...]))"
```

### Example 2: Heart Disease Prediction (Batch Mode)

```bash
# Generate batch mode model
python bacon-distill.py heart_disease.json heart_predict_batch.py --mode batch

# Use it
python -c "import numpy as np; from heart_predict_batch import predict; \
           X = np.random.random((100, 22)); print(predict(X))"
```

### Example 3: Deploy to AWS Lambda (Instance Mode)

```python
# lambda_function.py
from my_model import predict  # Zero dependencies!

def lambda_handler(event, context):
    features = event['features']
    prediction = predict(features)
    
    return {
        'statusCode': 200,
        'body': {'prediction': prediction}
    }
```

## Performance Expectations

Based on heart disease benchmark (297 samples):

| Mode | Throughput | Dependencies | Use Case |
|------|-----------|--------------|----------|
| **Instance** | ~2,600 samples/sec | None | Edge, serverless |
| **Batch** | ~2,400 samples/sec | NumPy | Data pipelines |
| **Original** | ~2,600 samples/sec | PyTorch, bacon-net | Training, research |

**Note**: Batch mode performance improves with larger batch sizes (1000+ samples).

## Troubleshooting

### "ImportError: No module named 'numpy'"
- You're using batch mode but NumPy isn't installed
- Solution: `pip install numpy` or use `--mode instance`

### "ValueError: Expected N features, got M"
- Your input has wrong number of features
- Check the docstring in generated file for feature order
- Ensure features are in ORIGINAL dataset order (pre-permutation)

### "Predictions differ from original model"
- Small differences (~0.001) are normal due to floating-point precision
- Both modes should match each other exactly
- If difference is large (>0.01), check:
  - Correct aggregator type specified
  - JSON export includes all metadata
  - Feature order matches training data

## Advanced Usage

### Custom Aggregators

```bash
# For models using arithmetic mean aggregator
python bacon-distill.py model.json inference.py --aggregator math.arithmetic --mode instance
```

### Verbose Output

```bash
# See detailed distillation process
python bacon-distill.py model.json inference.py --verbose
```

### Python API

```python
from bacon.utils import distill_bacon_to_code

# Full control over distillation
distill_bacon_to_code(
    json_file='model.json',
    output_file='inference.py',
    aggregator_type='lsp.half_weight',
    mode='instance'
)
```

## Next Steps

- Read [distillation-modes.md](distillation-modes.md) for detailed comparison
- See [samples/heart-disease/](../samples/heart-disease/) for complete example
- Run `compare_models_v2.py` to validate your distilled models

## Support

For issues or questions:
1. Check the [documentation](../docs/)
2. Review example notebooks in [samples/](../samples/)
3. Open an issue on GitHub
