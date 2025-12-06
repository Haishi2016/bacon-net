# Hello World with Negation

This sample demonstrates BACON's ability to learn boolean expressions that include negated variables using the transformation layer.

## What's Different from Hello World?

The standard `hello-world` sample learns expressions like `(A and B) or C`. This sample extends that to learn expressions that may include negations, such as `(NOT A and B) or C` or `(A and NOT B) or NOT C`.

## How It Works

### Data Generation

The `generate_boolean_data_with_negation()` function:
- Creates random boolean expressions with 2+ variables
- Each variable has a configurable probability of being negated
- Generates training data based on the expression's truth table

### Transformation Layer

The BACON model is configured with `use_transformation_layer=True`, which enables it to:
- Learn whether each input feature should be used as-is (identity: `x`)
- Or negated (negation: `1-x`)

This allows the model to discover which variables need to be negated to match the target expression.

## Running the Sample

```bash
python main.py
```

## Sample Output

```
🧠 Generating boolean data with negation...
⚡ Randomized input generation mode enabled.
➗ Expression: ((NOT A and B) or C)
🔢 Variables: A, B, C
🔄 Negated variables: A

🎯 Training with transformation layer enabled...
   The model can learn to negate features if needed.

🔥 Attempting to find the best model... 1/10
   🏋️ Epoch 0 - Loss: 539.5317
🧊 Low loss at epoch 2, sampling top-k permutations...
✅ Best permutation selected: (0, 1, 2) (Loss: 0.0000)
✅ Freezing best permutation with loss 0.0000
🎯 Early stopping triggered by reaching low loss: 0.000001 at epoch 0

🏆 Best accuracy: 100.00%

🔄 Transformation Layer Analysis
======================================================================

📊 Learned Transformations:
   ✅ A: negation    (confidence:  98.5%) [Expected: negation]
   ✅ B: identity    (confidence:  99.2%) [Expected: identity]
   ✅ C: identity    (confidence:  97.8%) [Expected: identity]

📈 Transformation Accuracy: 3/3 variables correct
======================================================================

🌳 Logical Tree Structure:

  [A]─0.50────┐
  [B]─0.50──[ AND ]─0.50────┐
  [C]─0.50─────────────────[ O R ]──OUTPUT
```

## Key Features

1. **Automatic Negation Learning**: The transformation layer automatically discovers which variables should be negated
2. **Verification**: The output shows whether the learned transformations match the expected negations
3. **High Accuracy**: When successful, achieves 100% accuracy on the boolean expression
4. **Interpretability**: The tree structure and transformation analysis clearly show the learned logic

## Parameters

- `negation_prob`: Probability that each variable will be negated (default: 0.4)
- `input_size`: Number of boolean variables (default: 3)
- `transformation_temperature`: Temperature for softmax selection (default: 1.0)

## Notes

- The transformation layer learns to apply negations independently of the permutation layer
- This demonstrates that BACON can learn complex logical relationships beyond simple conjunctions/disjunctions
- The transformation accuracy metric shows how well the model discovered the correct negations
