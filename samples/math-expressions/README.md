# Math Expressions Sample

This sample demonstrates using BACON with the `OperatorSetAggregator` to approximate mathematical expressions in regression mode.

## Overview

The sample allows you to:
1. Define a mathematical expression using single-letter variables (a-z)
2. Generate a synthetic dataset with configurable size and noise levels
3. Train a BACON network to learn the expression using arithmetic operators

## Configuration

Edit the configuration section at the top of `main.py`:

```python
# =============================================================================
# CONFIGURATION - Edit these values to customize the experiment
# =============================================================================

# Mathematical expression to approximate
EXPRESSION = "3*a + 2*b - c"

# Dataset configuration
NUM_SAMPLES = 200           # Number of training samples
INPUT_NOISE_PERCENT = 10.0  # Noise level for inputs (0-100)
OUTPUT_NOISE_PERCENT = 10.0 # Noise level for outputs (0-100)

# Training configuration
EPOCHS = 2000               # Number of training epochs
LEARNING_RATE = 0.01        # Learning rate
TREE_LAYOUT = "left"        # Tree structure: "left", "balanced", "full"

# Random seed for reproducibility
SEED = 42
```

## Usage

```bash
# Simply run after editing configuration
python main.py
```

## Expression Syntax

Use explicit multiplication operator `*` for coefficients:

| Expression | Syntax |
|------------|--------|
| 3a + 2b - c | `"3*a + 2*b - c"` |
| a×b + c | `"a*b + c"` |
| (a+b)×c | `"(a+b)*c"` |
| √a + b | `"sqrt(a) + b"` |
| a² | `"pow(a, 2)"` or `"a*a"` |

### Supported Operators

| Type | Examples |
|------|----------|
| Arithmetic | `+`, `-`, `*`, `/` |
| Parentheses | `(a+b)*c` |
| Functions | `sqrt(a)`, `sin(a)`, `cos(a)`, `tan(a)`, `exp(a)`, `log(a)`, `abs(a)`, `pow(a,2)` |
| Constants | `pi`, `e` |

## Supported Operators

The `OperatorSetAggregator` in arithmetic mode (`math.operator_set.arith`) supports:

| Operator | Symbol | Description |
|----------|--------|-------------|
| Add | `+` | Addition: `(a + b) / 2` (normalized to [0,1]) |
| Sub | `-` | Subtraction: `(a - b + 1) / 2` (normalized to [0,1]) |
| Mul | `*` | Multiplication: `a * b` |
| Div | `/` | Division: `tanh(a / (b + ε))` (safe division, normalized) |

## How It Works

1. **Expression Parsing**: The expression is parsed to extract variable names and create an evaluation function.

2. **Data Generation**: 
   - Random input values are generated in the range [0.1, 0.9]
   - The expression is evaluated for each sample
   - Outputs are normalized to [0, 1]
   - Gaussian noise is added to both inputs and outputs

3. **Model Training**:
   - BACON network is created with `OperatorSetAggregator`
   - The aggregator learns which operator to use at each tree node
   - MSE loss is used for regression
   - The model learns both the permutation (variable ordering) and operators

4. **Result Analysis**:
   - Learned operators are displayed for each tree node
   - Tree structure shows the discovered expression
   - R² score indicates regression quality

## Notes

- Input/output values are normalized to [0, 1] range for numerical stability
- Complex expressions may require more samples and epochs
- The noise levels simulate real-world measurement uncertainty
- Variable names must be single lowercase letters (a-z)

## Expected Output

```
🖥️  Using device: cuda
📐 Expression: a+b+c
📊 Variables: ['a', 'b', 'c']
📊 Generating 100 samples...
   Input shape: torch.Size([100, 3])
   Output range (original): [0.3456, 2.1234]
   Output range (normalized): [0.0123, 0.9876]
   Input noise: 10.0%
   Output noise: 10.0%

🧠 Creating BACON model with 3 inputs...

🏋️ Training for 3000 epochs...
   Epoch     1: MSE = 0.123456, R² = 0.1234
   ...
   Epoch  3000: MSE = 0.001234, R² = 0.9876

📊 Final Results:
   MSE: 0.001234
   R²:  0.9876

🔧 Learned Operator Selections:
   Node 1: add (conf=0.95) [add=0.95, sub=0.02, mul=0.02, div=0.01]
   Node 2: add (conf=0.93) [add=0.93, sub=0.03, mul=0.03, div=0.01]

📋 Learned Tree Structure:
...
```
