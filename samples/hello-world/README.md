# Hello, World! with BACON

This sample generates a dataset for a random left-associative classic Boolean expression, and attempts to use BACON to approximate that expression.

## Run the code
```bash
python main.py
```
The following is the output of a sample run that discovers the ```((A and B) and C)``` expression from the training data.

```bash
➗ Expression: ((A and B) and C)
🏆 Best accuracy: 100.00%

🧠 Logical Aggregation Tree (Left-Associative):

  [B]─0.50────┐
  [A]─0.50──[ AND ]─0.50────┐
  [C]─0.50─────────────────[ AND ]──OUTPUT
```

## More experiments
To generate more complex expressions, undate this line:
```python
# update to the number of input variables you want to use (we've tested up to 1,000 inputs)
input_size = 3
```
And then re-run the program.

⚠️ If the accuracy is less than 100%, it means that BACON has discovered an expression that closely approximates the target logic, but does not exactly replicate it. ⚠️

## Additional notes
* To get less tracing information on how BACON works, change logging level to `logging.ERROR`.
* This sample trains the BACON model with a `loss_amplifier=1000` parameter, which scales the loss function by a factor of 1000. When multiple terms are combined using Boolean operators like `AND`, the resulting value can become very small—especially in complex expressions. This amplifier exaggerates the loss, helping the model continue learning effectively.

* The `randomize=False` parameter instructs the utility program to generate all possible permutations of input values. This becomes infeasible for complex expressions. Therefore, by default, this parameter is set to `True` to use randomly generated samples instead.

* The `bool.min_max` aggregator swings between the min and max functions to approximate classic Boolean `AND` and `OR` behavior. It uses a parameterized gate to control this behavior, and during training, the Straight-Through Estimator (STE) trick is applied to allow gradients to flow through the non-differentiable switching operation. When the gate value is near 0, the aggregator behaves like min (logical `AND`); when near 1, it behaves like max (logical `OR`). This enables smooth optimization while preserving interpretable logical behavior.