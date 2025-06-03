# Hello, World! with BACON

This sample generates a dataset for a random classic Boolean expression, and attempts to use BACON to recover that expression.

## Run the code
```bash
python main.py
```
The following is the output of a sample run that discovers the ```((A and B) and C)``` expression from the training data.

```bash
🧠 Generating data...
⚡ Randomized input generation mode enabled.
➗ Expression: ((A and B) and C)
🔥 Attempting to find the best model... 1/10
   🏋️ Epoch 0 - Loss: 184.6348
🧊 Low loss at epoch 5, sampling top-k permutations...
   🔍 Perm (2, 1, 0) → Loss: 0.0000
✅ Best permutation selected: (2, 1, 0) (Loss: 0.0000)
✅ Freezing best permutation: (2, 1, 0) with loss 0.0000
🎯 Early stopping triggered by reaching low loss: 0.000001 at epoch 0
🧾 Indexes of best models: [0]
✅ Permutation is frozen: True
✅ Attempt 1 accuracy: 1.0000
🏆 Best accuracy: 100.00%

🧠 Logical Aggregation Tree (Left-Associative):

  [C]─0.50────┐
  [B]─0.50──[ AND ]─0.50────┐
  [A]─0.50─────────────────[ AND ]──OUTPUT
```

## More experiments
To generate more complex expressions, undate this line:
```python
# update to the number of input variables you want to use (recommended lower than 20)
input_size = 3
```
And then re-run the program.

⚠️ If the accuracy is less than 100%, it means that BACON has discovered an expression that closely approximates the target logic, but does not exactly replicate it. ⚠️