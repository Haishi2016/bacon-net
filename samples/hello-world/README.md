# Hello, World! with BACON

This sample generates a dataset for a random classic Boolean expression, and attempts to use BACON to recover that expression.

## Run the code
```bash
python main.py
```
The following is the output of a sample run that discovers the ```((A or B) and C)``` expression from the training data.

```bash
🧠 Generating data...
➗ Expression: ((A or B) and C)
🔥 Attempting to find the best model... 1/10
   Epoch 0 - Loss: 0.7860
   Epoch 200 - Loss: 0.3991
   Epoch 400 - Loss: 0.3724
   Epoch 600 - Loss: 0.0027
   Epoch 800 - Loss: 0.0021
   Epoch 1000 - Loss: 0.0015
   Epoch 1200 - Loss: 0.0012
🧊 Low loss at epoch 1341, sampling top-k permutations...
   🔍 Perm (0, 1, 2) → Loss: 0.0010
✅ Best permutation selected: (0, 1, 2) (Loss: 0.0010)
✅ Freezing best permutation: (0, 1, 2) with loss 0.0010
   Epoch 0 - Loss: 0.0010
   Epoch 200 - Loss: 0.0005
   Epoch 400 - Loss: 0.0002
   Epoch 600 - Loss: 0.0001
   Epoch 800 - Loss: 0.0001
   Epoch 1000 - Loss: 0.0001
   Epoch 1200 - Loss: 0.0000
   Epoch 1400 - Loss: 0.0000
   Epoch 1600 - Loss: 0.0000
   Epoch 1800 - Loss: 0.0000
Indexes of best models: [0]
🏆 Best accuracy: 100.00%

🧠 Logical Aggregation Tree (Left-Associative):

  [A]─0.50────┐
  [B]─0.50──[ O R ]─0.50────┐
  [C]─0.50─────────────────[ AND ]──OUTPUT
```

## More experiments
To generate more complex expressions, undate this line:
```python
# update to the number of input variables you want to use
input_size = 3
```
And then re-run the program.

💡 For complex expressions, you may need to use a larger ```max_epochs``` to give the training process enough time to converge. You may also want to increase ```freeze_loss_threshold``` so BACON tries to lock on a permuation faster.
The following are some settings that worked in earlier tests:

| input_size | max_epoch | freeze_loss_threshold |
|--------|--------|--------|
| 3 | 2000 | 0.001 |
| 5 | 8000 | 0.001 |
| 7 | 10000 | 0.003 |