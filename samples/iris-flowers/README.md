# Iris flower classification

This sample includes three programs, each training a classifier to distinguish one Iris species—Setosa, Versicolor, or Virginica—from the others.

## Virginica classification
```bash
python main-virginica.py
```
The following is sample output when the program detects a previously saved model file `assembler-virginica.pth`. To retrain the model from scratch, remove or rename the file and re-run the program.
```bash
🔍 Training BACON to detect class 'Iris-virginica' vs others
📂 Found saved model at assembler-virginica.pth, loading...
✅ Loaded model accuracy: 0.9667
✅ Best accuracy for 'Iris-virginica' vs rest: 96.67%
```
## Setosa classification
```bash
python main-setosa.py
```
As with Virginica, the program loads from `assembler-setosa.pth` if it exists. To retrain, remove or rename the file and rerun the program.

Because BACON operates on degrees of truth, all features are reversed before training. Setosa flowers tend to have smaller petals and sepals, so reversing the feature values encodes the degree to which they are small—preserving BACON’s semantic consistency.

```bash
🔍 Training BACON to detect class 'Iris-setosa' vs others
📂 Found saved model at assembler-setosa.pth, loading...
✅ Loaded model accuracy: 1.0000
✅ Best accuracy for 'Iris-setosa' vs rest: 100.00%
```
## Versicolor classification
```bash
python assembler-versicolor.py
```
⚠️ This program is not expected to converge. There is no clear intrinsic logic to distinguish Versicolor from the other two classes directly. A more feasible strategy is to detect Setosa and Virginica first and label remaining samples as Versicolor.

```bash

🔍 Training BACON to detect class 'Iris-versicolor' vs others

⚠️ THIS WILL NOT CONVERGE AS THERE'S NO CLEAR LOGIC TO DISTINGUISH Versicolor FROM OHTERS

🔥 Attempting to find the best model... 1/5
   Epoch 0 - Loss: 0.7838
   Epoch 200 - Loss: 0.5969
   Epoch 400 - Loss: 0.5968
   Epoch 600 - Loss: 0.5971
   Epoch 800 - Loss: 0.5962
   Epoch 1000 - Loss: 0.5957
   Epoch 1200 - Loss: 0.5954
   Epoch 1400 - Loss: 0.5952
   Epoch 1600 - Loss: 0.5962
   Epoch 1800 - Loss: 0.5954
   Epoch 2000 - Loss: 0.5955
   Epoch 2200 - Loss: 0.5950
   Epoch 2400 - Loss: 0.5951
   Epoch 2600 - Loss: 0.5954
   Epoch 2800 - Loss: 0.5953
🧾 Indexes of best models: []
    ...
```