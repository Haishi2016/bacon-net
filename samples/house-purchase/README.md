# House purchasing decisions
This sample illustrates two scenarios for deciding whether to purchase a house.

* **Condition 1:** The house must have at least 4 bedrooms, at least 3 bathrooms, and a total size of at least 3,000 square feet.

* **Condition 2:** The house must have at least 4 bedrooms, at least 3 bathrooms, and a lot size of at least 0.5 acres.

⚠️ Before running the programs, you need to download the [USA Real Estate Dataset](https://www.kaggle.com/datasets/ahmedshahriarsakib/usa-real-estate-dataset) and save the `realtor-data.csv` to  the `../../../realtor-data.csv` folder (or update the source code to use a folder of yoru choice).

## Condition 1
```bash
python main-condition1.py
```
The following is a sample output from running the program using the saved `assembler-condition1.pth model`. The output also shows how accuracy drops as features are progressively pruned from the tree. Since pruning the bottom four features—`bath`, `price`, `zip_code`, and `acre_lot`—has no effect on accuracy, these features are considered irrelevant.

Notably, `bath` is considered irrelevant in this case, even though the original condition includes `bath >= 3` as a criterion. This is because bath is strongly correlated with bed, and in this instance, the model determines that house_size and bed alone are sufficient for making the decision.

```bash
📂 Found saved model at ./assembler-condition1.pth, loading...
✅ Loaded model accuracy: 0.9285
🏆 Best accuracy: 92.85%

🧠 Logical Aggregation Tree (Left-Associative):

        [bath]─0.50────┐
       [price]─0.50──[a=1.95576501]─0.50────┐
    [zip_code]─0.50─────────────────[a=0.81853366]─0.50────┐
    [acre_lot]─0.50────────────────────────────────[a=0.86211991]─0.50────┐
         [bed]─0.50───────────────────────────────────────────────[a=-0.80629879]─0.50────┐
  [house_size]─0.50──────────────────────────────────────────────────────────────[a=1.87630153]──OUTPUT
✅ Accuracy after pruning 1 feature(s): 92.72%
✅ Accuracy after pruning 2 feature(s): 92.72%
✅ Accuracy after pruning 3 feature(s): 92.71%
✅ Accuracy after pruning 4 feature(s): 92.71%
✅ Accuracy after pruning 5 feature(s): 87.84%
```