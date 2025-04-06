import sys
sys.path.append('../../')
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
from bacon.baconNet import baconNet
import logging
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.datasets import fetch_openml

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

df = pd.read_csv('../../../realtor-data.csv')

df = df.sample(n=10000, random_state=42)

features = ['price', 'bed', 'bath', 'house_size', 'zip_code', 'acre_lot']

# df = df[df["city"].str.lower() == "aguada"]
df = df[features].dropna()

def purchasing_condition(row):
    return (
        row['bed'] >= 4 and
        row['bath'] >= 2 and 
        row['house_size'] >= 2500
    )

df['Buy'] = df.apply(purchasing_condition, axis=1).astype(float)

X = df[features]
y = df['Buy']

# Train/test split
X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = MinMaxScaler()
X_train_np = scaler.fit_transform(X_train_np)
X_test_np = scaler.transform(X_test_np)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train_np, dtype=torch.float32)
Y_train = torch.tensor(y_train_np.values.reshape(-1, 1), dtype=torch.float32)
X_test = torch.tensor(X_test_np, dtype=torch.float32)
Y_test = torch.tensor(y_test_np.values.reshape(-1, 1), dtype=torch.float32)

# Initialize and train the BACON model
bacon = baconNet(input_size=X.shape[1], freeze_loss_threshold=0.130)
(best_model, best_accuracy) = bacon.find_best_model(X_train, Y_train, X_test, Y_test, attempts=100, acceptance_threshold=0.90)
print(f"Best accuracy: {best_accuracy * 100:.2f}%")

# Visualize the BACON model's tree structure
bacon.print_tree_structure(features)
bacon.visualize_tree_structure(features)

accuracies = []

for i in range(1, X_test.shape[1]):
    func_eval = bacon.prune_features(i)
    kept_indices = bacon.assembler.locked_perm[i:].tolist()
    X_test_pruned = X_test[:, kept_indices]
    with torch.no_grad():
        pruned_output = func_eval(X_test_pruned)
        pruned_accuracy = (pruned_output.round() == Y_test).float().mean().item()
        accuracies.append(pruned_accuracy)
        print(f"✅ Accuracy after pruning {i} feature(s): {pruned_accuracy * 100:.2f}%")

plt.figure(figsize=(10, 5))
plt.plot(range(1, len(accuracies) + 1), [a * 100 for a in accuracies], marker='o')
plt.title("Accuracy vs. Number of Features Pruned")
plt.xlabel("Number of Features Pruned from Left")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.tight_layout()
plt.show()