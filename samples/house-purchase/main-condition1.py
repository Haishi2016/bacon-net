import sys
sys.path.append('../../')
from sklearn.model_selection import train_test_split
import torch
from bacon.baconNet import baconNet
from bacon.visualization import visualize_tree_structure, print_tree_structure
import logging
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
from bacon.utils import SigmoidScaler

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

df = pd.read_csv('../../../realtor-data.csv')
features = ['bed', 'bath', 'acre_lot', 'price', 'zip_code', 'house_size']

df = df[features].dropna()

df = df[df['house_size'] < 10000]
df = df[df['bed'] < 10]
df = df[df['bath'] < 10]

def purchasing_condition(row):
    return (
        row['bed'] >= 4 
        and row['bath'] >= 3
        and row['house_size'] >= 3000
    )

df['Buy'] = df.apply(purchasing_condition, axis=1).astype(float)

X = df[features]
y = df['Buy']

# Train/test split
X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(X, y, test_size=0.2)

X_test_np_original = X_test_np.copy()


# Standardize the features
scaler = SigmoidScaler()
X_train_np = scaler.fit_transform(X_train_np)
X_test_np = scaler.transform(X_test_np)


# Convert to PyTorch tensors
X_train = torch.tensor(X_train_np, dtype=torch.float32).to(device)
Y_train = torch.tensor(y_train_np.values.reshape(-1, 1), dtype=torch.float32).to(device)
X_test = torch.tensor(X_test_np, dtype=torch.float32).to(device)
Y_test = torch.tensor(y_test_np.values.reshape(-1, 1), dtype=torch.float32).to(device)

# Initialize and train the BACON model
bacon = baconNet(input_size=X.shape[1], freeze_loss_threshold=0.15, weight_mode='fixed', aggregator='lsp.half_weight')
(best_model, best_accuracy) = bacon.find_best_model(X_train, Y_train, X_test, Y_test, 
    attempts=100, 
    acceptance_threshold=0.90,
    max_epochs=3000,
    save_path="./assembler-condition1.pth")
print(f"🏆 Best accuracy: {best_accuracy * 100:.2f}%")
# Visualize the BACON model's tree structure
print_tree_structure(bacon.assembler, features)
visualize_tree_structure(bacon.assembler, features)

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

# pos_beds = []
# pos_baths = []
# pos_house_sizes = []

# neg_beds = []
# neg_baths = []
# neg_house_sizes = []

# for i in range(X_test.shape[0]):
#     with torch.no_grad():
#         sample = X_test[i].unsqueeze(0)  # Add batch dimension → shape becomes (1, num_features)
#         output = bacon.inference(sample)
#         row = X_test_np_original.iloc[i]
#         if abs(output.item() - 1) < 0.01:  # positive prediction
#             pos_beds.append(row['bed'])
#             pos_baths.append(row['bath'])
#             pos_house_sizes.append(row['house_size'])
#         elif abs(output.item()) < 0.01:  # negative prediction
#             neg_beds.append(row['bed'])
#             neg_baths.append(row['bath'])
#             neg_house_sizes.append(row['house_size'])

# def estimate_boundary(pos, neg):
#     return (np.percentile(pos, 10) + np.percentile(neg, 90)) / 2

# print(f"\n⏳ Estimating boundaries. This may take a while...\n")

# estimated_bed = estimate_boundary(pos_beds, neg_beds)
# estimated_bath = estimate_boundary(pos_baths, neg_baths)
# estimated_house_size = estimate_boundary(pos_house_sizes, neg_house_sizes)

# print(f"📈 Estimated bed threshold: {estimated_bed:.2f}")
# print(f"📈 Estimated bath threshold: {estimated_bath:.2f}")
# print(f"📈 Estimated house size threshold: {estimated_house_size:.2f}")