import sys
sys.path.append('../../')
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import torch
from bacon.baconNet import baconNet
import logging
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.datasets import fetch_openml
from scipy.stats import pearsonr
import numpy as np

from sklearn.utils import resample

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

df = pd.read_csv('../../../realtor-data.csv')

# df = df.sample(n=1000, random_state=88)

features = ['bed', 'bath', 'acre_lot', 'price', 'zip_code', 'house_size']

# df = df[df["city"].str.lower() == "aguada"]
df = df[features].dropna()

df = df[df['house_size'] < 10000]
df = df[df['bed'] < 10]
df = df[df['bath'] < 10]


print(df.describe())

# df = df[['house_size', 'zip_code']].dropna()

# # Ensure zip_code is numeric
# df['zip_code'] = pd.to_numeric(df['zip_code'], errors='coerce')
# df = df.dropna()

# # Calculate Pearson correlation
# corr, _ = pearsonr(df['zip_code'], df['house_size'])
# print(f"📊 Pearson correlation between zip_code and house_size: {corr:.4f}")

# # Scatter plot with regression line
# plt.figure(figsize=(10, 5))
# sns.regplot(x='zip_code', y='house_size', data=df, scatter_kws={'alpha': 0.3}, line_kws={'color': 'red'})
# plt.title('Correlation Between Zip Code and House Size')
# plt.xlabel('Zip Code (as numeric)')
# plt.ylabel('House Size (sq ft)')
# plt.tight_layout()
# plt.show()

def purchasing_condition(row):
    return (
        row['bed'] >= 4 
        and row['bath'] >= 3
        and row['house_size'] >= 3000
    )

df['Buy'] = df.apply(purchasing_condition, axis=1).astype(float)

# # Separate majority and minority classes
# df_majority = df[df['Buy'] == 1]
# df_minority = df[df['Buy'] == 0]

# # Upsample minority class
# df_minority_upsampled = resample(
#     df_minority,
#     replace=True,                 # sample with replacement
#     n_samples=len(df_majority),  # match the majority class sizemor
#     random_state=42
# )

# # Combine and shuffle
# df_balanced = pd.concat([df_majority, df_minority_upsampled])
# df_balanced = df_balanced.sample(frac=1, random_state=42)


# X = df_balanced[features]
# y = df_balanced['Buy']
# print(y.value_counts())

X = df[features]
y = df['Buy']

# Train/test split
X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(X, y, test_size=0.2)

X_test_np_original = X_test_np.copy()


# Standardize the features
scaler = StandardScaler()
X_train_np = scaler.fit_transform(X_train_np)
X_test_np = scaler.transform(X_test_np)


# Convert to PyTorch tensors
X_train = torch.tensor(X_train_np, dtype=torch.float32)
Y_train = torch.tensor(y_train_np.values.reshape(-1, 1), dtype=torch.float32)
X_test = torch.tensor(X_test_np, dtype=torch.float32)
Y_test = torch.tensor(y_test_np.values.reshape(-1, 1), dtype=torch.float32)

# Initialize and train the BACON model
bacon = baconNet(input_size=X.shape[1], freeze_loss_threshold=0.06)
(best_model, best_accuracy) = bacon.find_best_model(X_train, Y_train, X_test, Y_test, attempts=100, acceptance_threshold=0.95)
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

pos_beds = []
pos_baths = []
pos_house_sizes = []

neg_beds = []
neg_baths = []
neg_house_sizes = []

for i in range(X_test.shape[0]):
    with torch.no_grad():
        sample = X_test[i].unsqueeze(0)  # Add batch dimension → shape becomes (1, num_features)
        output = bacon.inference(sample)
        row = X_test_np_original.iloc[i]
        if abs(output.item() - 1) < 0.01:  # positive prediction
            pos_beds.append(row['bed'])
            pos_baths.append(row['bath'])
            pos_house_sizes.append(row['house_size'])
        elif abs(output.item()) < 0.01:  # negative prediction
            neg_beds.append(row['bed'])
            neg_baths.append(row['bath'])
            neg_house_sizes.append(row['house_size'])

def estimate_boundary(pos, neg):
    return (np.percentile(pos, 10) + np.percentile(neg, 90)) / 2

estimated_bed = estimate_boundary(pos_beds, neg_beds)
estimated_bath = estimate_boundary(pos_baths, neg_baths)
estimated_house_size = estimate_boundary(pos_house_sizes, neg_house_sizes)

print(f"Estimated bed threshold: {estimated_bed:.2f}")
print(f"Estimated bath threshold: {estimated_bath:.2f}")
print(f"Estimated house size threshold: {estimated_house_size:.2f}")