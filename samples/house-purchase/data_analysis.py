import sys
sys.path.append('../../')
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
import torch
from bacon.baconNet import baconNet
import logging
import matplotlib.pyplot as plt
import pandas as pd

import numpy as np
from sklearn.datasets import fetch_openml

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')

df = pd.read_csv('../../../realtor-data.csv')

df = df.sample(n=10000, random_state=None)

features = ['bed', 'bath', 'house_size']

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

X_test_np_original = X_test_np.copy()

# Standardize the features
scaler = RobustScaler()
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