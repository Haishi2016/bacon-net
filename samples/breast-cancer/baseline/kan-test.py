# ✅ Save as breast_cancer_kan_demo.py

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from ucimlrepo import fetch_ucirepo

# Import from the kan package (community fork)
from kan import *

# 1. Load and prepare breast cancer dataset
breast_cancer = fetch_ucirepo(id=17)
X = breast_cancer.data.features
y = breast_cancer.data.targets

# Ensure it's a DataFrame
X = pd.DataFrame(X, columns=breast_cancer.data.feature_names)

# Encode labels
le = LabelEncoder()
y = le.fit_transform(y.values.ravel())

# Normalize features to [0, 1]
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# 2. Define and train KANModel (community fork API)
model = KAN(input_dim=X.shape[1], output_dim=1, hidden_dim=16, num_layers=2, grid_size=20)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Training loop
for epoch in range(1000):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch} - Loss: {loss.item():.4f}")

# Evaluate
model.eval()
with torch.no_grad():
    preds = torch.sigmoid(model(X_test_tensor)).squeeze().numpy()
    preds_bin = (preds > 0.5).astype(int)
    print("\nAccuracy:", accuracy_score(y_test, preds_bin))
    print(classification_report(y_test, preds_bin))

# 3. Plot the learned basis functions for each feature (using internal API)
plt.figure(figsize=(20, 15))
for i in range(X.shape[1]):
    plt.subplot(5, 7, i+1)
    grid_x = torch.linspace(0, 1, 1000).unsqueeze(-1)
    basis_output = model.kan_layers[0].basis_layers[i](grid_x).detach().cpu().numpy()
    plt.plot(grid_x.cpu().numpy(), basis_output)
    plt.title(X.columns[i])
plt.suptitle("KAN Learned Basis Functions for Each Feature (First Layer)", fontsize=16)
plt.tight_layout()
plt.show()

# 4. Approximate feature importance (variance of basis output)
with torch.no_grad():
    basis_variances = []
    for i in range(X.shape[1]):
        grid_x = torch.linspace(0, 1, 1000).unsqueeze(-1)
        basis_output = model.kan_layers[0].basis_layers[i](grid_x).detach().cpu().numpy()
        basis_variances.append(np.var(basis_output))

feat_imp_df = pd.DataFrame({
    'Feature': X.columns,
    'Variance-Based Importance': basis_variances
}).sort_values('Variance-Based Importance', ascending=False)

plt.figure(figsize=(10, 6))
plt.barh(feat_imp_df['Feature'], feat_imp_df['Variance-Based Importance'])
plt.gca().invert_yaxis()
plt.xlabel("Variance of Basis Output")
plt.title("Approximate Feature Importance (KAN Basis Variance)")
plt.grid(True, axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

print("\nApproximate Top Features (by basis variance):")
print(feat_imp_df.head(10))
