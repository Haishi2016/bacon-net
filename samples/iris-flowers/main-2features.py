import sys
sys.path.append('../../')
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import torch
from bacon.baconNet import baconNet
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(message)s')

# Fetch Iris dataset
iris = fetch_ucirepo(id=53)
X = iris.data.features
y_raw = iris.data.targets.values.ravel()

# Encode class labels
le = LabelEncoder()
y_encoded = le.fit_transform(y_raw)  # Setosa=0, Versicolor=1, Virginica=2
class_names = le.classes_

# Preprocess features
scaler = MinMaxScaler()
X_np = scaler.fit_transform(X)

pl_idx = X.columns.get_loc("petal length")
pw_idx = X.columns.get_loc("petal width")

# Find rows where both features are 1.0
epsilon = 1e-6


X_tensor = torch.tensor(X_np, dtype=torch.float32)

# Train one-vs-rest classifiers
target_class = 2  # Virginica

print(f"\n🔍 Training BACON to detect class '{class_names[target_class]}' vs others")

y_binary = (y_encoded == target_class).astype(np.float32)
y_tensor = torch.tensor(y_binary.reshape(-1, 1), dtype=torch.float32)

# Split into train/test
X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
    X_np, y_binary, test_size=0.2, random_state=42, stratify=y_binary)

# Convert to torch tensors
X_train = torch.tensor(X_train_np, dtype=torch.float32)
Y_train = torch.tensor(y_train_np.reshape(-1, 1), dtype=torch.float32)
X_test = torch.tensor(X_test_np, dtype=torch.float32)
Y_test = torch.tensor(y_test_np.reshape(-1, 1), dtype=torch.float32)


X_train_2f = X_train[:, [pl_idx, pw_idx]]
X_test_2f = X_test[:, [pl_idx, pw_idx]]


# Train BACON
bacon = baconNet(input_size=X_train_2f.shape[1], freeze_loss_threshold=0.09)
best_model, best_accuracy = bacon.find_best_model(
    X_train_2f, Y_train, X_test_2f, Y_test, attempts=100, acceptance_threshold=0.80
)
print(f"✅ Best accuracy for '{class_names[target_class]}' vs rest: {best_accuracy * 100:.2f}%")

# Explainability
feature_names = X.columns.tolist()
bacon.print_tree_structure(feature_names)
