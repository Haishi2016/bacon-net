import sys
sys.path.append('../../')
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import torch
from bacon.baconNet import baconNet
import logging


logging.basicConfig(level=logging.INFO,
                    format='%(message)s')

breast_cancer = fetch_ucirepo(id=17)
X = breast_cancer.data.features.iloc[:, 0:10]  # mean values only
y = LabelEncoder().fit_transform(breast_cancer.data.targets.values.ravel())

# Train/test split
X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize
scaler = MinMaxScaler()
X_train_np = scaler.fit_transform(X_train_np)
X_test_np = scaler.transform(X_test_np)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train_np, dtype=torch.float32)
Y_train = torch.tensor(y_train_np.reshape(-1, 1), dtype=torch.float32)
X_test = torch.tensor(X_test_np, dtype=torch.float32)
Y_test = torch.tensor(y_test_np.reshape(-1, 1), dtype=torch.float32)
bacon = baconNet()
(best_model, best_accuracy) = bacon.find_best_model(X_test, Y_test, attempts=100, acceptance_threshold=0.95)
print(f"Best accuracy: {best_accuracy * 100:.2f}%")
bacon.print_tree_structure(X.columns.tolist())