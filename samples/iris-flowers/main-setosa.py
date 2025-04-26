import sys
sys.path.append('../../')
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, RobustScaler
import torch
from bacon.baconNet import baconNet
from bacon.visualization import visualize_tree_structure, print_tree_structure
import logging
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Fetch Iris dataset
iris = fetch_ucirepo(id=53)
X = iris.data.features
y_raw = iris.data.targets.values.ravel()

# Encode class labels
le = LabelEncoder()
y_encoded = le.fit_transform(y_raw)  # Setosa=0, Versicolor=1, Virginica=2
class_names = le.classes_

# Preprocess features
scaler = RobustScaler()
X_np = scaler.fit_transform(X)
X_tensor = torch.tensor(X_np, dtype=torch.float32)

# Train one-vs-rest classifiers
target_class = 0  # Setosa

print(f"\n🔍 Training BACON to detect class '{class_names[target_class]}' vs others")

y_binary = (y_encoded == target_class).astype(np.float32)
y_tensor = torch.tensor(y_binary.reshape(-1, 1), dtype=torch.float32)

# Split into train/test
X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
    X_np, y_binary, test_size=0.2, random_state=42, stratify=y_binary)

# Convert to torch tensors
X_train = torch.tensor(X_train_np, dtype=torch.float32).to(device)
Y_train = torch.tensor(y_train_np.reshape(-1, 1), dtype=torch.float32).to(device)
X_test = torch.tensor(X_test_np, dtype=torch.float32).to(device)
Y_test = torch.tensor(y_test_np.reshape(-1, 1), dtype=torch.float32).to(device)

# Train BACON
bacon = baconNet(input_size=X.shape[1], freeze_loss_threshold=0.21)
best_model, best_accuracy = bacon.find_best_model(
    X_train, Y_train, X_test, Y_test, attempts=100, acceptance_threshold=0.90, max_epochs=3000
)
print(f"✅ Best accuracy for '{class_names[target_class]}' vs rest: {best_accuracy * 100:.2f}%")

# Explainability
feature_names = X.columns.tolist()
visualize_tree_structure(bacon.assembler, feature_names)