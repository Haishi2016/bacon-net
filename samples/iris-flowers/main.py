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

# Check if any sample has normalized sepal width and sepal length both equal to 1
sepal_width_col = X.columns.get_loc("sepal width")
sepal_length_col = X.columns.get_loc("sepal length")

# Find rows where both features are 1.0
epsilon = 1e-6
sepal_width_col = X.columns.get_loc("sepal width")
sepal_length_col = X.columns.get_loc("sepal length")

# Find rows where both normalized sepal width and sepal length are close to 1.0
matches = np.where(
    (np.abs(X_np[:, sepal_width_col] - 1.0) < epsilon) &
    (np.abs(X_np[:, sepal_length_col] - 1.0) < epsilon)
)[0]

if len(matches) > 0:
    print(f"⚠️ Found {len(matches)} sample(s) with both sepal width and sepal length ≈ 1.0")
    print("Sample indices:", matches)
    print("Corresponding class labels:", [class_names[y_encoded[i]] for i in matches])
    print("Feature values:\n", X.iloc[matches])
else:
    print("✅ No sample has both sepal width and sepal length ≈ 1.0")



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


# X_test_petals_only = X_test.clone()
# X_test_petals_only[:, sepal_width_col] = 0.0  # sepal length
# X_test_petals_only[:, sepal_length_col] = 0.0  # sepal width

# Train BACON
bacon = baconNet(input_size=X.shape[1], freeze_loss_threshold=0.09)
best_model, best_accuracy = bacon.find_best_model(
    X_train, Y_train, X_test, Y_test, attempts=100, acceptance_threshold=0.80
)
print(f"✅ Best accuracy for '{class_names[target_class]}' vs rest: {best_accuracy * 100:.2f}%")

# Explainability
feature_names = X.columns.tolist()
bacon.print_tree_structure(feature_names)
# # bacon.visualize_tree_structure(feature_names)
# eval_func = bacon.prune_features(2)

# pl_idx = X.columns.get_loc("petal length")
# pw_idx = X.columns.get_loc("petal width")

# # Extract petal features from test set
# X_test_petal = X_test[:, [pl_idx, pw_idx]]

# # Evaluate pruned model
# with torch.no_grad():
#     pruned_output = eval_func(X_test_petal)
#     pruned_accuracy = (pruned_output.round() == Y_test).float().mean().item()
#     print(f"✅ Accuracy after pruning: {pruned_accuracy * 100:.2f}%")