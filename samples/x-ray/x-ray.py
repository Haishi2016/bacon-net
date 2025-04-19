import pandas as pd
from sklearn.preprocessing import RobustScaler
import sys
sys.path.append('../../')
import torch
from bacon.baconNet import baconNet
import logging
import matplotlib.pyplot as plt
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format='%(message)s')

feature_names = [
    'bbox_ratio',
    'brightness_diff',
    'vertical_pos',
    'horizontal_pos',
    'texture_std',
    'edge_density',
    'aspect_ratio',
    'gradient_magnitude',
    'laplacian_variance',
    'intensity_contrast',
    'center_offset',
    'circularity',
    'glcm_contrast',
    'lung_zone',
    'asymmetry'
]

# Binary labels
def is_mass(label):
    # return 1.0 if label in dos_attacks else 0.0
    # return 1.0 if label == 'neptune' else 0.0
    return 0.0 if label == "Mass" else 1.0

# Load training and test data
df = pd.read_csv('xray_features_15.csv')

df["target"] = df["label"].apply(is_mass)
X = df.drop(columns=['image', 'label', 'target'])
y = df["target"]


X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(X, y, test_size=0.2, random_state=42)

df_train = X_train_np.copy()
df_train["target"] = y_train_np

df_majority = df_train[df_train['target'] == 1]
df_minority = df_train[df_train['target'] == 0]

# Upsample minority class
df_minority_upsampled = resample(
    df_minority,
    replace=True,                 # sample with replacement
    n_samples=len(df_majority),  # match the majority class sizemor
    random_state=42
)

# Combine and shuffle
df_balanced = pd.concat([df_majority, df_minority_upsampled])
df_balanced = df_balanced.sample(frac=1, random_state=42)

print(df_balanced['target'].value_counts())

scaler = RobustScaler()
X_train = scaler.fit_transform(df_balanced[feature_names])
X_test = scaler.transform(X_test_np[feature_names])
y_train = df_balanced["target"].values
y_test = y_test_np.values

# === Convert to tensors ===
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_test_tensor = torch.tensor(y_test.reshape(-1, 1), dtype=torch.float32)

bacon = baconNet(input_size=X_train.shape[1], freeze_loss_threshold=0.03, lock_loss_tolerance=0.04)
best_model, best_accuracy = bacon.find_best_model(
    X_train_tensor, Y_train_tensor, X_test_tensor, Y_test_tensor, 
    attempts=100, acceptance_threshold=0.90
)
print(f"✅ Best accuracy: {best_accuracy * 100:.2f}%")

# Visualize the BACON model's tree structure
bacon.print_tree_structure(feature_names)
bacon.visualize_tree_structure(feature_names)

accuracies = []

for i in range(1, X_test_tensor.shape[1]):
    func_eval = bacon.prune_features(i)
    kept_indices = bacon.assembler.locked_perm[i:].tolist()
    X_test_pruned = X_test_tensor[:, kept_indices]
    with torch.no_grad():
        pruned_output = func_eval(X_test_pruned)
        pruned_accuracy = (pruned_output.round() == Y_test_tensor).float().mean().item()
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