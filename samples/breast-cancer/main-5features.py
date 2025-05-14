import sys
sys.path.append('../../')
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, LabelEncoder
import torch
from bacon.baconNet import baconNet
from bacon.visualization import (
    print_tree_structure, 
    visualize_tree_structure, 
    plot_sorted_predictions_with_labels, 
    print_metrics, 
    plot_precision_vs_threshold, 
    plot_sorted_predictions_with_errors,
    plot_feature_sensitivity,
    plot_all_feature_correlations,
    plot_feature_correlation,
    overlay_sorted_predictions_and_feature,
    plot_feature_sensitivity_synthetic,
    print_table_structure
)
from bacon.utils import (
    balance_classes, 
    find_best_threshold,
    analyze_bacon_tree_conjunctive_disjunctive,
    SigmoidScaler
)
import logging
import matplotlib.pyplot as plt
import pandas as pd

logging.basicConfig(level=logging.INFO, format='%(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
breast_cancer = fetch_ucirepo(id=17)
X = breast_cancer.data.features.iloc[:, 0:30]  # mean values only
y = LabelEncoder().fit_transform(breast_cancer.data.targets.values.ravel())


df = pd.DataFrame(X, columns=breast_cancer.data.features.columns[:30])
df['target'] = y

# Balance the dataset
# balanced_df = balance_classes(df, target_col='target', replication_factor=5)

# Separate back
X = df.drop(columns=['target'])
X = df[['area3', 'concave_points1', 'area2', 'perimeter3', 'radius3']]
y = df['target']

feature_names = X.columns.tolist()

# Train/test split
X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(X, y, test_size=0.2, random_state=None)


# Standardize
# scaler1 = RobustScaler()
# X_train_np = scaler1.fit_transform(X_train_np)
# X_test_np = scaler1.transform(X_test_np)

# Standardize
scaler2 = SigmoidScaler()
X_train_np = scaler2.fit_transform(X_train_np)
X_test_np = scaler2.transform(X_test_np)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train_np, dtype=torch.float32).to(device)
Y_train = torch.tensor(y_train_np.to_numpy().reshape(-1, 1), dtype=torch.float32).to(device)
Y_test = torch.tensor(y_test_np.to_numpy().reshape(-1, 1), dtype=torch.float32).to(device)
X_test = torch.tensor(X_test_np, dtype=torch.float32).to(device)
# 205 - 95.08%, 95.96%
# 200 - 95.43%
# 195 - 96.49%
# 140 - 94.55%
bacon = baconNet(input_size=5, freeze_loss_threshold=70, loss_amplifier=1000, is_frozen=True, weight_penalty_strength=1e-2)
(best_model, best_accuracy) = bacon.find_best_model(X_train, Y_train, X_test, Y_test, attempts=100, acceptance_threshold=0.90, max_epochs=3600, save_path='assembler-5features.pth')
print(f"🏆 Best accuracy: {best_accuracy * 100:.2f}%")
X_all = torch.cat([X_train, X_test], dim=0)
Y_all = torch.cat([Y_train, Y_test], dim=0)

# col_idx = df.columns.get_loc('concave_points3')
# X_all[:, col_idx] = 0.8

analysis = analyze_bacon_tree_conjunctive_disjunctive(bacon.assembler)
print(analysis)

# X_normalized = normalize_features(X_all, feature_names)
# X_tensor_normalized = torch.tensor(X_normalized.values, dtype=torch.float32).to(device)

plot_all_feature_correlations(X_all, feature_names)

print_tree_structure(bacon.assembler, X.columns.tolist())
print_table_structure(bacon.assembler, X.columns.tolist())
visualize_tree_structure(bacon.assembler, X.columns.tolist())

accuracies = []
accuracy_drops = []
feature_contributions = []

best_threshold, best_score = find_best_threshold(bacon, X_all, Y_all, metric='recall')
print(f"Best threshold for recall: {best_threshold:.2f}, Best score: {best_score:.4f}")
print_metrics(bacon, X_all, Y_all, threshold=best_threshold)
print_metrics(bacon, X_all, Y_all, threshold=0.02)
print_metrics(bacon, X_all, Y_all, threshold=0.1)


best_threshold, best_score = find_best_threshold(bacon, X_all, Y_all, metric='precision')
print(f"Best threshold for precision: {best_threshold:.2f}, Best score: {best_score:.4f}")
print_metrics(bacon, X_all, Y_all, threshold=best_threshold)

best_threshold, best_score = find_best_threshold(bacon, X_all, Y_all, metric='accuracy')
print(f"Best threshold for accuracy: {best_threshold:.2f}, Best score: {best_score:.4f}")
print_metrics(bacon, X_all, Y_all, threshold=best_threshold)


plot_sorted_predictions_with_labels(bacon, X_all, Y_all, threshold=best_threshold)
plot_sorted_predictions_with_errors(bacon, X_all, Y_all, threshold=best_threshold)

for feature in feature_names:
    plot_feature_sensitivity_synthetic(bacon, X_all, feature, feature_names)