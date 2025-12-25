# The following two lines are needed to refer to local package instead of published package

import sys
sys.path.insert(0, '../../')

import pandas as pd
from sklearn.preprocessing import RobustScaler
import torch
from bacon.baconNet import baconNet
from bacon.visualization import print_tree_structure, visualize_tree_structure
import logging
import matplotlib.pyplot as plt
from sklearn.utils import resample
from bacon.transformationLayer import IdentityTransformation, NegationTransformation
from sklearn.preprocessing import QuantileTransformer

logging.basicConfig(level=logging.INFO, format='%(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

feature_names = [
    'duration', 
    'protocol_type', # tcp, upd, icmp
    'service', # aol, auth, bgp, courier, etc.
    'flag', # 'OTH', 'REJ', 'RSTO', 'RSTOS0', 'RSTR', 'S0', 'S1', 'S2', 'S3', 'SF', 'SH'
    'src_bytes', 
    'dst_bytes', 
    'land', # 0 or 1 
    'wrong_fragment', 
    'urgent',
    'hot', 
    'num_failed_logins', 
    'logged_in', # 0 or 1
    'num_compromised',
    'root_shell', 
    'su_attempted', 
    'num_root', 
    'num_file_creations',
    'num_shells', 
    'num_access_files', 
    'num_outbound_cmds',
    'is_host_login', # 0 or 1
    'is_guest_login', # 0 or 1 
    'count', 
    'srv_count',
    'serror_rate', 
    'srv_serror_rate', 
    'rerror_rate', 
    'srv_rerror_rate',
    'same_srv_rate', 
    'diff_srv_rate', 
    'srv_diff_host_rate',
    'dst_host_count', 
    'dst_host_srv_count', 
    'dst_host_same_srv_rate',
    'dst_host_diff_srv_rate', 
    'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 
    'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 
    'dst_host_rerror_rate',
    'dst_host_srv_rerror_rate', 
    'label',         # e.g. "neptune"
    'difficulty'                                 # e.g. 21
]


# Load training and test data
train_df = pd.read_csv('../../../KDDTrain+.txt', header=None)
test_df = pd.read_csv('../../../KDDTest+.txt', header=None)

train_df.columns = feature_names
test_df.columns = feature_names

print(train_df['dst_bytes'].describe())
filtered = train_df[train_df['dst_bytes']!=0].copy()
print(filtered.describe())


# # Define attack types
# dos_attacks = [
#     'back', 'land', 'neptune', 'pod', 'smurf', 'teardrop',
#     'apache2', 'udpstorm', 'processtable', 'worm'
# ]

# Define attack types
dos_attacks = [
    'neptune'
]

# Binary labels
def is_dos(label):
    return 1.0 if label in dos_attacks else 0.0
    # return 1.0 if label == 'neptune' else 0.0    
    
train_df['target'] = train_df['label'].apply(is_dos)
test_df['target'] = test_df['label'].apply(is_dos)

drop_cols = ['label', 'target', 'protocol_type', 'service', 'flag', 'difficulty']

# Balance the training data BEFORE creating X_train/y_train
df_majority = train_df[train_df['target'] == 1]
df_minority = train_df[train_df['target'] == 0]

# Upsample minority class
df_minority_upsampled = resample(
    df_minority,
    replace=True,                 # sample with replacement
    n_samples=len(df_majority),  # match the majority class size
    random_state=42
)

# Combine and shuffle
df_balanced = pd.concat([df_majority, df_minority_upsampled])
df_balanced = df_balanced.sample(frac=1, random_state=42)

print(train_df['target'].value_counts())

# Use unbalanced training data
X_train = train_df.drop(columns=drop_cols)
y_train = train_df['target']

X_test = test_df.drop(columns=drop_cols)
y_test = test_df['target']

scaler = QuantileTransformer(
    output_distribution="uniform",  # -> [0,1]
    n_quantiles=1000,               # or smaller for big data
    subsample=int(1e5),             # default
    random_state=42
)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_test_tensor = torch.tensor(y_test.values.reshape(-1, 1), dtype=torch.float32)

# Use actual input size for transformations
num_features = X_train.shape[1]

# After loading the data, before training
print("\n📊 Feature correlation with DoS attacks:")
for col in ['src_bytes', 'dst_bytes', 'logged_in']:
    if col in df_balanced.columns:
        dos_mean = df_balanced[df_balanced['target'] == 1.0][col].mean()
        normal_mean = df_balanced[df_balanced['target'] == 0.0][col].mean()
        dos_median = df_balanced[df_balanced['target'] == 1.0][col].median()
        normal_median = df_balanced[df_balanced['target'] == 0.0][col].median()
        print(f"{col}:")
        print(f"  DoS mean: {dos_mean:.4f}, median: {dos_median:.4f}")
        print(f"  Normal mean: {normal_mean:.4f}, median: {normal_median:.4f}")
        print(f"  Correct indicator: {'identity' if dos_mean > normal_mean else 'negation'}")

# Special analysis for dst_bytes
print("\n📊 dst_bytes < 0.5 analysis:")
low_dst = df_balanced[df_balanced['dst_bytes'] < 0.5]
high_dst = df_balanced[df_balanced['dst_bytes'] >= 0.5]
print(f"When dst_bytes < 0.5:")
print(f"  Total samples: {len(low_dst)}")
print(f"  DoS attacks: {low_dst['target'].sum():.0f} ({low_dst['target'].mean()*100:.2f}%)")
print(f"  Normal: {(len(low_dst) - low_dst['target'].sum()):.0f} ({(1-low_dst['target'].mean())*100:.2f}%)")
print(f"\nWhen dst_bytes >= 0.5:")
print(f"  Total samples: {len(high_dst)}")
print(f"  DoS attacks: {high_dst['target'].sum():.0f} ({high_dst['target'].mean()*100:.2f}%)")
print(f"  Normal: {(len(high_dst) - high_dst['target'].sum()):.0f} ({(1-high_dst['target'].mean())*100:.2f}%)")

bacon = baconNet(
    input_size=num_features, 
    freeze_loss_threshold=0.13, 
    lock_loss_tolerance=0.02,
    use_transformation_layer=True,
    transformations=[
        IdentityTransformation(num_features),
        NegationTransformation(num_features)
    ],
    use_class_weighting=False,
    weight_mode='fixed',
    aggregator='lsp.half_weight',    
    permutation_final_temperature=4.0,
    transformation_final_temperature=0.05,
    permutation_initial_temperature=5.0,    
    transformation_initial_temperature=1.0)

X_test_tensor = X_test_tensor.to(device)
Y_test_tensor = Y_test_tensor.to(device)    
X_train_tensor = X_train_tensor.to(device)
Y_train_tensor = Y_train_tensor.to(device)  

# 89.23%
#   hierarchical_group_size = 17
best_model, best_accuracy = bacon.find_best_model(
    X_train_tensor, Y_train_tensor, X_test_tensor, Y_test_tensor, 
    use_hierarchical_permutation=True,
    hierarchical_group_size=10,
    hierarchical_epochs_per_attempt=3000,  # Max epochs (safety limit)
    annealing_epochs=2000,  # Temperature annealing period
    frozen_training_epochs=2000,  # Train for 200 epochs after freezing
    convergence_patience=500,  # Stop if no improvement for 500 epochs
    convergence_delta=0.001,  # Minimum improvement threshold
    freeze_confidence_threshold=0.80,  # Lower threshold to accept Sinkhorn constraint (was 0.90)
    loss_weight_perm_sparsity=5.0,  # Aggressive sparsity to push toward one-hot (increased from 0.1)
    hierarchical_bleed_ratio=0.5,        
    max_epochs=12000,
    attempts=1, 
    acceptance_threshold=1.0
)
print(f"✅ Best accuracy: {best_accuracy * 100:.2f}%")

filtered_features = [f for f in feature_names if f not in drop_cols]

# Visualize the BACON model's tree structure
print_tree_structure(bacon.assembler, filtered_features)
visualize_tree_structure(bacon.assembler, filtered_features)

accuracies = []

# Baseline accuracy
with torch.no_grad():
    baseline_output = bacon(X_test_tensor)
    baseline_accuracy = (baseline_output.round() == Y_test_tensor).float().mean().item()
    print(f"✅ Baseline accuracy: {baseline_accuracy * 100:.2f}%")

# Feature importance by neutralizing features from left (least important)
print("\n🔍 Testing feature importance by neutralizing features:")
print("   (Setting removed features equal to their partner and andness=0.5 for passthrough)")

def evaluate_with_neutralized_features(model, X, y, num_neutralized):
    """Evaluate accuracy with the first num_neutralized features neutralized.
    
    Neutralization: Set weights so neutralized inputs have weight=0, keeping inputs have weight=1.
    This makes the aggregator pass through the kept input: LSP(left, right, a, 0, 1) ≈ right
    
    In left-associative tree:
    - Aggregator 0: combines input[0] (left) and input[1] (right)
    - Aggregator i: combines result[i-1] (left) and input[i+1] (right)
    
    To neutralize the first k features, set weight_left=0, weight_right=1 for first k aggregators.
    """
    if num_neutralized == 0:
        with torch.no_grad():
            return (model(X).round() == y).float().mean().item()
    
    # Store original weight_mode and weights
    original_weight_mode = model.assembler.weight_mode
    original_weights = []
    
    # Temporarily switch to trainable mode so forward pass uses our modified weights
    model.assembler.weight_mode = "trainable"
    
    for i in range(num_neutralized):
        original_weights.append(model.assembler.weights[i].data.clone())
        # Set left weight to 0 (neutralize), right weight to 1 (pass through)
        model.assembler.weights[i].data[0] = 0.0  # left input weight = 0
        model.assembler.weights[i].data[1] = 1.0  # right input weight = 1
    
    with torch.no_grad():
        result = (model(X).round() == y).float().mean().item()
    
    # Restore original weights and weight_mode
    for i in range(num_neutralized):
        model.assembler.weights[i].data.copy_(original_weights[i])
    model.assembler.weight_mode = original_weight_mode
    
    return result

for i in range(1, X_test_tensor.shape[1]):
    masked_accuracy = evaluate_with_neutralized_features(bacon, X_test_tensor, Y_test_tensor, i)
    accuracies.append(masked_accuracy)
    print(f"✅ Accuracy with {i} feature(s) neutralized: {masked_accuracy * 100:.2f}%")

plt.figure(figsize=(10, 5))
plt.plot(range(1, len(accuracies) + 1), [a * 100 for a in accuracies], marker='o')
plt.title("Accuracy vs. Number of Features Pruned")
plt.xlabel("Number of Features Pruned from Left")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.tight_layout()
plt.show()