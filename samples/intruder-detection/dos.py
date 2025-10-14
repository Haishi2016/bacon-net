import copy
import pandas as pd
from sklearn.preprocessing import RobustScaler
import sys

sys.path.append('../../')
import torch
from bacon.baconNet import baconNet
from bacon.frozonInputToLeaf import frozenInputToLeaf

import logging
import matplotlib.pyplot as plt
from sklearn.utils import resample

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

# Define attack types
dos_attacks = [
    'back', 'land', 'neptune', 'pod', 'smurf', 'teardrop',
    'apache2', 'udpstorm', 'processtable', 'worm'
]

# Binary labels
def is_dos(label):
    # return 1.0 if label in dos_attacks else 0.0
    return 1.0 if label == 'neptune' else 0.0
    #return 0.0 if label == "normal" else 1.0

train_df['target'] = train_df['label'].apply(is_dos)
test_df['target'] = test_df['label'].apply(is_dos)

drop_cols = ['label', 'target', 'protocol_type', 'service', 'flag', 'difficulty']
X_train = train_df.drop(columns=drop_cols)
y_train = train_df['target']

X_test = test_df.drop(columns=drop_cols)
y_test = test_df['target']

df_majority = train_df[train_df['target'] == 1]
df_minority = train_df[train_df['target'] == 0]

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
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
Y_train_tensor = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
Y_test_tensor = torch.tensor(y_test.values.reshape(-1, 1), dtype=torch.float32)

bacon = baconNet(input_size=X_train.shape[1])

X_test_tensor = X_test_tensor.to(device)
Y_test_tensor = Y_test_tensor.to(device)    
X_train_tensor = X_train_tensor.to(device)
Y_train_tensor = Y_train_tensor.to(device)  

best_model, best_accuracy = bacon.find_best_model(
    X_train_tensor, Y_train_tensor, X_test_tensor, Y_test_tensor, 
    attempts=100, acceptance_threshold=0.90
)
print(f"✅ Best accuracy: {best_accuracy * 100:.2f}%")

filtered_features = [f for f in feature_names if f not in drop_cols]


# Visualize the BACON model's tree structure
bacon.print_tree_structure(filtered_features)
bacon.visualize_tree_structure(filtered_features)

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

disjunction_threshold = -1  # Tune this to control aggressiveness
accuracies = []

# Try pruning based on disjunction magnitude
func_eval = bacon.assembler.prune_by_disjunction(disjunction_threshold)

with torch.no_grad():
    pruned_output = func_eval(X_test_tensor)  # no need to change inputs
    pruned_accuracy = (pruned_output.round() == Y_test_tensor).float().mean().item()
    accuracies.append(pruned_accuracy)
    print(f"✅ Accuracy after disjunction pruning (a > {disjunction_threshold}): {pruned_accuracy * 100:.2f}%")

n = 27

func_eval_input = bacon.prune_features(n)
kept_indices = bacon.assembler.locked_perm[n:].tolist()
X_test_pruned = X_test_tensor[:, kept_indices]

with torch.no_grad():
    pruned_output = func_eval_input(X_test_pruned)
    pruned_accuracy = (pruned_output.round() == Y_test_tensor).float().mean().item()
    print(f"✅ Accuracy after pruning {n} feature(s): {pruned_accuracy * 100:.2f}%")

# # Create a shallow copy of the frozen model
# bacon_small = copy.deepcopy(bacon)
# bacon_small.assembler.num_leaves = len(kept_indices)
# bacon_small.assembler.weights = bacon.assembler.weights[n:]
# bacon_small.assembler.biases = bacon.assembler.biases[n:]
# bacon_small.assembler.fc_out = copy.deepcopy(bacon.assembler.fc_out)
# bacon_small.assembler.locked_perm = bacon.assembler.locked_perm[n:]
# bacon_small.assembler.input_to_leaf = frozenInputToLeaf(torch.arange(len(kept_indices)), len(kept_indices))
# bacon_small.assembler.is_frozen = True

# func_eval_combined = bacon_small.assembler.prune_by_disjunction()

# with torch.no_grad():
#     pruned_output = func_eval_combined(X_test_pruned)
#     pruned_accuracy = (pruned_output.round() == Y_test_tensor).float().mean().item()
#     print(f"🎯 Combined pruning (inputs + disjunctions) accuracy: {pruned_accuracy:.2%}")

