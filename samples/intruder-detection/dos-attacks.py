# The following two lines are needed to refer to local package instead of published package

import sys
sys.path.insert(0, '../../')
sys.path.insert(0, '../')

import pandas as pd
import torch
from bacon.baconNet import baconNet
from bacon.visualization import print_tree_structure, visualize_tree_structure
import logging
import matplotlib.pyplot as plt
from bacon.transformationLayer import IdentityTransformation, NegationTransformation
from bacon.utils import SigmoidScaler
from common import run_standard_analysis

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
    'back', 'land', 'neptune', 'pod', 'smurf', 'teardrop'    
]

# Binary labels
def is_dos(label):
    return 1.0 if label in dos_attacks else 0.0    
    
train_df['target'] = train_df['label'].apply(is_dos)
test_df['target'] = test_df['label'].apply(is_dos)

# Show positive rate (DoS attack rate) in data
train_positive_rate = train_df['target'].mean()
test_positive_rate = test_df['target'].mean()
print(f"Training set - Positive rate (DoS attacks): {train_positive_rate * 100:.2f}% ({int(train_df['target'].sum())}/{len(train_df)} samples)")
print(f"Test set - Positive rate (DoS attacks): {test_positive_rate * 100:.2f}% ({int(test_df['target'].sum())}/{len(test_df)} samples)")

# Calculate correlation between srv_count and target
srv_count_target_corr = train_df['srv_count'].corr(train_df['target'])
print(f"Correlation between srv_count and target: {srv_count_target_corr:.4f}")
srv_serror_rate_target_corr = train_df['srv_serror_rate'].corr(train_df['target'])
print(f"Correlation between srv_serror_rate and target: {srv_serror_rate_target_corr:.4f}")

drop_cols = ['label', 'target', 'protocol_type', 'service', 'flag', 'difficulty']

# Use unbalanced training data
X_train = train_df.drop(columns=drop_cols)
y_train = train_df['target']

X_test = test_df.drop(columns=drop_cols)
y_test = test_df['target']

scaler = SigmoidScaler(alpha=4, beta=-1)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

X_train_tensor = torch.tensor(X_train, dtype=torch.float32).to(device)
Y_train_tensor = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
Y_test_tensor = torch.tensor(y_test.values.reshape(-1, 1), dtype=torch.float32).to(device)

# Use actual input size for transformations
num_features = X_train.shape[1]

bacon = baconNet(
    input_size=num_features, 
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
    acceptance_threshold=0.85
)
print(f"✅ Best accuracy: {best_accuracy * 100:.2f}%")

filtered_features = [f for f in feature_names if f not in drop_cols]

# Run standard analysis
run_standard_analysis(
    bacon,
    X_train_tensor, Y_train_tensor, X_test_tensor, Y_test_tensor,
    filtered_features,
    title_prefix="DoS Attacks"
)
