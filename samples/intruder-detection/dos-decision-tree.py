import pandas as pd
from sklearn.preprocessing import QuantileTransformer
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
from sklearn import tree

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

print("📊 Dataset loaded")
print(f"Training samples: {len(train_df)}")
print(f"Test samples: {len(test_df)}")

# Define attack types
dos_attacks = [
    'back', 'land', 'neptune', 'pod', 'smurf', 'teardrop'    
]

# Binary labels
def is_dos(label):
    return 1.0 if label in dos_attacks else 0.0
    
train_df['target'] = train_df['label'].apply(is_dos)
test_df['target'] = test_df['label'].apply(is_dos)

drop_cols = ['label', 'target', 'protocol_type', 'service', 'flag', 'difficulty']

# Balance the training data BEFORE creating X_train/y_train
df_majority = train_df[train_df['target'] == 1]
df_minority = train_df[train_df['target'] == 0]

print(f"\n📊 Before balancing:")
print(f"DoS attacks: {len(df_majority)}")
print(f"Normal: {len(df_minority)}")

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

print(f"\n📊 After balancing:")
print(train_df['target'].value_counts())

# Use unbalanced training data
X_train = train_df.drop(columns=drop_cols)
y_train = train_df['target']

X_test = test_df.drop(columns=drop_cols)
y_test = test_df['target']

# Get feature names for later
feature_names_filtered = [f for f in feature_names if f not in drop_cols]

print(f"\n📊 Features used: {len(feature_names_filtered)}")
print(f"Features: {feature_names_filtered}")

# Apply same scaling as BACON model
scaler = QuantileTransformer(
    output_distribution="uniform",  # -> [0,1]
    n_quantiles=1000,
    subsample=int(1e5),
    random_state=42
)

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n🌲 Training Decision Tree...")

# Train Decision Tree with same depth as BACON tree structure
# BACON has binary tree with ~41 features, so max_depth ~6-7 is comparable
dt_classifier = DecisionTreeClassifier(
    max_depth=7,  # Comparable to BACON's binary tree depth
    min_samples_split=100,  # Prevent overfitting
    min_samples_leaf=50,
    random_state=42
)

dt_classifier.fit(X_train_scaled, y_train)

print("✅ Decision Tree trained")

# Evaluate on training set
y_train_pred = dt_classifier.predict(X_train_scaled)
train_accuracy = accuracy_score(y_train, y_train_pred)
print(f"\n📊 Training Accuracy: {train_accuracy * 100:.2f}%")

# Evaluate on test set
y_test_pred = dt_classifier.predict(X_test_scaled)
test_accuracy = accuracy_score(y_test, y_test_pred)
print(f"📊 Test Accuracy: {test_accuracy * 100:.2f}%")

print("\n📊 Classification Report:")
print(classification_report(y_test, y_test_pred, target_names=['Normal', 'DoS']))

print("\n📊 Confusion Matrix:")
cm = confusion_matrix(y_test, y_test_pred)
print(cm)
print(f"True Negatives: {cm[0][0]}, False Positives: {cm[0][1]}")
print(f"False Negatives: {cm[1][0]}, True Positives: {cm[1][1]}")

# Feature importance
print("\n📊 Top 10 Most Important Features:")
feature_importance = dt_classifier.feature_importances_
feature_importance_dict = dict(zip(feature_names_filtered, feature_importance))
sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

for i, (feature, importance) in enumerate(sorted_features[:10], 1):
    print(f"{i}. {feature}: {importance:.4f}")

# Visualize the decision tree (first few levels)
print("\n📊 Visualizing Decision Tree (saving to file)...")
plt.figure(figsize=(20, 10))
tree.plot_tree(
    dt_classifier, 
    max_depth=3,  # Only show first 3 levels for readability
    feature_names=feature_names_filtered,
    class_names=['Normal', 'DoS'],
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Decision Tree for DoS Detection (First 3 Levels)")
plt.tight_layout()
plt.savefig('dos_decision_tree.png', dpi=150, bbox_inches='tight')
print("✅ Decision tree visualization saved to 'dos_decision_tree.png'")

print(f"\n✅ Final Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"📊 Tree depth: {dt_classifier.get_depth()}")
print(f"📊 Number of leaves: {dt_classifier.get_n_leaves()}")
