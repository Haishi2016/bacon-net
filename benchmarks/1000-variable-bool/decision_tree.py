# Decision Tree approach for 1000-variable boolean expression inference

import sys
sys.path.insert(0, '../../')

from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
import torch
from bacon.utils import generate_classic_boolean_data
import time

print("=" * 80)
print("🌲 Decision Tree: 1000-Variable Boolean Expression Inference")
print("=" * 80)

input_size = 1000

print(f"\n📊 Generating boolean expression with {input_size} variables...")
print(f"   Using randomized sampling (10,000 samples)")

# Generate same data as BACON
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

x_train, y_train, expr_info = generate_classic_boolean_data(
    input_size, 
    repeat_factor=10000,  # 10k training samples
    randomize=True, 
    device=device
)

x_test, y_test, _ = generate_classic_boolean_data(
    input_size,
    repeat_factor=10000,  # 5k test samples  
    randomize=True,
    device=device
)

# Convert to numpy for sklearn
x_train_np = x_train.cpu().numpy()
y_train_np = y_train.cpu().numpy().ravel()
x_test_np = x_test.cpu().numpy()
y_test_np = y_test.cpu().numpy().ravel()

print(f"✅ Data generated")
print(f"   Training samples: {len(x_train_np)}")
print(f"   Test samples: {len(x_test_np)}")
print(f"   Expression: {expr_info['expression_text'][:100]}...")  # Show first 100 chars

print("\n🔧 Configuring Decision Tree...")
# Use comparable depth to BACON's binary tree
# For 1000 inputs, binary tree depth is ~log2(1000) ≈ 10
dt_classifier = DecisionTreeClassifier(
    random_state=42,
    criterion="gini",
    max_depth=None,           # let it grow
    min_samples_split=2,
    min_samples_leaf=1
)

print("✅ Model configured")
print(f"   Max depth: 15")
print(f"   Min samples split: 20")
print(f"   Min samples leaf: 10")

print("\n🔥 Training Decision Tree...")
start_time = time.time()

dt_classifier.fit(x_train_np, y_train_np)

training_time = time.time() - start_time

print("✅ Training complete")

# Evaluate
y_train_pred = dt_classifier.predict(x_train_np)
y_test_pred = dt_classifier.predict(x_test_np)

train_accuracy = accuracy_score(y_train_np, y_train_pred)
test_accuracy = accuracy_score(y_test_np, y_test_pred)

print("\n" + "=" * 80)
print("📊 DECISION TREE RESULTS")
print("=" * 80)
print(f"🏆 Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"📊 Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"⏱️  Training Time: {training_time:.2f} seconds")

print("\n📊 Tree Statistics:")
print(f"   Actual depth: {dt_classifier.get_depth()}")
print(f"   Number of leaves: {dt_classifier.get_n_leaves()}")
print(f"   Total nodes: {dt_classifier.tree_.node_count}")

# Feature importance (top 20)
print("\n📊 Top 20 Most Important Features:")
feature_importance = dt_classifier.feature_importances_
var_names = expr_info['var_names']
feature_importance_dict = dict(zip(var_names, feature_importance))
sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

for i, (feature, importance) in enumerate(sorted_features[:20], 1):
    if importance > 0:
        print(f"   {i}. {feature}: {importance:.4f}")

# Count how many features are actually used
features_used = sum(1 for imp in feature_importance if imp > 0)
print(f"\n📊 Features actually used: {features_used}/{input_size}")

print("\n📊 Classification Report:")
print(classification_report(y_test_np, y_test_pred, target_names=['False', 'True']))

print("\n✅ Decision Tree benchmark complete!")
