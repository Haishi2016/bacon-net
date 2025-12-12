# Explainable rule-based approach for 1000-variable boolean expression inference
# Using imodels library for interpretable rule-based models

import sys
sys.path.insert(0, '../../')

import torch
from bacon.utils import generate_classic_boolean_data
import time
import numpy as np
from imodels import BoostedRulesClassifier

print("=" * 80)
print("📋 Explainable Rule-Based: 1000-Variable Boolean Expression Inference")
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
    repeat_factor=5000,  # 5k test samples  
    randomize=True,
    device=device
)

# Convert to numpy
var_names = expr_info['var_names']
x_train_np = x_train.cpu().numpy()
y_train_np = y_train.cpu().numpy().ravel().astype(int)
x_test_np = x_test.cpu().numpy()
y_test_np = y_test.cpu().numpy().ravel().astype(int)

print(f"✅ Data generated")
print(f"   Training samples: {len(x_train_np)}")
print(f"   Test samples: {len(x_test_np)}")
print(f"   Expression: {expr_info['expression_text'][:100]}...")  # Show first 100 chars
print(f"   Variables: {var_names[:5]}...{var_names[-5:]}")

print("\n🔧 Configuring Boosted Rules Classifier...")
# BoostedRulesClassifier generates interpretable IF-THEN rules
# Based on RuleFit algorithm (gradient boosting with interpretable rules)
classifier = BoostedRulesClassifier(
    n_estimators=100,  # Number of boosting iterations
    random_state=42
)
print("✅ Model configured")
print(f"   Algorithm: Boosted Rules (RuleFit - gradient boosting with IF-THEN rules)")
print(f"   Estimators: 100")

print("\n🔥 Training model...")
start_time = time.time()

classifier.fit(x_train_np, y_train_np)

training_time = time.time() - start_time

print("✅ Training complete")

# Evaluate
y_train_pred = classifier.predict(x_train_np)
y_test_pred = classifier.predict(x_test_np)

train_accuracy = np.mean(y_train_pred == y_train_np)
test_accuracy = np.mean(y_test_pred == y_test_np)

print("\n" + "=" * 80)
print("📊 EXPLAINABLE RULES RESULTS")
print("=" * 80)
print(f"🏆 Test Accuracy: {test_accuracy * 100:.2f}%")
print(f"📊 Training Accuracy: {train_accuracy * 100:.2f}%")
print(f"⏱️  Training Time: {training_time:.2f} seconds")

# Show learned rules
if hasattr(classifier, 'rules_'):
    print("\n📋 Learned Rules:")
    for i, rule in enumerate(classifier.rules_[:10], 1):  # Show first 10 rules
        print(f"   {i}. {rule}")
    if len(classifier.rules_) > 10:
        print(f"   ... and {len(classifier.rules_) - 10} more rules")
    print(f"\n📊 Total rules: {len(classifier.rules_)}")

# Feature importance
if hasattr(classifier, 'feature_importances_'):
    print("\n📊 Top 20 Most Important Features:")
    feature_importance = classifier.feature_importances_
    feature_importance_dict = dict(zip(var_names, feature_importance))
    sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)

    for i, (feature, importance) in enumerate(sorted_features[:20], 1):
        if importance > 0:
            print(f"   {i}. {feature}: {importance:.4f}")

    # Count features used
    features_used = sum(1 for imp in feature_importance if imp > 0)
    print(f"\n📊 Features actually used: {features_used}/{input_size}")

# Classification metrics
from sklearn.metrics import classification_report, confusion_matrix

print("\n📊 Classification Report:")
print(classification_report(y_test_np, y_test_pred, target_names=['False', 'True']))

print("\n📊 Confusion Matrix:")
cm = confusion_matrix(y_test_np, y_test_pred)
print(cm)
print(f"True Negatives: {cm[0][0]}, False Positives: {cm[0][1]}")
print(f"False Negatives: {cm[1][0]}, True Positives: {cm[1][1]}")

print("\n✅ Explainable rule-based benchmark complete!")
print("💡 Using imodels BoostedRulesClassifier - generates human-readable IF-THEN rules")
