from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the Breast Cancer Wisconsin Diagnostic dataset
breast_cancer = fetch_ucirepo(id=17)
X = breast_cancer.data.features
y = breast_cancer.data.targets

# Encode labels if necessary
y = LabelEncoder().fit_transform(y.values.ravel())

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=None)

# Standardize the features
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Decision Tree with depth limit for compactness
clf = DecisionTreeClassifier(max_depth=2, random_state=None)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")

# Visualize the Decision Tree
plt.figure(figsize=(12, 6))
plot_tree(clf, feature_names=X.columns.tolist(), class_names=['Benign', 'Malignant'], filled=True, rounded=True)
plt.title("Compact Decision Tree (max_depth=2)")
plt.show()