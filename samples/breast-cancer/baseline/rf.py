from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from ucimlrepo import fetch_ucirepo
from sklearn.tree import plot_tree
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
# Load the Breast Cancer Wisconsin Diagnostic dataset
breast_cancer = fetch_ucirepo(id=17)
X = breast_cancer.data.features
y = breast_cancer.data.targets

# Ensure it's a DataFrame
X = pd.DataFrame(X, columns=breast_cancer.data.feature_names)

# Encode labels if necessary
y = LabelEncoder().fit_transform(y.values.ravel())

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features, keeping DataFrame format
scaler = MinMaxScaler()
X_train = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
X_test = pd.DataFrame(scaler.transform(X_test), columns=X.columns)

# Train Random Forest
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")

# Find the best tree in the ensemble based on its own predictions
best_tree = None
best_tree_score = 0
best_tree_index = -1

for i, tree in enumerate(clf.estimators_):
    y_tree_pred = tree.predict(X_test)
    acc = accuracy_score(y_test, y_tree_pred)
    if acc > best_tree_score:
        best_tree_score = acc
        best_tree = tree
        best_tree_index = i

print(f"\nBest individual tree is Tree #{best_tree_index} with accuracy {best_tree_score:.4f}")

# Plot the best tree (limit depth for readability)
plt.figure(figsize=(20, 10))
plot_tree(best_tree, 
          feature_names=X.columns,  # ✅ Always use DataFrame columns now
          class_names=['Benign', 'Malignant'],
          filled=True, 
          max_depth=2)
plt.title(f"Best Tree #{best_tree_index} (Accuracy: {best_tree_score:.4f}) - Limited to depth 3")
plt.show()

# Extract feature importances from the trained Random Forest
importances = clf.feature_importances_

# Create a DataFrame for easy viewing and sorting
feat_imp_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': importances
}).sort_values('Importance', ascending=False)

# Plot
plt.figure(figsize=(10, 6))
plt.barh(feat_imp_df['Feature'], feat_imp_df['Importance'])
plt.gca().invert_yaxis()  # Highest importance at the top
plt.xlabel("Feature Importance (Gini Importance)")
plt.title("Global Feature Importance from Random Forest")
plt.grid(True, axis='x', linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()

# Also print top features
print("\nTop Features Globally:")
print(feat_imp_df.head(10))