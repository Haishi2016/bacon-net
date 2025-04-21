from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np

# Load the Breast Cancer Wisconsin Diagnostic dataset
breast_cancer = fetch_ucirepo(id=17)
X_full = breast_cancer.data.features
y = breast_cancer.data.targets

# Select only the top 5 features discovered via BACON
selected_features = ['radius2', 'radius3', 'texture3', 'concave_points1', 'smoothness3']
X = X_full[selected_features]

# Encode labels if necessary
y = LabelEncoder().fit_transform(y.values.ravel())

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

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
