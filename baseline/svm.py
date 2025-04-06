from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
import shap

# Load the Breast Cancer Wisconsin Diagnostic dataset
breast_cancer = fetch_ucirepo(id=17)
X = breast_cancer.data.features  # Use only mean values
y = breast_cancer.data.targets
feature_names = breast_cancer.data.feature_names

# Encode labels if necessary
y = LabelEncoder().fit_transform(y.values.ravel())

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Support Vector Machine
svm_clf = SVC(kernel='rbf', probability=True, random_state=42)
svm_clf.fit(X_train, y_train)

# Evaluate SVM
y_pred_svm = svm_clf.predict(X_test)

print("\nSVM Classification Report:")
print(classification_report(y_test, y_pred_svm))
print("SVM Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_svm))
print(f"SVM Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")
print(f"SVM F1 Score: {f1_score(y_test, y_pred_svm):.4f}")

# # Use a subset of the training data as background for KernelExplainer
# X_background = shap.sample(X_train, 100, random_state=42)  # Small representative set
# explainer = shap.KernelExplainer(svm_clf.predict_proba, X_background)

# # Compute SHAP values on a subset of test data for speed
# X_explain = X_test[:100]
# shap_values = explainer.shap_values(X_explain)

# # Visualization: SHAP summary plot for class 1 (malignant)
# shap.summary_plot(shap_values[1], features=X_explain, feature_names=feature_names)
