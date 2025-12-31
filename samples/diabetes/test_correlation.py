"""Test correlation among specific diabetes features."""

import sys
sys.path.insert(0, '../../')

from ucimlrepo import fetch_ucirepo
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Fetch dataset
print("Loading CDC Diabetes Health Indicators dataset...")
diabetes = fetch_ucirepo(id=891)

# Extract features
X = diabetes.data.features
y = diabetes.data.targets

# Create dataframe with features of interest
features_of_interest = ['Income', 'HvyAlcoholConsump', 'CholCheck', 'HighBP', 'HighChol']
df = X[features_of_interest].copy()

print("\n" + "="*70)
print("CORRELATION ANALYSIS: Diabetes Risk Factors")
print("="*70)

# Basic statistics
print("\n📊 Feature Statistics:")
print(df.describe())

print("\n📈 Feature Value Counts:")
for col in features_of_interest:
    print(f"\n{col}:")
    print(df[col].value_counts().sort_index())

# Correlation matrix
print("\n" + "="*70)
print("CORRELATION MATRIX")
print("="*70)
corr_matrix = df.corr()
print(corr_matrix.round(3))

# Pairwise correlations sorted by strength
print("\n📊 Pairwise Correlations (sorted by strength):")
print("-" * 70)
correlations = []
for i in range(len(features_of_interest)):
    for j in range(i+1, len(features_of_interest)):
        feat1 = features_of_interest[i]
        feat2 = features_of_interest[j]
        corr_value = corr_matrix.loc[feat1, feat2]
        correlations.append((feat1, feat2, corr_value))

# Sort by absolute correlation value
correlations.sort(key=lambda x: abs(x[2]), reverse=True)

for feat1, feat2, corr_val in correlations:
    strength = "STRONG" if abs(corr_val) > 0.5 else "MODERATE" if abs(corr_val) > 0.3 else "WEAK"
    print(f"{feat1:25s} ↔ {feat2:25s}: {corr_val:7.4f} [{strength}]")

# Correlation with target (Diabetes_binary)
print("\n" + "="*70)
print("CORRELATION WITH DIABETES TARGET")
print("="*70)
df_with_target = df.copy()
df_with_target['Diabetes_binary'] = y['Diabetes_binary'].values
target_corr = df_with_target.corr()['Diabetes_binary'].drop('Diabetes_binary').sort_values(ascending=False)
print("\n📊 Feature correlations with Diabetes:")
for feat, corr_val in target_corr.items():
    strength = "STRONG" if abs(corr_val) > 0.5 else "MODERATE" if abs(corr_val) > 0.3 else "WEAK"
    print(f"{feat:30s}: {corr_val:7.4f} [{strength}]")

# Cross-tabulation for categorical pairs
print("\n" + "="*70)
print("CONTINGENCY TABLES (High Correlation Pairs)")
print("="*70)

# Show detailed cross-tabs for the top 3 correlations
for i, (feat1, feat2, corr_val) in enumerate(correlations[:3]):
    print(f"\n{i+1}. {feat1} vs {feat2} (correlation: {corr_val:.4f}):")
    print("-" * 50)
    crosstab = pd.crosstab(df[feat1], df[feat2], margins=True)
    print(crosstab)
    
    # Calculate conditional probabilities
    print(f"\n   P({feat2}=1 | {feat1}):")
    for val in sorted(df[feat1].unique()):
        if val != 'All':
            subset = df[df[feat1] == val]
            prob = (subset[feat2] == 1).mean()
            print(f"   {feat1}={val}: {prob:.4f} ({prob*100:.2f}%)")

# Visualize correlation matrix
print("\n📊 Generating correlation heatmap...")
plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
            square=True, linewidths=1, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix: Diabetes Risk Factors', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('diabetes_correlation_matrix.png', dpi=150, bbox_inches='tight')
print("✅ Saved correlation heatmap to: diabetes_correlation_matrix.png")

# Visualize pairwise scatter plots for top correlations
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for i, (feat1, feat2, corr_val) in enumerate(correlations[:3]):
    # Jitter for better visualization of binary data
    x_jitter = df[feat1] + np.random.normal(0, 0.02, len(df))
    y_jitter = df[feat2] + np.random.normal(0, 0.02, len(df))
    
    axes[i].scatter(x_jitter, y_jitter, alpha=0.1, s=1)
    axes[i].set_xlabel(feat1)
    axes[i].set_ylabel(feat2)
    axes[i].set_title(f'r = {corr_val:.4f}')
    axes[i].grid(True, alpha=0.3)

plt.suptitle('Top 3 Correlated Feature Pairs', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('diabetes_top_correlations.png', dpi=150, bbox_inches='tight')
print("✅ Saved scatter plots to: diabetes_top_correlations.png")

print("\n" + "="*70)
print("✅ Analysis complete!")
print("="*70)
