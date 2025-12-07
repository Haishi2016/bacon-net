import sys
sys.path.insert(0, '../../')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

print("📂 Loading Home Credit Default Risk dataset...")

# Load the dataset
if not pd.io.common.file_exists('./application_train.csv'):
    print("\n❌ application_train.csv not found!")
    sys.exit(1)

df = pd.read_csv('./application_train.csv')
print(f"✅ Loaded {len(df):,} applications")

# ========== Feature Sets ==========
numeric_features = [
    'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
    'CNT_CHILDREN', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION',
    'DAYS_ID_PUBLISH', 'DAYS_LAST_PHONE_CHANGE',
    'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
    'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY',
]

existing_numeric = [f for f in numeric_features if f in df.columns]

# Target variable (1 = default, 0 = repaid)
if 'TARGET' not in df.columns:
    print("❌ Error: TARGET column not found")
    sys.exit(1)

# ========== Normalization Functions ==========
def normalize_amount_feature(series, feature_name, percentile_cap=0.99):
    """Normalize financial amount features with outlier handling."""
    cap_value = series.quantile(percentile_cap)
    series_capped = series.clip(upper=cap_value)
    series_log = np.log1p(series_capped)
    min_val = series_log.min()
    max_val = series_log.max()
    series_normalized = (series_log - min_val) / (max_val - min_val + 1e-8)
    print(f"  ✅ {feature_name}: capped at {cap_value:.0f}, log-transformed, normalized")
    return series_normalized

print("\n🔄 Normalizing features...")
df_encoded = df.copy()

# DAYS_BIRTH: younger age → higher truth value
if 'DAYS_BIRTH' in existing_numeric:
    age_years = -df_encoded['DAYS_BIRTH'] / 365
    df_encoded['DAYS_BIRTH_NORM'] = 1 - ((age_years - age_years.min()) / (age_years.max() - age_years.min() + 1e-8))
    print(f"  ✅ DAYS_BIRTH: young age = high truth")

# DAYS_EMPLOYED: longer employment → higher truth
if 'DAYS_EMPLOYED' in existing_numeric:
    employed_days = df_encoded['DAYS_EMPLOYED'].replace({365243: np.nan})
    employed_years = -employed_days / 365
    employed_years = employed_years.clip(0, 50)
    df_encoded['DAYS_EMPLOYED_NORM'] = (employed_years - employed_years.min()) / (employed_years.max() - employed_years.min() + 1e-8)
    df_encoded['DAYS_EMPLOYED_NORM'] = df_encoded['DAYS_EMPLOYED_NORM'].fillna(0.0)
    print(f"  ✅ DAYS_EMPLOYED: long employment = high truth")

# Other DAYS features: more recent → higher truth
for days_feat in ['DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'DAYS_LAST_PHONE_CHANGE']:
    if days_feat in existing_numeric:
        days_val = -df_encoded[days_feat] / 365
        df_encoded[f'{days_feat}_NORM'] = 1 - ((days_val - days_val.min()) / (days_val.max() - days_val.min() + 1e-8))
        print(f"  ✅ {days_feat}: recent = high truth")

# AMT features: Apply robust normalization
for amt_feat in ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE']:
    if amt_feat in existing_numeric:
        df_encoded[f'{amt_feat}_NORM'] = normalize_amount_feature(df_encoded[amt_feat], amt_feat)

# CNT_CHILDREN: Normalize
if 'CNT_CHILDREN' in existing_numeric:
    val = df_encoded['CNT_CHILDREN']
    df_encoded['CNT_CHILDREN_NORM'] = val / (val.max() + 1e-8)
    print(f"  ✅ CNT_CHILDREN: normalized")

# EXT_SOURCE: Already [0,1]
for ext_feat in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']:
    if ext_feat in existing_numeric:
        df_encoded[f'{ext_feat}_NORM'] = df_encoded[ext_feat].fillna(0)
        print(f"  ✅ {ext_feat}: kept as-is (high = low risk)")

# Region ratings: Normalize
for region_feat in ['REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY']:
    if region_feat in existing_numeric:
        val = df_encoded[region_feat]
        df_encoded[f'{region_feat}_NORM'] = (val - val.min()) / (val.max() - val.min() + 1e-8)
        print(f"  ✅ {region_feat}: normalized")

# Derived ratios
print("\n📐 Deriving credit-risk ratios...")
df_encoded['CREDIT_INCOME_RATIO'] = df_encoded['AMT_CREDIT'] / (df_encoded['AMT_INCOME_TOTAL'] + 1e-8)
df_encoded['ANNUITY_INCOME_RATIO'] = df_encoded['AMT_ANNUITY'] / (df_encoded['AMT_INCOME_TOTAL'] + 1e-8)

for ratio_feat in ['CREDIT_INCOME_RATIO', 'ANNUITY_INCOME_RATIO']:
    val = df_encoded[ratio_feat]
    df_encoded[f'{ratio_feat}_NORM'] = val / (val.max() + 1e-8)
    print(f"  ✅ {ratio_feat}: normalized")

# ========== Collect normalized features ==========
normalized_features = [f'{feat}_NORM' for feat in existing_numeric] + ['CREDIT_INCOME_RATIO_NORM', 'ANNUITY_INCOME_RATIO_NORM']
normalized_features = [f for f in normalized_features if f in df_encoded.columns]

print(f"\n📊 Analyzing {len(normalized_features)} normalized features")

# Sample for visualization (use 5000 samples for clarity)
df_sample = df_encoded.sample(n=min(5000, len(df_encoded)), random_state=42)

# ========== Plot distributions by target ==========
print("\n📊 Creating distribution plots...")

# Select most important features to plot
important_features = [
    'AMT_INCOME_TOTAL_NORM', 'AMT_CREDIT_NORM', 'AMT_ANNUITY_NORM',
    'DAYS_BIRTH_NORM', 'DAYS_EMPLOYED_NORM',
    'EXT_SOURCE_1_NORM', 'EXT_SOURCE_2_NORM', 'EXT_SOURCE_3_NORM',
    'CREDIT_INCOME_RATIO_NORM', 'ANNUITY_INCOME_RATIO_NORM'
]
important_features = [f for f in important_features if f in normalized_features]

# Create figure with subplots
n_features = len(important_features)
n_cols = 3
n_rows = (n_features + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
axes = axes.flatten() if n_features > 1 else [axes]

for idx, feature in enumerate(important_features):
    ax = axes[idx]
    
    # Plot distributions for each target class
    df_sample[df_sample['TARGET'] == 0][feature].hist(
        bins=50, alpha=0.6, label='Repaid (0)', color='green', ax=ax, density=True
    )
    df_sample[df_sample['TARGET'] == 1][feature].hist(
        bins=50, alpha=0.6, label='Default (1)', color='red', ax=ax, density=True
    )
    
    # Calculate mean for each class
    mean_0 = df_sample[df_sample['TARGET'] == 0][feature].mean()
    mean_1 = df_sample[df_sample['TARGET'] == 1][feature].mean()
    
    ax.axvline(mean_0, color='green', linestyle='--', linewidth=2, alpha=0.8, label=f'Mean (0): {mean_0:.2f}')
    ax.axvline(mean_1, color='red', linestyle='--', linewidth=2, alpha=0.8, label=f'Mean (1): {mean_1:.2f}')
    
    ax.set_title(feature.replace('_NORM', ''), fontsize=10)
    ax.set_xlabel('Normalized Value [0,1]')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

# Remove extra subplots
for idx in range(n_features, len(axes)):
    fig.delaxes(axes[idx])

plt.tight_layout()
plt.savefig('feature_distributions_by_target.png', dpi=300, bbox_inches='tight')
print("💾 Saved: feature_distributions_by_target.png")

# ========== Box plots for key features ==========
print("\n📊 Creating box plots...")

fig2, axes2 = plt.subplots(2, 5, figsize=(18, 8))
axes2 = axes2.flatten()

for idx, feature in enumerate(important_features[:10]):
    ax = axes2[idx]
    
    data_to_plot = [
        df_sample[df_sample['TARGET'] == 0][feature].dropna(),
        df_sample[df_sample['TARGET'] == 1][feature].dropna()
    ]
    
    bp = ax.boxplot(data_to_plot, labels=['Repaid (0)', 'Default (1)'],
                    patch_artist=True, widths=0.6)
    
    # Color the boxes
    bp['boxes'][0].set_facecolor('lightgreen')
    bp['boxes'][1].set_facecolor('lightcoral')
    
    ax.set_title(feature.replace('_NORM', ''), fontsize=10)
    ax.set_ylabel('Normalized Value')
    ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('feature_boxplots_by_target.png', dpi=300, bbox_inches='tight')
print("💾 Saved: feature_boxplots_by_target.png")

# ========== Correlation with target ==========
print("\n📊 Computing correlations with TARGET...")

correlations = []
for feature in normalized_features:
    corr = df_encoded[[feature, 'TARGET']].corr().iloc[0, 1]
    correlations.append((feature.replace('_NORM', ''), corr))

correlations.sort(key=lambda x: abs(x[1]), reverse=True)

print("\n🔍 Top 15 Features by Correlation with Default:")
print(f"{'Feature':<35} {'Correlation':>12} {'Direction'}")
print("=" * 60)
for feature, corr in correlations[:15]:
    direction = "Higher = More Default" if corr > 0 else "Higher = Less Default"
    print(f"{feature:<35} {corr:>12.4f}  {direction}")

# Plot correlation bar chart
fig3, ax3 = plt.subplots(figsize=(10, 8))
features_sorted = [f[0] for f in correlations[:20]]
corrs_sorted = [f[1] for f in correlations[:20]]

colors = ['red' if c > 0 else 'green' for c in corrs_sorted]
ax3.barh(features_sorted, corrs_sorted, color=colors, alpha=0.7)
ax3.axvline(x=0, color='black', linestyle='--', linewidth=1)
ax3.set_xlabel('Correlation with TARGET (Default)')
ax3.set_title('Top 20 Features by Correlation with Default Risk')
ax3.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('feature_correlations.png', dpi=300, bbox_inches='tight')
print("\n💾 Saved: feature_correlations.png")

# ========== Statistics Summary ==========
print("\n📊 Feature Statistics Summary:")
print(f"{'Feature':<35} {'Mean(0)':<10} {'Mean(1)':<10} {'Std(0)':<10} {'Std(1)':<10}")
print("=" * 80)

for feature in important_features:
    mean_0 = df_encoded[df_encoded['TARGET'] == 0][feature].mean()
    mean_1 = df_encoded[df_encoded['TARGET'] == 1][feature].mean()
    std_0 = df_encoded[df_encoded['TARGET'] == 0][feature].std()
    std_1 = df_encoded[df_encoded['TARGET'] == 1][feature].std()
    
    print(f"{feature.replace('_NORM', ''):<35} {mean_0:<10.4f} {mean_1:<10.4f} {std_0:<10.4f} {std_1:<10.4f}")

print("\n✅ Analysis complete! Generated 3 visualization files:")
print("   1. feature_distributions_by_target.png - Histograms with means")
print("   2. feature_boxplots_by_target.png - Box plots showing quartiles")
print("   3. feature_correlations.png - Correlation with default risk")

plt.show()
