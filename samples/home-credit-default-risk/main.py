import sys
sys.path.insert(0, '../../')  # Use insert(0, ...) to prioritize local version over installed package
from sklearn.model_selection import train_test_split
import torch
from bacon.baconNet import baconNet
from bacon.visualization import visualize_tree_structure, print_tree_structure
import logging
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("📂 Loading Home Credit Default Risk dataset...")

# Load the dataset
if not pd.io.common.file_exists('./application_train.csv'):
    print("\n❌ application_train.csv not found!")
    print("\nPlease download the dataset:")
    print("  1. Go to https://www.kaggle.com/c/home-credit-default-risk/data")
    print("  2. Download application_train.csv")
    print("  3. Place it in this directory")
    print("\nNote: You may need to accept the competition rules first.")
    sys.exit(1)

df = pd.read_csv('./application_train.csv')
print(f"✅ Loaded {len(df):,} applications")

# ========== Microsoft Research Feature Set ==========
# Following InterpretML conventions for Home Credit Default Risk

# Numeric features
numeric_features = [
    'AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE',
    'CNT_CHILDREN', 'DAYS_BIRTH', 'DAYS_EMPLOYED', 'DAYS_REGISTRATION',
    'DAYS_ID_PUBLISH', 'DAYS_LAST_PHONE_CHANGE',
    'EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3',
    'AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY',
    'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON',
    'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR',
    'REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY',
]

# Binary features (already 0/1)
binary_features = [
    'REG_CITY_NOT_LIVE_CITY', 'REG_CITY_NOT_WORK_CITY',
    'LIVE_CITY_NOT_WORK_CITY', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY',
]

# Categorical features
categorical_features = [
    'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE',
    'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE',
]

# Filter to existing columns
existing_numeric = [f for f in numeric_features if f in df.columns]
existing_binary = [f for f in binary_features if f in df.columns]
existing_categorical = [f for f in categorical_features if f in df.columns]

print(f"\n✅ Found {len(existing_numeric)} numeric features")
print(f"✅ Found {len(existing_binary)} binary features")
print(f"✅ Found {len(existing_categorical)} categorical features")

# Derive credit-risk ratios (MSR convention)
print("\n📐 Deriving credit-risk ratios...")
df['CREDIT_INCOME_RATIO'] = df['AMT_CREDIT'] / (df['AMT_INCOME_TOTAL'] + 1e-8)
df['ANNUITY_INCOME_RATIO'] = df['AMT_ANNUITY'] / (df['AMT_INCOME_TOTAL'] + 1e-8)
derived_features = ['CREDIT_INCOME_RATIO', 'ANNUITY_INCOME_RATIO']

# Target variable
if 'TARGET' not in df.columns:
    print("❌ Error: TARGET column not found")
    sys.exit(1)

# TARGET: 1 = default, 0 = repaid
# For BACON: 1 = approve (won't default), 0 = reject (will default)
df['Approved'] = 1 - df['TARGET']

print(f"\n📊 Dataset statistics:")
print(f"  Total applications: {len(df):,}")
print(f"  Default rate: {df['TARGET'].mean() * 100:.2f}%")

# ========== Handle Categorical Features ==========
print("\n🔄 One-hot encoding categorical features...")
df_encoded = df.copy()
categorical_columns = []

for cat_feature in existing_categorical:
    # Fill missing values with "Missing"
    df_encoded[cat_feature] = df_encoded[cat_feature].fillna("Missing").astype(str)
    # One-hot encode
    dummies = pd.get_dummies(df_encoded[cat_feature], prefix=cat_feature, drop_first=False)
    df_encoded = pd.concat([df_encoded, dummies], axis=1)
    categorical_columns.extend(dummies.columns.tolist())
    print(f"  ✅ {cat_feature}: {len(dummies.columns)} categories")

# Binary features: convert Y/N to 1/0 if needed
for bin_feature in existing_binary:
    if bin_feature in df_encoded.columns:
        if df_encoded[bin_feature].dtype == 'object':
            df_encoded[bin_feature] = (df_encoded[bin_feature] == 'Y').astype(float)
        else:
            df_encoded[bin_feature] = df_encoded[bin_feature].astype(float)

print(f"\n✅ Created {len(categorical_columns)} one-hot encoded features")

# ========== Normalize to "Level of Truth" [0,1] ==========
print("\n🔄 Normalizing features to [0,1] as 'level of truth'...")

# Days features: Convert negative days to positive, then normalize
# DAYS_BIRTH: younger age → higher truth value (lower risk)
if 'DAYS_BIRTH' in existing_numeric:
    age_years = -df_encoded['DAYS_BIRTH'] / 365
    # Invert: young = 1, old = 0
    df_encoded['DAYS_BIRTH'] = 1 - ((age_years - age_years.min()) / (age_years.max() - age_years.min() + 1e-8))
    print(f"  ✅ DAYS_BIRTH: young age = high truth")

# DAYS_EMPLOYED: longer employment → higher truth (more stable)
if 'DAYS_EMPLOYED' in existing_numeric:
    # Handle anomalous positive values (unemployed coded as 365243)
    employed_days = df_encoded['DAYS_EMPLOYED'].replace({365243: np.nan})
    employed_years = -employed_days / 365
    employed_years = employed_years.clip(0, 50)  # Cap at 50 years
    df_encoded['DAYS_EMPLOYED'] = (employed_years - employed_years.min()) / (employed_years.max() - employed_years.min() + 1e-8)
    df_encoded['DAYS_EMPLOYED'] = df_encoded['DAYS_EMPLOYED'].fillna(0.5)  # Missing = unknown
    print(f"  ✅ DAYS_EMPLOYED: long employment = high truth")

# Other DAYS features: more recent → higher truth
for days_feat in ['DAYS_REGISTRATION', 'DAYS_ID_PUBLISH', 'DAYS_LAST_PHONE_CHANGE']:
    if days_feat in existing_numeric:
        days_val = -df_encoded[days_feat] / 365  # Convert to positive years
        # Invert: recent = 1, long ago = 0
        df_encoded[days_feat] = 1 - ((days_val - days_val.min()) / (days_val.max() - days_val.min() + 1e-8))
        print(f"  ✅ {days_feat}: recent = high truth")

# AMT features: Normalize min-max
for amt_feat in ['AMT_INCOME_TOTAL', 'AMT_CREDIT', 'AMT_ANNUITY', 'AMT_GOODS_PRICE']:
    if amt_feat in existing_numeric:
        val = df_encoded[amt_feat]
        df_encoded[amt_feat] = (val - val.min()) / (val.max() - val.min() + 1e-8)
        print(f"  ✅ {amt_feat}: normalized to [0,1]")

# CNT_CHILDREN: Normalize
if 'CNT_CHILDREN' in existing_numeric:
    val = df_encoded['CNT_CHILDREN']
    df_encoded['CNT_CHILDREN'] = val / (val.max() + 1e-8)
    print(f"  ✅ CNT_CHILDREN: normalized")

# EXT_SOURCE: Already [0,1], but higher EXT_SOURCE = lower risk
# Invert so high value = high risk (consistent semantics)
for ext_feat in ['EXT_SOURCE_1', 'EXT_SOURCE_2', 'EXT_SOURCE_3']:
    if ext_feat in existing_numeric:
        df_encoded[ext_feat] = 1 - df_encoded[ext_feat].fillna(0.5)  # Missing = 0.5 (unknown)
        print(f"  ✅ {ext_feat}: inverted (high = high risk)")

# Credit bureau requests: Normalize
for bureau_feat in ['AMT_REQ_CREDIT_BUREAU_HOUR', 'AMT_REQ_CREDIT_BUREAU_DAY',
                     'AMT_REQ_CREDIT_BUREAU_WEEK', 'AMT_REQ_CREDIT_BUREAU_MON',
                     'AMT_REQ_CREDIT_BUREAU_QRT', 'AMT_REQ_CREDIT_BUREAU_YEAR']:
    if bureau_feat in existing_numeric:
        val = df_encoded[bureau_feat].fillna(0)
        df_encoded[bureau_feat] = val / (val.max() + 1e-8)
        print(f"  ✅ {bureau_feat}: normalized")

# Region ratings: Normalize
for region_feat in ['REGION_RATING_CLIENT', 'REGION_RATING_CLIENT_W_CITY']:
    if region_feat in existing_numeric:
        val = df_encoded[region_feat]
        df_encoded[region_feat] = (val - val.min()) / (val.max() - val.min() + 1e-8)
        print(f"  ✅ {region_feat}: normalized")

# Derived ratios: Normalize (higher ratio = more strain = higher risk)
for ratio_feat in derived_features:
    val = df_encoded[ratio_feat]
    df_encoded[ratio_feat] = val / (val.max() + 1e-8)
    print(f"  ✅ {ratio_feat}: normalized (high ratio = high truth)")

# Combine all features
all_features = existing_numeric + existing_binary + derived_features + categorical_columns

# Fill any remaining NaN with 0.5 (unknown truth)
df_final = df_encoded[all_features + ['Approved']].fillna(0.5)

print(f"\n📊 Final feature count: {len(all_features)} features")
print(f"   - {len(existing_numeric) + len(derived_features)} numeric")
print(f"   - {len(existing_binary)} binary")
print(f"   - {len(categorical_columns)} categorical (one-hot)")

# Balance dataset
print(f"\n📊 Dataset size: {len(df_final):,} samples")
print(f"📈 Approval rate: {df_final['Approved'].mean() * 100:.2f}%")

if df_final['Approved'].mean() < 0.3 or df_final['Approved'].mean() > 0.7:
    print("\n⚖️ Balancing dataset...")
    approved_df = df_final[df_final['Approved'] == 1]
    rejected_df = df_final[df_final['Approved'] == 0]
    
    min_samples = min(len(approved_df), len(rejected_df))
    min_samples = min(min_samples, 20000)  # Cap for training efficiency
    
    balanced_df = pd.concat([
        approved_df.sample(n=min_samples, random_state=42),
        rejected_df.sample(n=min_samples, random_state=42)
    ])
    df_final = balanced_df.sample(frac=1, random_state=42).reset_index(drop=True)
    print(f"✅ Balanced dataset size: {len(df_final):,} samples")
    print(f"📈 New approval rate: {df_final['Approved'].mean() * 100:.2f}%")

X = df_final[all_features]
y = df_final['Approved']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Convert to numpy arrays
X_train_np = X_train.values
X_test_np = X_test.values

# Ensure all values are numeric (convert any remaining object types)
X_train_np = X_train_np.astype(np.float32)
X_test_np = X_test_np.astype(np.float32)

print("\n✅ Features already normalized as 'level of truth' values [0,1]")

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train_np, dtype=torch.float32).to(device)
Y_train_tensor = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test_np, dtype=torch.float32).to(device)
Y_test_tensor = torch.tensor(y_test.values.reshape(-1, 1), dtype=torch.float32).to(device)

print(f"\n📊 Training set: {len(X_train):,} samples")
print(f"📊 Test set: {len(X_test):,} samples")
print(f"📊 Feature dimension: {X_train.shape[1]}")

print("\n🧠 Training BACON model...")
print("⏳ This may take a few minutes...\n")

# Initialize and train the BACON model with transformation layer
bacon = baconNet(
    input_size=X_train.shape[1],
    freeze_loss_threshold=0.15,
    weight_mode='fixed',
    aggregator='lsp.half_weight',
    use_transformation_layer=True,  # Enable transformation layer
    transformation_temperature=1.0,  # Temperature for softmax selection
    transformation_use_gumbel=False,  # Use standard softmax (not Gumbel)
    max_permutations=40,
)

(best_model, best_accuracy) = bacon.find_best_model(
    X_train_tensor, Y_train_tensor, X_test_tensor, Y_test_tensor,
    attempts=30,
    acceptance_threshold=0.15,  # Lower threshold due to complexity
    max_epochs=15000,
    save_path="./assembler-credit-risk.pth"
)

print(f"\n🏆 Best accuracy: {best_accuracy * 100:.2f}%\n")

# Visualize the BACON model's tree structure
print_tree_structure(bacon.assembler, all_features)
visualize_tree_structure(bacon.assembler, all_features)

# Analyze transformation layer if enabled
if bacon.assembler.transformation_layer is not None:
    print("\n🔄 Transformation Layer Analysis\n")
    print("=" * 70)
    
    trans_summary = bacon.assembler.transformation_layer.get_transformation_summary()
    selected_transforms = bacon.assembler.transformation_layer.get_selected_transformations()
    
    # Count transformations
    identity_count = (selected_transforms == 0).sum().item()
    negation_count = (selected_transforms == 1).sum().item()
    
    print(f"\n📊 Transformation Statistics:")
    print(f"   Identity (x):        {identity_count} features ({identity_count/len(all_features)*100:.1f}%)")
    print(f"   Negation (1-x):      {negation_count} features ({negation_count/len(all_features)*100:.1f}%)")
    
    print(f"\n🎯 Top Features Using Negation (1-x):")
    negated_features = []
    for feat_idx, summary in trans_summary.items():
        if summary['transformation'] == 'negation':
            negated_features.append((all_features[feat_idx], summary['probability']))
    
    negated_features.sort(key=lambda x: x[1], reverse=True)
    for feature_name, prob in negated_features[:10]:  # Show top 10
        print(f"   {feature_name:30s} (confidence: {prob*100:.1f}%)")
    
    print(f"\n🎯 Top Features Using Identity (x):")
    identity_features = []
    for feat_idx, summary in trans_summary.items():
        if summary['transformation'] == 'identity':
            identity_features.append((all_features[feat_idx], summary['probability']))
    
    identity_features.sort(key=lambda x: x[1], reverse=True)
    for feature_name, prob in identity_features[:10]:  # Show top 10
        print(f"   {feature_name:30s} (confidence: {prob*100:.1f}%)")
    
    print("\n" + "=" * 70)
    
    # Visualize transformation probabilities
    probs = bacon.assembler.transformation_layer.get_transformation_probabilities()
    negation_probs = probs[:, 1].cpu().detach().numpy()  # Probabilities for negation
    
    plt.figure(figsize=(14, 6))
    
    # Plot negation probabilities
    plt.subplot(1, 2, 1)
    plt.bar(range(len(all_features)), negation_probs)
    plt.axhline(y=0.5, color='r', linestyle='--', alpha=0.5, label='Threshold (0.5)')
    plt.title('Negation Probability per Feature')
    plt.xlabel('Feature Index')
    plt.ylabel('P(1-x)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot selected transformations
    plt.subplot(1, 2, 2)
    selected = selected_transforms.cpu().numpy()
    colors = ['blue' if s == 0 else 'orange' for s in selected]
    plt.bar(range(len(all_features)), [1]*len(all_features), color=colors, alpha=0.7)
    plt.title('Selected Transformations')
    plt.xlabel('Feature Index')
    plt.ylabel('Type')
    plt.yticks([0, 1], ['', ''])
    plt.legend([plt.Rectangle((0,0),1,1, color='blue'), 
                plt.Rectangle((0,0),1,1, color='orange')],
               ['Identity (x)', 'Negation (1-x)'])
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('transformation_analysis.png', dpi=300, bbox_inches='tight')
    print("\n💾 Transformation analysis saved to 'transformation_analysis.png'")
    plt.show()



# Perform feature pruning analysis
print("\n🔍 Analyzing feature importance through pruning...\n")
accuracies = []

for i in range(1, min(X_test.shape[1], 20)):  # Limit to first 20 for readability
    func_eval = bacon.prune_features(i)
    kept_indices = bacon.assembler.locked_perm[i:].tolist()
    X_test_pruned = X_test_tensor[:, kept_indices]
    with torch.no_grad():
        pruned_output = func_eval(X_test_pruned)
        pruned_accuracy = (pruned_output.round() == Y_test_tensor).float().mean().item()
        accuracies.append(pruned_accuracy)
        print(f"✅ Accuracy after pruning {i} feature(s): {pruned_accuracy * 100:.2f}%")

# Plot accuracy vs. pruned features
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(accuracies) + 1), [a * 100 for a in accuracies], marker='o')
plt.title("Credit Risk Accuracy vs. Number of Features Pruned")
plt.xlabel("Number of Features Pruned from Left")
plt.ylabel("Accuracy (%)")
plt.grid(True)
plt.tight_layout()
plt.savefig('feature_pruning_analysis.png', dpi=300, bbox_inches='tight')
print("\n💾 Feature pruning analysis saved to 'feature_pruning_analysis.png'")
plt.show()

print("\n✅ Analysis complete!")
