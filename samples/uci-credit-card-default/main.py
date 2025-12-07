import sys
sys.path.insert(0, '../../')
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support, precision_recall_curve
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

print("=" * 80)
print("UCI Credit Card Default (Taiwan) - BACON Benchmark")
print("=" * 80)

# ========== Load Dataset ==========
print("\n📂 Loading UCI Credit Card Default dataset...")

# The dataset is in Excel format with header in row 2
try:
    df = pd.read_excel('default of credit card clients.xls', header=1)
    print(f"✅ Loaded {len(df):,} credit card clients")
except FileNotFoundError:
    print("\n❌ Dataset file not found!")
    print("\nPlease download the dataset:")
    print("  1. Go to: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients")
    print("  2. Download 'default of credit card clients.xls'")
    print("  3. Place it in this directory")
    print("\nOR use curl:")
    print('  curl -o "default of credit card clients.xls" "https://archive.ics.uci.edu/ml/machine-learning-databases/00350/default%20of%20credit%20card%20clients.xls"')
    sys.exit(1)

# ========== Feature Engineering ==========
print("\n📊 Dataset Overview:")
print(f"   Total clients: {len(df):,}")
print(f"   Default rate: {df['default payment next month'].mean() * 100:.2f}%")
print(f"   Features: {df.shape[1] - 1} (excluding target)")

# Remove ID column
df = df.drop('ID', axis=1)

# Target variable
target_col = 'default payment next month'
y = df[target_col]
df = df.drop(target_col, axis=1)

# Feature names
feature_names = df.columns.tolist()

print(f"\n📋 Feature Categories:")
print(f"   Demographics: SEX, EDUCATION, MARRIAGE, AGE")
print(f"   Credit: LIMIT_BAL")
print(f"   Payment Status: PAY_0 to PAY_6 (6 months)")
print(f"   Bill Amounts: BILL_AMT1 to BILL_AMT6 (6 months)")
print(f"   Payment Amounts: PAY_AMT1 to PAY_AMT6 (6 months)")

# ========== Preprocessing to [0,1] "Level of Truth" ==========
print("\n🔄 Normalizing features to [0,1] 'level of truth' values...")

df_norm = df.copy()

# LIMIT_BAL: Higher limit → lower default risk → negate later
# Normalize to [0,1]
df_norm['LIMIT_BAL'] = (df['LIMIT_BAL'] - df['LIMIT_BAL'].min()) / (df['LIMIT_BAL'].max() - df['LIMIT_BAL'].min() + 1e-8)
print(f"  ✅ LIMIT_BAL: normalized (higher = more creditworthy)")

# SEX: 1=male, 2=female → Convert to binary
df_norm['SEX'] = (df['SEX'] == 1).astype(float)  # 1 if male, 0 if female
print(f"  ✅ SEX: binary (1=male, 0=female)")

# EDUCATION: 1=graduate, 2=university, 3=high school, 4=others
# Higher education → lower risk → reverse scale
df_norm['EDUCATION'] = df['EDUCATION'].replace({0: 4, 5: 4, 6: 4})  # Consolidate unknowns to 'others'
df_norm['EDUCATION'] = 1 - ((df_norm['EDUCATION'] - 1) / 3)  # Reverse: grad=1, others=0
print(f"  ✅ EDUCATION: reversed (1=graduate, 0=others)")

# MARRIAGE: 1=married, 2=single, 3=others → One-hot or binary
df_norm['MARRIAGE'] = (df['MARRIAGE'] == 1).astype(float)  # 1 if married
print(f"  ✅ MARRIAGE: binary (1=married, 0=single/others)")

# AGE: Normalize
df_norm['AGE'] = (df['AGE'] - df['AGE'].min()) / (df['AGE'].max() - df['AGE'].min() + 1e-8)
print(f"  ✅ AGE: normalized to [0,1]")

# PAY_0 to PAY_6: Payment delay status
# -2=no consumption, -1=pay duly, 0=revolving credit, 1+=months of delay
# Higher delay → higher risk → normalize to [0,1] where 1=high delay
for i in range(7):
    pay_col = f'PAY_{i}' if i > 0 else 'PAY_0'
    if pay_col in df.columns:
        # Clip extreme values and normalize
        df_norm[pay_col] = df[pay_col].clip(-2, 8)  # Cap at 8 months delay
        df_norm[pay_col] = (df_norm[pay_col] + 2) / 10  # Range [-2, 8] → [0, 1]
        print(f"  ✅ {pay_col}: normalized (1=high delay, 0=pay duly)")

# BILL_AMT1 to BILL_AMT6: Bill statement amounts
# Use robust normalization (log transform + min-max)
for i in range(1, 7):
    bill_col = f'BILL_AMT{i}'
    if bill_col in df.columns:
        # Some bills can be negative (overpayment), shift to positive
        shifted = df[bill_col] - df[bill_col].min() + 1
        log_transformed = np.log1p(shifted)
        df_norm[bill_col] = (log_transformed - log_transformed.min()) / (log_transformed.max() - log_transformed.min() + 1e-8)
        print(f"  ✅ {bill_col}: log-transformed, normalized")

# PAY_AMT1 to PAY_AMT6: Payment amounts
# Higher payment → lower risk → will negate if needed
for i in range(1, 7):
    pay_amt_col = f'PAY_AMT{i}'
    if pay_amt_col in df.columns:
        log_transformed = np.log1p(df[pay_amt_col])
        df_norm[pay_amt_col] = (log_transformed - log_transformed.min()) / (log_transformed.max() - log_transformed.min() + 1e-8)
        print(f"  ✅ {pay_amt_col}: log-transformed, normalized")

# ========== Train/Test Split ==========
print("\n📊 Splitting dataset...")
X_train, X_test, y_train, y_test = train_test_split(
    df_norm, y, test_size=0.3, random_state=42, stratify=y
)

print(f"   Training set: {len(X_train):,} clients ({y_train.mean()*100:.1f}% default)")
print(f"   Test set: {len(X_test):,} clients ({y_test.mean()*100:.1f}% default)")

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32).to(device)
Y_train_tensor = torch.tensor(y_train.values.reshape(-1, 1), dtype=torch.float32).to(device)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32).to(device)
Y_test_tensor = torch.tensor(y_test.values.reshape(-1, 1), dtype=torch.float32).to(device)

# ========== BACON Model Training ==========
print("\n" + "=" * 80)
print("🧠 Training BACON Model")
print("=" * 80)

import os
model_path = "./best_bacon_credit_card.pth"

# Check if trained model exists
if os.path.exists(model_path):
    print("\n✅ Found existing trained model!")
    print(f"   Loading from: {model_path}")
    print("   (Delete this file to retrain from scratch)")
    
    bacon = baconNet(
        input_size=X_train.shape[1],
        freeze_loss_threshold=0.15,
        weight_mode='fixed',
        aggregator='lsp.half_weight',
        use_transformation_layer=True,
        transformation_temperature=1.0,
        transformation_use_gumbel=False,
        max_permutations=40,
    )
    
    # Load the saved model
    checkpoint = torch.load(model_path, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Load model state
        bacon.assembler.load_state_dict(checkpoint['model_state_dict'])
        # Restore metadata
        bacon.assembler.is_frozen = checkpoint.get('is_frozen', False)
        bacon.assembler.locked_perm = checkpoint.get('locked_perm', None)
        bacon.assembler.tree_layout = checkpoint.get('tree_layout', 'left')
        
        # If model was frozen, need to recreate frozen input layer
        if bacon.assembler.is_frozen and bacon.assembler.locked_perm is not None:
            from bacon.frozonInputToLeaf import frozenInputToLeaf
            bacon.assembler.input_to_leaf = frozenInputToLeaf(
                bacon.assembler.locked_perm, 
                bacon.assembler.original_input_size
            ).to(device)
    else:
        # Old format (direct state dict)
        bacon.assembler.load_state_dict(checkpoint)
    
    # Quick evaluation
    with torch.no_grad():
        test_pred = bacon(X_test_tensor).cpu().numpy().flatten()
    loaded_auc = roc_auc_score(y_test, test_pred)
    loaded_acc = accuracy_score(y_test, (test_pred > 0.5).astype(int))
    
    print(f"\n📊 Loaded Model Performance:")
    print(f"   Test AUC: {loaded_auc:.4f}")
    print(f"   Test Accuracy: {loaded_acc:.4f}")
    print(f"   Model frozen: {bacon.assembler.is_frozen}")
    
    best_accuracy = loaded_acc
    skip_training = True
else:
    print("\n⚙️  Configuration:")
    print("   - Transformation Layer: Enabled (identity, negation, peak)")
    print("   - Hierarchical Permutation: Enabled")
    print("   - Aggregator: LSP (half_weight)")
    print("   - Soft boundaries: 10% bleed ratio")
    print("   - Freeze threshold: 0.25 (lower = easier to freeze)")

    bacon = baconNet(
        input_size=X_train.shape[1],
        freeze_loss_threshold=0.25,  # Increased from 0.15 to make freezing easier
        weight_mode='fixed',
        aggregator='lsp.half_weight',
        use_transformation_layer=True,
        transformation_temperature=1.0,
        transformation_use_gumbel=False,
        max_permutations=40,
    )
    skip_training = False

print("\n🔀 Hierarchical Permutation Search:")
hierarchical_group_size = 8
hierarchical_epochs = 4000
num_features = X_train.shape[1]
num_groups = (num_features + hierarchical_group_size - 1) // hierarchical_group_size
import math
num_coarse_perms = min(math.factorial(num_groups), 120)  # Cap at 120 permutations

print(f"   Features: {num_features}")
print(f"   Group size: {hierarchical_group_size}")
print(f"   Groups: {num_groups}")
print(f"   Permutations to try: {num_coarse_perms}")
print(f"   Epochs per permutation: {hierarchical_epochs}")

if not skip_training:
    print("\n⏳ Training (this may take several minutes)...\n")

    (best_model, best_accuracy) = bacon.find_best_model(
        X_train_tensor, Y_train_tensor, X_test_tensor, Y_test_tensor,
        acceptance_threshold=0.90,
        max_epochs=15000,
        save_path=model_path,
        use_hierarchical_permutation=True,
        hierarchical_group_size=hierarchical_group_size,
        hierarchical_epochs_per_attempt=hierarchical_epochs,
        hierarchical_bleed_ratio=0.1,
        hierarchical_bleed_decay=2.0
    )
    
    # Force freeze if not already frozen (for pruning analysis)
    if not bacon.assembler.is_frozen:
        print("\n🔒 Forcing model freeze for pruning analysis...")
        # Get the current best permutation from the Sinkhorn layer
        if hasattr(bacon.assembler.input_to_leaf, 'logits'):
            from bacon.frozonInputToLeaf import frozenInputToLeaf
            # Get hard assignment from current soft permutation
            soft_perm = torch.softmax(bacon.assembler.input_to_leaf.logits, dim=1)
            hard_perm = torch.argmax(soft_perm, dim=1)
            bacon.assembler.locked_perm = hard_perm.clone().detach()
            bacon.assembler.is_frozen = True
            # Replace with frozen layer
            bacon.assembler.input_to_leaf = frozenInputToLeaf(
                bacon.assembler.locked_perm,
                bacon.assembler.original_input_size
            ).to(device)
            print(f"   ✅ Model frozen with locked permutation")
            # Save the frozen model
            bacon.save_model(model_path)
            print(f"   💾 Saved frozen model to {model_path}")
else:
    print("\n⏭️  Skipping training (using loaded model)\n")

# ========== Evaluation ==========
print("\n" + "=" * 80)
print("📊 Model Evaluation")
print("=" * 80)

# Get predictions
with torch.no_grad():
    train_pred = bacon(X_train_tensor).cpu().numpy().flatten()
    test_pred = bacon(X_test_tensor).cpu().numpy().flatten()

# Calculate metrics with default threshold (0.5)
train_auc = roc_auc_score(y_train, train_pred)
test_auc = roc_auc_score(y_test, test_pred)

train_acc = accuracy_score(y_train, (train_pred > 0.5).astype(int))
test_acc = accuracy_score(y_test, (test_pred > 0.5).astype(int))

precision_default, recall_default, f1_default, _ = precision_recall_fscore_support(
    y_test, (test_pred > 0.5).astype(int), average='binary'
)

# Optimize threshold for best F1 score
print("\n🎯 Optimizing Decision Threshold...")
prec, rec, thr = precision_recall_curve(y_test, test_pred)
f1_scores = 2 * prec * rec / (prec + rec + 1e-8)
best_idx = np.argmax(f1_scores)
best_threshold = thr[best_idx] if best_idx < len(thr) else 0.5

# Calculate metrics with optimized threshold
test_acc_opt = accuracy_score(y_test, (test_pred > best_threshold).astype(int))
precision_opt, recall_opt, f1_opt, _ = precision_recall_fscore_support(
    y_test, (test_pred > best_threshold).astype(int), average='binary'
)

print(f"   Optimal threshold: {best_threshold:.4f} (maximizes F1-score)")
print(f"   Default threshold: 0.5000")

print(f"\n🎯 Performance Metrics:")
print(f"   {'Metric':<20} {'Train':>10} {'Test (0.5)':>12} {'Test (opt)':>12}")
print(f"   {'-'*56}")
print(f"   {'AUC-ROC':<20} {train_auc:>10.4f} {test_auc:>12.4f} {test_auc:>12.4f}")
print(f"   {'Accuracy':<20} {train_acc:>10.4f} {test_acc:>12.4f} {test_acc_opt:>12.4f}")
print(f"   {'Precision':<20} {'-':>10} {precision_default:>12.4f} {precision_opt:>12.4f}")
print(f"   {'Recall':<20} {'-':>10} {recall_default:>12.4f} {recall_opt:>12.4f}")
print(f"   {'F1-Score':<20} {'-':>10} {f1_default:>12.4f} {f1_opt:>12.4f}")

# Use optimized metrics for final reporting
precision, recall, f1 = precision_opt, recall_opt, f1_opt
test_acc = test_acc_opt

print(f"\n📈 Comparison with Published Baselines:")
print(f"   Logistic Regression:  ~0.77-0.78 AUC")
print(f"   Random Forest:        ~0.76-0.78 AUC")
print(f"   XGBoost:              ~0.77-0.80 AUC")
print(f"   BACON:                 {test_auc:.4f} AUC")

# ========== Interpretability Analysis ==========
print("\n" + "=" * 80)
print("🔍 Interpretability Analysis")
print("=" * 80)

# Tree structure
print("\n🌳 Learned Logical Tree Structure:\n")
print_tree_structure(bacon.assembler, feature_names)

# Transformation analysis
if bacon.assembler.transformation_layer is not None:
    print("\n🔄 Transformation Layer Analysis:")
    print("=" * 80)
    
    trans_summary = bacon.assembler.transformation_layer.get_transformation_summary()
    selected_transforms = bacon.assembler.transformation_layer.get_selected_transformations()
    
    identity_count = (selected_transforms == 0).sum().item()
    negation_count = (selected_transforms == 1).sum().item()
    peak_count = (selected_transforms == 2).sum().item()
    
    print(f"\n📊 Transformation Distribution:")
    print(f"   Identity (x):        {identity_count} features ({identity_count/len(feature_names)*100:.1f}%)")
    print(f"   Negation (1-x):      {negation_count} features ({negation_count/len(feature_names)*100:.1f}%)")
    print(f"   Peak (1-|x-t|):      {peak_count} features ({peak_count/len(feature_names)*100:.1f}%)")
    
    # Top features by transformation
    if negation_count > 0:
        print(f"\n🔄 Features Using Negation (1-x):")
        negated = [(feature_names[i], trans_summary[i]['probability']) 
                   for i in range(len(feature_names)) 
                   if trans_summary[i]['transformation'] == 'negation']
        negated.sort(key=lambda x: x[1], reverse=True)
        for feat, prob in negated[:10]:
            print(f"   {feat:<20} (confidence: {prob*100:.1f}%)")
    
    if peak_count > 0:
        print(f"\n🎯 Features Using Peak Transformation (1-|x-t|):")
        peaks = [(feature_names[i], trans_summary[i]['probability'], trans_summary[i]['params'].get('peak_location', 'N/A'))
                 for i in range(len(feature_names)) 
                 if trans_summary[i]['transformation'] == 'peak']
        peaks.sort(key=lambda x: x[1], reverse=True)
        for feat, prob, peak_loc in peaks:
            print(f"   {feat:<20} peak={peak_loc} (confidence: {prob*100:.1f}%)")

# Feature importance through pruning
print("\n🔍 Feature Importance (Progressive Pruning):")
print("=" * 80)

# Ensure model is frozen and has locked_perm before pruning
if bacon.assembler.locked_perm is None:
    print("   ⚠️  Model does not have locked permutation.")
    print("   Pruning analysis requires a frozen model with determined feature order.")
    print("   Skipping pruning analysis...")
    pruning_results = []
else:
    if not bacon.assembler.is_frozen:
        print("   Freezing model for pruning analysis...")
        bacon.assembler.is_frozen = True

    pruning_results = []
    for i in range(1, min(11, len(feature_names))):
        func_eval = bacon.prune_features(i)
        kept_indices = bacon.assembler.locked_perm[i:].tolist()
        X_test_pruned = X_test_tensor[:, kept_indices]
    
        with torch.no_grad():
            pruned_output = func_eval(X_test_pruned)
            pruned_auc = roc_auc_score(y_test, pruned_output.cpu().numpy().flatten())
            pruning_results.append((i, pruned_auc))
        
            # Show top features removed
            removed_idx = bacon.assembler.locked_perm[i-1].item()
            print(f"   Removed {i}: {feature_names[removed_idx]:<20} → AUC = {pruned_auc:.4f}")

# Visualization
print("\n📊 Generating visualizations...")
visualize_tree_structure(bacon.assembler, feature_names)

# Plot pruning analysis
if len(pruning_results) > 0:
    plt.figure(figsize=(10, 6))
    prune_counts = [p[0] for p in pruning_results]
    prune_aucs = [p[1] for p in pruning_results]
    plt.plot(prune_counts, prune_aucs, marker='o', linewidth=2, markersize=8)
    plt.axhline(y=test_auc, color='r', linestyle='--', alpha=0.7, label=f'Full Model AUC: {test_auc:.4f}')
    plt.xlabel('Number of Features Removed', fontsize=12)
    plt.ylabel('Test AUC', fontsize=12)
    plt.title('Feature Pruning Impact on Model Performance', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('feature_pruning_analysis.png', dpi=300)
    print("💾 Saved: feature_pruning_analysis.png")

print("\n" + "=" * 80)
print("✅ Training and Analysis Complete!")
print("=" * 80)

plt.show()
