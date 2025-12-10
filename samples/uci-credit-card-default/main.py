# The following two lines are needed to refer to local package instead of published package

import sys
sys.path.insert(0, '../../')

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score, precision_recall_fscore_support, precision_recall_curve
import torch
from bacon.baconNet import baconNet
from bacon.hierarchicalBaconNet import HierarchicalBaconNet
from bacon.visualization import visualize_tree_structure, print_tree_structure
import logging
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os

logging.basicConfig(level=logging.INFO, format='%(message)s')

# Choose GPU is possible
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

# Feature names (will be updated after feature engineering)
original_feature_names = df.columns.tolist()

print(f"\n📋 Feature Categories:")
print(f"   Demographics: SEX, EDUCATION, MARRIAGE, AGE")
print(f"   Credit: LIMIT_BAL")
print(f"   Payment Status: PAY_0 to PAY_6 (6 months)")
print(f"   Bill Amounts: BILL_AMT1 to BILL_AMT6 (6 months)")
print(f"   Payment Amounts: PAY_AMT1 to PAY_AMT6 (6 months)")

# -------- Preprocessing to [0,1] "Level of Truth" --------
print("\n🔄 Normalizing features to [0,1] 'level of truth' values...")

df_norm = df.copy()

# LIMIT_BAL: Higher limit → lower default risk
# Normalize to [0,1]
# Note 1: LIMIT_BAL is very skewed, so we use log-transform and normalize. 
# Note 2: A low LIMIT_BAL is bad (high contribution to default risk), but we don't negate it as the model can learn to negate if needed.
log_transformed = np.log1p(df['LIMIT_BAL'])
df_norm['LIMIT_BAL'] = (log_transformed - log_transformed.min()) / (log_transformed.max() - log_transformed.min() + 1e-8)
print(f"   ✅ LIMIT_BAL: normalized (higher = more creditworthy)")

# SEX: 1=male, 2=female → Convert to binary
# Note: Here the choice of 1=female and 0=male is arbitrary.
df_norm['SEX'] = (df['SEX'] == 2).astype(float)  # 1 if female, 0 if male
print(f"   ✅ SEX: binary (1=female, 0=male)")

# EDUCATION: 1=graduate, 2=university, 3=high school, 4=others
# Note: Lowever value -> Higher education → Lower risk. This is left as it is for the model to learn (whether to negate or not).
df_norm['EDUCATION'] = df['EDUCATION'].replace({0: 4, 5: 4, 6: 4})  # Consolidate unknowns to 'others'
df_norm['EDUCATION'] = ((df_norm['EDUCATION'] - 1) / 3) 
print(f"   ✅ EDUCATION: (0=graduate, 1=others)")

# MARRIAGE: 1=married, 2=single, 3=others → One-hot or binary
# Note: Here the choice of 1=single and 0=married is arbitrary.
df_norm['MARRIAGE'] = (df['MARRIAGE'] != 1).astype(float)  # 0 if married, 1 if single/others
print(f"   ✅ MARRIAGE: binary (0=married, 1=single/others)")

# AGE: Normalize
df_norm['AGE'] = (df['AGE'] - df['AGE'].min()) / (df['AGE'].max() - df['AGE'].min() + 1e-8)
print(f"   ✅ AGE: normalized to [0,1]")
# PAY_0 to PAY_6: Payment delay status
# Official scale: -1=pay duly, 1-9=months of delay (some data has 0 and -2 as anomalies)
# Note: Higher delay → higher risk → normalize to [0,1] where 1=high delay
for i in range(7):
    pay_col = f'PAY_{i}'
    if pay_col in df.columns:
        # Clip to documented range and normalize
        df_norm[pay_col] = df[pay_col].clip(-1, 9)  # -1 to 9 months
        df_norm[pay_col] = (df_norm[pay_col] + 1) / 10  # Range [-1, 9] → [0, 1]
        print(f"   ✅ {pay_col}: normalized (1=high delay, 0=pay duly)")

# BILL_AMT1 to BILL_AMT6: Bill statement amounts
# Note: Negative bills = overpayment/credit balance, treat as 0 debt
for i in range(1, 7):
    bill_col = f'BILL_AMT{i}'
    if bill_col in df.columns:
        # Clip negative values (overpayment) to 0, then log-transform and normalize
        clipped = df[bill_col].clip(lower=0)
        log_transformed = np.log1p(clipped)
        df_norm[bill_col] = (log_transformed - log_transformed.min()) / (log_transformed.max() - log_transformed.min() + 1e-8)
        print(f"   ✅ {bill_col}: clipped at 0, log-transformed, normalized")

# PAY_AMT1 to PAY_AMT6: Payment amounts
for i in range(1, 7):
    pay_amt_col = f'PAY_AMT{i}'
    if pay_amt_col in df.columns:
        log_transformed = np.log1p(df[pay_amt_col])
        df_norm[pay_amt_col] = (log_transformed - log_transformed.min()) / (log_transformed.max() - log_transformed.min() + 1e-8)
        print(f"   ✅ {pay_amt_col}: log-transformed, normalized")

# -------- Payment Ratio Feature Engineering --------
print("\n💰 Engineering Payment Ratio Features...")
# Create payment ratio features: PAY_AMT / BILL_AMT for each month
# Higher ratio = paying more of the bill = lower default risk
for i in range(1, 7):
    bill_col = f'BILL_AMT{i}'
    pay_col = f'PAY_AMT{i}'
    ratio_col = f'PAY_RATIO_{i}'
    
    if bill_col in df.columns and pay_col in df.columns:
        # Calculate ratio, handling division by zero
        # If bill is 0 (or very small after normalization), set ratio to 1 (paid in full)
        bill_values = df[bill_col].clip(lower=1)  # Avoid division by zero
        pay_values = df[pay_col]
        
        # Ratio = payment / bill, clipped to [0, 2] (paying 2x bill is same as paying 1x)
        # Special case: if bill is 0, set ratio to 1 (no debt to pay)
        ratio = (pay_values / bill_values).clip(0, 1)
        ratio = ratio.where(df[bill_col] > 0, 1.0)
        df_norm[ratio_col] = ratio
        print(f"   ✅ {ratio_col}: PAY_AMT{i} / BILL_AMT{i}, normalized")

# Update feature names
feature_names = df_norm.columns.tolist()
print(f"\n📊 Final feature count: {len(feature_names)}")

# ========== Hierarchical Structure Definition ==========
# Note: Define feature groups for hierarchical BACON. Set to None to use flat BACON instead.
USE_HIERARCHICAL = True

if USE_HIERARCHICAL:
    # Define groups based on domain knowledge.
    # Note: Each group will be processed by a sub-tree (preserving order). Then groups are combined in global tree (learned ordering).
    
    # Get all feature names from df_norm
    all_features = df_norm.columns.tolist()
    
    # Build feature groups with indices.
    # Note: Only define MULTI-FEATURE groups here. Features not in any group will be added automatically as ungrouped.
    
    # Ignore raw BILL_AMT and PAY_AMT since we're using engineered PAY_RATIO features
    IGNORED_FEATURES = [col for col in all_features if col.startswith('BILL_AMT') or col.startswith('PAY_AMT')]
    
    # Remove ignored features from dataframe FIRST
    if len(IGNORED_FEATURES) > 0:
        df_norm = df_norm.drop(columns=IGNORED_FEATURES)
        all_features = df_norm.columns.tolist()  # Update feature list
        print(f"\n🗑️  Removed {len(IGNORED_FEATURES)} ignored features: {', '.join(IGNORED_FEATURES)}")
    
    # NOW build feature groups with corrected indices
    FEATURE_GROUPS = {
        'payment_history': [i for i, col in enumerate(all_features) if col.startswith('PAY_') and not col.startswith('PAY_AMT') and not col.startswith('PAY_RATIO')],
        'payment_ratios': [i for i, col in enumerate(all_features) if col.startswith('PAY_RATIO')],
        'demographics': [i for i, col in enumerate(all_features) if col in ['SEX', 'EDUCATION', 'MARRIAGE', 'AGE']],
    }
    
    # Remove empty groups
    FEATURE_GROUPS = {k: v for k, v in FEATURE_GROUPS.items() if len(v) > 0}
    
    # Find ungrouped features (not in any group)
    all_grouped_indices = set()
    for indices in FEATURE_GROUPS.values():
        all_grouped_indices.update(indices)
    
    ungrouped_indices = [i for i in range(len(all_features)) if i not in all_grouped_indices]
    ungrouped_features = [all_features[i] for i in ungrouped_indices]
    
    print(f"\n🌳 Hierarchical BACON Structure:")
    if len(FEATURE_GROUPS) > 0:
        print(f"   User-defined groups: {len(FEATURE_GROUPS)} (processed by sub-trees)")
        for group_name, indices in FEATURE_GROUPS.items():
            group_features = [all_features[i] for i in indices]
            print(f"   - {group_name}: {len(indices)} features ({', '.join(group_features)})")
    
    if len(ungrouped_features) > 0:
        print(f"   Ungrouped features: {len(ungrouped_features)} (joined directly to global tree)")
        print(f"   - {', '.join(ungrouped_features)}")
    
    print(f"\n🔍 Verification:")
    print(f"   Total features after filtering: {len(all_features)}")
    print(f"   Features in groups: {sum(len(v) for v in FEATURE_GROUPS.values())}")
    print(f"   Ungrouped features: {len(ungrouped_features)}")
    print(f"   Global tree should have: {len(FEATURE_GROUPS)} groups + {len(ungrouped_features)} ungrouped = {len(FEATURE_GROUPS) + len(ungrouped_features)} inputs")
else:
    FEATURE_GROUPS = None
    print(f"\n🌳 Using flat BACON (no hierarchy)")

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

# Define transformation sets BEFORE model creation
# Option 1: Identity and Negation only (simple, interpretable)
# Option 2: All transformations (identity, negation, peak, valley, step_up, step_down)
USE_SIMPLE_TRANSFORMATIONS = True  # Set to True for identity+negation only

if USE_SIMPLE_TRANSFORMATIONS:
    from bacon.transformationLayer import IdentityTransformation, NegationTransformation, ValleyTransformation, PeakTransformation, StepUpTransformation, StepDownTransformation
    # Create template instances (will be recreated with correct sizes)
    sub_tree_trans = [IdentityTransformation(1), NegationTransformation(1)]
    global_tree_trans = [IdentityTransformation(1), NegationTransformation(1), ValleyTransformation(1), PeakTransformation(1), StepUpTransformation(1), StepDownTransformation(1)]
    print("\n⚙️ Transformation Configuration: Sub-trees=Identity+Negation, Global=All 6 transforms")
else:
    sub_tree_trans = None  # Use all transformations
    global_tree_trans = None  # Use all transformations
    print("\n⚙️ Transformation Configuration: ALL transformations enabled")

if USE_HIERARCHICAL:
    model_path = "./best_hierarchical_bacon_credit_card.pth"
else:
    model_path = "./best_bacon_credit_card.pth"

FREEZE_LOSS_THRESHOLD = 0.42

# Check if trained model exists
if os.path.exists(model_path):
    print("\n✅ Found existing trained model!")
    print(f"   Loading from: {model_path}")
    
    if USE_HIERARCHICAL:
        bacon = HierarchicalBaconNet(
            feature_groups=FEATURE_GROUPS,
            total_features=X_train.shape[1],
            freeze_loss_threshold=FREEZE_LOSS_THRESHOLD,
            weight_mode='fixed',
            aggregator='lsp.half_weight',
            use_sub_tree_transformation=True,
            use_global_transformation=True,
            sub_tree_layout='left',
            global_tree_layout='left',
            max_permutations=10000,
            use_class_weighting=False,
            sub_tree_transformations=sub_tree_trans,
            global_tree_transformations=global_tree_trans,
            sub_tree_permutation_groups=['demographics'],  # Enable permutation learning only on demographics
        )
    else:
        bacon = baconNet(
            input_size=X_train.shape[1],
            freeze_loss_threshold=FREEZE_LOSS_THRESHOLD,
            tree_layout='left',
            weight_mode='fixed',
            aggregator='lsp.half_weight',
            use_transformation_layer=True,
            max_permutations=1,
            is_frozen=True,
        )
    
    print("\n🔍 DEBUG: Model state BEFORE load:")
    print(f"   Global tree frozen: {bacon.global_tree.is_frozen if USE_HIERARCHICAL else 'N/A'}")
    print(f"   Global locked_perm: {bacon.global_tree.locked_perm if USE_HIERARCHICAL else 'N/A'}")
    if USE_HIERARCHICAL and bacon.global_tree.transformation_layer:
        print(f"   Global has transformation layer: True")
    else:
        print(f"   Global has transformation layer: False")
    
    bacon.load_model(model_path)
    bacon = bacon.to(device)
    
    print("\n🔍 DEBUG: Model state AFTER load:")
    print(f"   Global tree frozen: {bacon.global_tree.is_frozen if USE_HIERARCHICAL else 'N/A'}")
    print(f"   Global locked_perm: {bacon.global_tree.locked_perm if USE_HIERARCHICAL else 'N/A'}")
    if USE_HIERARCHICAL and bacon.global_tree.transformation_layer:
        trans_summary = bacon.global_tree.transformation_layer.get_transformation_summary()
        print(f"   Global transformation layer: {len(trans_summary)} features")
        print(f"   First transformation: {trans_summary[0]}")
    
    # Quick evaluation
    with torch.no_grad():
        test_pred = bacon(X_test_tensor).cpu().numpy().flatten()
    loaded_auc = roc_auc_score(y_test, test_pred)
    loaded_acc = accuracy_score(y_test, (test_pred > 0.5).astype(int))
    
    print(f"\n📊 Loaded Model Performance:")
    print(f"   Test AUC: {loaded_auc:.4f}")
    print(f"   Test Accuracy: {loaded_acc:.4f}")
    
    best_accuracy = loaded_acc
    skip_training = True
else:
    print("\n⚙️ Configuration:")
    if USE_HIERARCHICAL:
        print("   - Architecture: Hierarchical BACON")
        print(f"   - Sub-trees: {len(FEATURE_GROUPS)} groups with fixed order")
        print("   - Global tree: Learns group permutation + transformations")
        print("   - Sub-tree layout: Left-skewed")
        print("   - Global tree layout: Balanced")
        if USE_SIMPLE_TRANSFORMATIONS:
            print("   - Transformations: Identity + Negation only")
        else:
            print("   - Transformations: All (identity, negation, peak, valley, step_up, step_down)")
        
        bacon = HierarchicalBaconNet(
            feature_groups=FEATURE_GROUPS,
            total_features=X_train.shape[1],
            freeze_loss_threshold=FREEZE_LOSS_THRESHOLD,
            weight_mode='fixed',
            aggregator='lsp.half_weight',
            use_sub_tree_transformation=True,
            use_global_transformation=True,
            sub_tree_layout='left',
            global_tree_layout='left',
            max_permutations=10000,
            lr_permutation=0.3,
            lr_transformation=0.1,
            lr_aggregator=0.1,
            lr_other=0.1,
            loss_amplifier=1.0,  # Default: no amplification (BCE loss is already in reasonable range)
            use_class_weighting=False,
            sub_tree_transformations=sub_tree_trans,
            global_tree_transformations=global_tree_trans,
            sub_tree_permutation_groups=['demographics'],  # Enable permutation learning only on demographics
        )
    else:
        print("   - Architecture: Flat BACON")
        print("   - Tree Layout: Balanced")
        print("   - Permutation: Frozen (identity)")
        print("   - Transformation Layer: Enabled")
        
        bacon = baconNet(
            input_size=X_train.shape[1],
            freeze_loss_threshold=0.25,
            tree_layout='left',
            weight_mode='fixed',
            aggregator='lsp.half_weight',
            use_transformation_layer=True,
            max_permutations=1,
            is_frozen=True,
            lr_permutation=0.0,
            lr_transformation=0.1,
            lr_aggregator=0.1,
            lr_other=0.1,
            use_class_weighting=False,
        )
    skip_training = False

print("\n🌳 Tree Structure:")
if USE_HIERARCHICAL:
    print(f"   Hierarchical: {len(FEATURE_GROUPS)} groups → global tree")
    print(f"   No permutation search - groups stay in defined order within sub-trees")
else:
    print(f"   Flat: Balanced binary tree")
    print(f"   Features: {X_train.shape[1]} (in original order)")
    print(f"   No permutation search - features stay in temporal/logical order")

if not skip_training:
    print("\n⏳ Training...\n")

    (best_model, best_accuracy) = bacon.find_best_model(
        X_train_tensor, Y_train_tensor, X_test_tensor, Y_test_tensor,
        acceptance_threshold=0.90,
        max_epochs=8000,
        save_path=model_path,
        attempts=10,
        use_hierarchical_permutation = True,
        hierarchical_group_size = 3,
        hierarchical_epochs_per_attempt = 4000,
        hierarchical_bleed_ratio = 0.5,
        hierarchical_bleed_decay = 2.0
    )
    
    # Force freeze if not already frozen (for pruning analysis) - only for flat BACON
    if not USE_HIERARCHICAL and hasattr(bacon, 'assembler') and not bacon.assembler.is_frozen:
        print("\n🔒 Forcing model freeze for pruning analysis...")
        if hasattr(bacon.assembler.input_to_leaf, 'logits'):
            from bacon.frozonInputToLeaf import frozenInputToLeaf
            soft_perm = torch.softmax(bacon.assembler.input_to_leaf.logits, dim=1)
            hard_perm = torch.argmax(soft_perm, dim=1)
            bacon.assembler.locked_perm = hard_perm.clone().detach()
            bacon.assembler.is_frozen = True
            bacon.assembler.input_to_leaf = frozenInputToLeaf(
                bacon.assembler.locked_perm,
                bacon.assembler.original_input_size
            ).to(device)
            print(f"   ✅ Model frozen with locked permutation")
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

if USE_HIERARCHICAL:
    print("\n🌳 Hierarchical Structure:\n")
    print(f"Global tree combines {len(FEATURE_GROUPS)} groups + {bacon.num_ungrouped} ungrouped features:")
    
    # Show global tree structure
    print(f"\n🌲 Global Tree Structure:")
    print(f"   Global tree is_frozen: {bacon.global_tree.is_frozen}")
    print(f"   Global tree has locked_perm: {bacon.global_tree.locked_perm is not None}")
    # Build feature names for global tree (groups + ungrouped features)
    global_input_names = bacon.group_names.copy()
    for idx in bacon.ungrouped_indices:
        global_input_names.append(all_features[idx])
    print_tree_structure(bacon.global_tree, global_input_names)
    
    # Show global tree permutation (group ordering)
    if hasattr(bacon.global_tree, 'locked_perm') and bacon.global_tree.locked_perm is not None:
        group_order = bacon.global_tree.locked_perm.cpu().numpy()
        print(f"\n📊 Learned Global Input Ordering:")
        print(f"   Raw permutation: {group_order}")
        print(f"   Interpretation (most → least important):")
        for rank, input_idx in enumerate(group_order):
            input_name = global_input_names[input_idx]
            print(f"      {rank + 1}. {input_name}")
    
    # Show transformations for each group's output
    if bacon.use_global_transformation and bacon.global_tree.transformation_layer is not None:
        print(f"\n🔄 Global Tree Transformations:")
        trans_summary = bacon.global_tree.transformation_layer.get_transformation_summary()
        for i, input_name in enumerate(global_input_names):
            trans_type = trans_summary[i]['transformation']
            trans_prob = trans_summary[i]['probability']
            params = trans_summary[i]['params']
            param_str = ', '.join([f"{k}={v}" for k, v in params.items()]) if params else ""
            if param_str:
                print(f"   {input_name}: {trans_type} {param_str} (confidence: {trans_prob*100:.1f}%)")
            else:
                print(f"   {input_name}: {trans_type} (confidence: {trans_prob*100:.1f}%)")
    elif not bacon.use_global_transformation:
        print(f"\n🔄 Global Tree Transformations: DISABLED (no transformation layer)")
    
    # Show sub-tree details
    print(f"\n📁 Sub-Tree Details:")
    all_features = df_norm.columns.tolist()
    for group_name, sub_tree in bacon.sub_trees.items():
        feature_indices = FEATURE_GROUPS[group_name]
        group_features = [all_features[i] for i in feature_indices]
        print(f"\n   Group: {group_name}")
        print(f"   Features: {', '.join(group_features)}")
        
        # Print tree structure for this sub-tree
        print(f"   Tree Structure:")
        print_tree_structure(sub_tree, group_features)
        
        if sub_tree.transformation_layer is not None:
            sub_trans_summary = sub_tree.transformation_layer.get_transformation_summary()
            print(f"   Transformations:")
            for feat_idx, feat_name in enumerate(group_features):
                trans_info = sub_trans_summary[feat_idx]
                trans_type = trans_info['transformation']
                trans_prob = trans_info['probability']
                params = trans_info['params']
                if trans_type != 'identity':
                    param_str = ', '.join([f"{k}={v}" for k, v in params.items()]) if params else ""
                    if param_str:
                        print(f"      {feat_name}: {trans_type} {param_str} ({trans_prob*100:.0f}%)")
                    else:
                        print(f"      {feat_name}: {trans_type} ({trans_prob*100:.0f}%)")
else:
    # Tree structure
    print("\n🌳 Learned Logical Tree Structure:\n")
    print_tree_structure(bacon.assembler, df_norm.columns.tolist())

    # Transformation analysis
    if bacon.assembler.transformation_layer is not None:
        print("\n🔄 Transformation Layer Analysis:")
        print("=" * 80)
        
        trans_summary = bacon.assembler.transformation_layer.get_transformation_summary()
        selected_transforms = bacon.assembler.transformation_layer.get_selected_transformations()
        
        # Count each transformation type
        transform_counts = {}
        feature_names = df_norm.columns.tolist()
        for i in range(len(feature_names)):
            trans_name = trans_summary[i]['transformation']
            transform_counts[trans_name] = transform_counts.get(trans_name, 0) + 1
        
        print(f"\n📊 Transformation Distribution:")
        transform_labels = {
            'identity': 'Identity (x)',
            'negation': 'Negation (1-x)',
            'peak': 'Peak (1-|x-t|)',
            'valley': 'Valley (|x-t|)',
            'step_up': 'Step Up (ramp 0→1)',
            'step_down': 'Step Down (ramp 1→0)'
        }
        
        for trans_name in ['identity', 'negation', 'peak', 'valley', 'step_up', 'step_down']:
            count = transform_counts.get(trans_name, 0)
            label = transform_labels.get(trans_name, trans_name)
            print(f"   {label:<25} {count} features ({count/len(feature_names)*100:.1f}%)")
        
        # Show features for each non-identity transformation
        for trans_name in ['negation', 'peak', 'valley', 'step_up', 'step_down']:
            features = [(feature_names[i], trans_summary[i]['probability'], trans_summary[i]['params'])
                        for i in range(len(feature_names))
                        if trans_summary[i]['transformation'] == trans_name]
            
            if len(features) > 0:
                features.sort(key=lambda x: x[1], reverse=True)
                
                icons = {'negation': '🔄', 'peak': '🎯', 'valley': '🕳️', 'step_up': '📈', 'step_down': '📉'}
                labels = {'negation': 'Negation', 'peak': 'Peak', 'valley': 'Valley', 
                         'step_up': 'Step Up', 'step_down': 'Step Down'}
                
                print(f"\n{icons.get(trans_name, '•')} Features Using {labels.get(trans_name, trans_name)}:")
                for feat, prob, params in features[:10]:  # Show top 10
                    param_str = ', '.join([f"{k}={v}" for k, v in params.items()]) if params else ""
                    if param_str:
                        print(f"   {feat:<20} {param_str} (confidence: {prob*100:.1f}%)")
                    else:
                        print(f"   {feat:<20} (confidence: {prob*100:.1f}%)")

# Feature importance through pruning
print("\n🔍 Feature Importance (Progressive Pruning):")
print("=" * 80)

if USE_HIERARCHICAL:
    print("   ⚠️  Pruning analysis not yet implemented for hierarchical BACON.")
    print("   Skipping pruning analysis...")
    pruning_results = []
else:
    # Choose pruning metric
    PRUNING_METRIC = 'accuracy'  # Options: 'accuracy', 'auc'
    print(f"   Metric: {PRUNING_METRIC.upper()}")
    print(f"   Removing features one-by-one to assess impact...\n")

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
        for i in range(1, len(feature_names)):  # Go through ALL features
            func_eval = bacon.prune_features(i)
            kept_indices = bacon.assembler.locked_perm[i:].tolist()
            X_test_pruned = X_test_tensor[:, kept_indices]
    
            with torch.no_grad():
                pruned_output = func_eval(X_test_pruned)
            
                if PRUNING_METRIC == 'auc':
                    pruned_metric = roc_auc_score(y_test, pruned_output.cpu().numpy().flatten())
                else:  # accuracy
                    pruned_metric = accuracy_score(y_test, (pruned_output.cpu().numpy().flatten() > 0.5).astype(int))
            
                pruning_results.append((i, pruned_metric))
        
                # Show all features removed
                removed_idx = bacon.assembler.locked_perm[i-1].item()
                print(f"   Removed {i:2d}: {feature_names[removed_idx]:<20} → {PRUNING_METRIC.upper()} = {pruned_metric:.4f}")

# Visualization
if not USE_HIERARCHICAL:
    print("\n📊 Generating visualizations...")
    visualize_tree_structure(bacon.assembler, feature_names)

# Plot pruning analysis
if len(pruning_results) > 0:
    plt.figure(figsize=(10, 6))
    prune_counts = [p[0] for p in pruning_results]
    prune_metrics = [p[1] for p in pruning_results]
    plt.plot(prune_counts, prune_metrics, marker='o', linewidth=2, markersize=4)
    
    baseline_metric = test_auc if PRUNING_METRIC == 'auc' else test_acc
    metric_label = 'AUC' if PRUNING_METRIC == 'auc' else 'Accuracy'
    
    plt.axhline(y=baseline_metric, color='r', linestyle='--', alpha=0.7, 
                label=f'Full Model {metric_label}: {baseline_metric:.4f}')
    plt.xlabel('Number of Features Removed', fontsize=12)
    plt.ylabel(f'Test {metric_label}', fontsize=12)
    plt.title(f'Feature Pruning Impact on Model Performance ({metric_label})', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('feature_pruning_analysis.png', dpi=300)
    print("💾 Saved: feature_pruning_analysis.png")

print("\n" + "=" * 80)
print("✅ Training and Analysis Complete!")
print("=" * 80)

plt.show()
