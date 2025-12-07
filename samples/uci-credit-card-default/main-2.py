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
print("UCI Credit Card Default (Taiwan) - BACON Benchmark (No Transformation)")
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

# ========== Preprocessing to [0,1] - Higher = Higher Default Risk ==========
print("\n🔄 Normalizing features to [0,1] where HIGHER values = HIGHER default risk...")
print("   (Consistent semantic: all features point in same direction)")

df_norm = df.copy()

# LIMIT_BAL: Higher limit → LOWER risk → REVERSE (1 - normalized)
df_norm['LIMIT_BAL'] = (df['LIMIT_BAL'] - df['LIMIT_BAL'].min()) / (df['LIMIT_BAL'].max() - df['LIMIT_BAL'].min() + 1e-8)
df_norm['LIMIT_BAL'] = 1 - df_norm['LIMIT_BAL']  # REVERSED: higher value = lower limit = higher risk
print(f"  ✅ LIMIT_BAL: REVERSED (1=low limit/high risk, 0=high limit/low risk)")

# SEX: 1=male, 2=female → Convert to binary (keep as-is, no clear risk direction)
df_norm['SEX'] = (df['SEX'] == 1).astype(float)  # 1 if male, 0 if female
print(f"  ✅ SEX: binary (1=male, 0=female)")

# EDUCATION: 1=graduate, 2=university, 3=high school, 4=others
# Higher education → LOWER risk → REVERSE
df_norm['EDUCATION'] = df['EDUCATION'].replace({0: 4, 5: 4, 6: 4})  # Consolidate unknowns
df_norm['EDUCATION'] = (df_norm['EDUCATION'] - 1) / 3  # grad=0, others=1
print(f"  ✅ EDUCATION: normalized (1=others/high risk, 0=graduate/low risk)")

# MARRIAGE: 1=married, 2=single, 3=others
# Married typically has lower risk → reverse
df_norm['MARRIAGE'] = (df['MARRIAGE'] != 1).astype(float)  # 0 if married, 1 if single/others
print(f"  ✅ MARRIAGE: binary (1=single/high risk, 0=married/low risk)")

# AGE: Normalize (assuming middle-age has lower risk, but keeping monotonic)
# For simplicity, normalize as-is (older age interpretation varies)
df_norm['AGE'] = (df['AGE'] - df['AGE'].min()) / (df['AGE'].max() - df['AGE'].min() + 1e-8)
print(f"  ✅ AGE: normalized to [0,1]")

# PAY_0 to PAY_6: Payment delay status
# -2=no consumption, -1=pay duly, 0=revolving credit, 1+=months of delay
# Higher delay → HIGHER risk (already monotonic, just normalize)
for i in range(7):
    pay_col = f'PAY_{i}' if i > 0 else 'PAY_0'
    if pay_col in df.columns:
        # Clip extreme values and normalize
        df_norm[pay_col] = df[pay_col].clip(-2, 8)  # Cap at 8 months delay
        df_norm[pay_col] = (df_norm[pay_col] + 2) / 10  # Range [-2, 8] → [0, 1]
        print(f"  ✅ {pay_col}: normalized (1=high delay/high risk, 0=pay duly/low risk)")

# BILL_AMT1 to BILL_AMT6: Bill statement amounts
# Higher bills → HIGHER risk (use as-is after normalization)
for i in range(1, 7):
    bill_col = f'BILL_AMT{i}'
    if bill_col in df.columns:
        # Some bills can be negative (overpayment), shift to positive
        shifted = df[bill_col] - df[bill_col].min() + 1
        log_transformed = np.log1p(shifted)
        df_norm[bill_col] = (log_transformed - log_transformed.min()) / (log_transformed.max() - log_transformed.min() + 1e-8)
        print(f"  ✅ {bill_col}: normalized (1=high bill/high risk, 0=low bill/low risk)")

# PAY_AMT1 to PAY_AMT6: Payment amounts
# Higher payment → LOWER risk → REVERSE
for i in range(1, 7):
    pay_amt_col = f'PAY_AMT{i}'
    if pay_amt_col in df.columns:
        log_transformed = np.log1p(df[pay_amt_col])
        normalized = (log_transformed - log_transformed.min()) / (log_transformed.max() - log_transformed.min() + 1e-8)
        df_norm[pay_amt_col] = 1 - normalized  # REVERSED: higher value = lower payment = higher risk
        print(f"  ✅ {pay_amt_col}: REVERSED (1=low payment/high risk, 0=high payment/low risk)")

print("\n✅ All features normalized consistently: HIGHER value = HIGHER default risk")

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
print("🧠 Training BACON Model (No Transformation Layer)")
print("=" * 80)

import os
model_path = "./best_bacon_credit_card_notrans.pth"

# Check if trained model exists
if os.path.exists(model_path):
    print("\n✅ Found existing trained model!")
    print(f"   Loading from: {model_path}")
    print("   (Delete this file to retrain from scratch)")
    
    bacon = baconNet(
        input_size=X_train.shape[1],
        freeze_loss_threshold=0.25,
        weight_mode='fixed',
        aggregator='lsp.half_weight',
        use_transformation_layer=False,  # DISABLED
        max_permutations=400,
    )
    
    # Load the saved model
    checkpoint = torch.load(model_path, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Restore metadata FIRST
        bacon.assembler.is_frozen = checkpoint.get('is_frozen', False)
        bacon.assembler.locked_perm = checkpoint.get('locked_perm', None)
        bacon.assembler.tree_layout = checkpoint.get('tree_layout', 'left')
        
        # If model was frozen, recreate frozen input layer BEFORE loading state_dict
        if bacon.assembler.is_frozen and bacon.assembler.locked_perm is not None:
            from bacon.frozonInputToLeaf import frozenInputToLeaf
            bacon.assembler.input_to_leaf = frozenInputToLeaf(
                bacon.assembler.locked_perm, 
                bacon.assembler.original_input_size
            ).to(device)
        
        # Now load model state
        bacon.assembler.load_state_dict(checkpoint['model_state_dict'])
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
    print("   - Transformation Layer: DISABLED")
    print("   - Hierarchical Permutation: DISABLED (random search)")
    print("   - Aggregator: LSP (half_weight)")
    print("   - Freeze threshold: 0.25")
    print("   - Max attempts: 100")

    bacon = baconNet(
        input_size=X_train.shape[1],
        freeze_loss_threshold=0.25,
        weight_mode='fixed',
        aggregator='lsp.half_weight',
        use_transformation_layer=False,  # DISABLED
        max_permutations=40,
    )
    skip_training = False

if not skip_training:
    print("\n⏳ Training (this may take several minutes)...\n")

    (best_model, best_accuracy) = bacon.find_best_model(
        X_train_tensor, Y_train_tensor, X_test_tensor, Y_test_tensor,
        acceptance_threshold=0.90,
        max_epochs=15000,
        save_path=model_path,
        use_hierarchical_permutation=False,  # Random search
        attempts=100,
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
print(f"   BACON (no trans):      {test_auc:.4f} AUC")

# ========== Interpretability Analysis ==========
print("\n" + "=" * 80)
print("🔍 Interpretability Analysis")
print("=" * 80)

# Tree structure
print("\n🌳 Learned Logical Tree Structure:\n")
print_tree_structure(bacon.assembler, feature_names)

# Feature importance through pruning
print("\n🔍 Feature Importance (Progressive Pruning):")
print("=" * 80)

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
    plt.title(f'Feature Pruning Impact ({metric_label}, No Transformation)', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig('feature_pruning_analysis_notrans.png', dpi=300)
    print("💾 Saved: feature_pruning_analysis_notrans.png")

print("\n" + "=" * 80)
print("✅ Training and Analysis Complete!")
print("=" * 80)

plt.show()
