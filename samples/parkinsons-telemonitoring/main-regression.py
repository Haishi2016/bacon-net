"""Parkinson's Telemonitoring - BACON Regression Mode

This version uses BACON in regression mode to predict continuous UPDRS scores
instead of binary classification, which is more appropriate for this dataset.
"""

# Note: required to import baconNet from local folder
import sys
sys.path.insert(0, '../../')
sys.path.insert(0, '../')

import torch
import logging
from bacon.baconNet import baconNet
from bacon.transformationLayer import IdentityTransformation, NegationTransformation
from bacon.visualization import visualize_tree_structure, print_tree_structure
from dataset import prepare_data

logging.basicConfig(level=logging.INFO, format='%(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Prepare data in REGRESSION mode (continuous UPDRS scores)
X_train, Y_train, X_test, Y_test, feature_names, target_stats = prepare_data(
    device, 
    target='motor_UPDRS',
    mode='regression'  # Use continuous values, not binary classification
)
num_features = len(feature_names)

# Store target stats for denormalization
y_min = target_stats['min']
y_max = target_stats['max']

print(f"\n📊 Model will use {num_features} input features")
print(f"🎯 Target range (normalized): [{Y_train.min():.2f}, {Y_train.max():.2f}]")
print(f"🎯 Target normalization: Min-Max [0, 1]")
print(f"🎯 Original range: [{y_min:.2f}, {y_max:.2f}] UPDRS")

# Configure transformations
trans = [
    IdentityTransformation(1), 
    NegationTransformation(1)
]

# Create BACON model for regression
# Using math.avg or lsp aggregator for continuous output
bacon = baconNet(
    num_features, 
    aggregator='lsp.full_weight',
    weight_mode='trainable',    
    use_transformation_layer=True,
    transformations=trans,
    weight_normalization='softmax',
    weight_penalty_strength=1e-3,
    permutation_initial_temperature=5.0,
    permutation_final_temperature=4.0
)

# Train the model
print("\n🔧 Training BACON model in regression mode...")
(best_model, best_accuracy) = bacon.find_best_model(
    X_train, Y_train, 
    X_test, Y_test,     
    use_hierarchical_permutation=True,
    hierarchical_group_size=8,
    hierarchical_bleed_ratio=0.5,
    acceptance_threshold=1.0,
    attempts=10,
    max_epochs=2000,    
    save_model=True,
    binary_threshold=-1.0  # Disable binarization for regression
)

print(f"\n🏆 Best R² score: {best_accuracy * 100:.2f}%")

# Calculate and display regression metrics
with torch.no_grad():
    predictions = bacon(X_test)
    
    # Denormalize predictions and targets using inverse min-max
    # Inverse: y = y_norm * (y_max - y_min) + y_min
    predictions_denorm = predictions * (y_max - y_min) + y_min
    Y_test_denorm = Y_test * (y_max - y_min) + y_min
    
    # Calculate metrics on denormalized values
    mse = torch.mean((predictions_denorm - Y_test_denorm) ** 2).item()
    mae = torch.mean(torch.abs(predictions_denorm - Y_test_denorm)).item()
    rmse = torch.sqrt(torch.tensor(mse)).item()
    
    # Calculate R² manually for verification (on denormalized values)
    ss_res = torch.sum((Y_test_denorm - predictions_denorm) ** 2).item()
    ss_tot = torch.sum((Y_test_denorm - Y_test_denorm.mean()) ** 2).item()
    r2 = 1 - (ss_res / ss_tot)
    
    print(f"\n📈 Regression Metrics (on original UPDRS scale):")
    print(f"   MSE:  {mse:.6f}")
    print(f"   MAE:  {mae:.6f}")
    print(f"   RMSE: {rmse:.6f}")
    print(f"   R²:   {r2:.6f}")
    
    # Show some example predictions
    print(f"\n🔍 Sample Predictions (first 10, original UPDRS scale):")
    print(f"{'Target':>8} {'Predicted':>10} {'Error':>8}")
    print("-" * 30)
    for i in range(min(10, len(X_test))):
        target = Y_test_denorm[i].item()
        pred = predictions_denorm[i].item()
        error = abs(target - pred)
        print(f"{target:8.2f} {pred:10.2f} {error:8.2f}")
    
    # Error statistics
    errors = torch.abs(predictions_denorm - Y_test_denorm)
    print(f"\n📊 Error Statistics (original UPDRS scale):")
    print(f"   Min error:    {errors.min().item():.2f}")
    print(f"   Max error:    {errors.max().item():.2f}")
    print(f"   Median error: {errors.median().item():.2f}")
    print(f"   Mean error:   {errors.mean().item():.2f} (MAE)")
    
    # Evaluate as binary classifier using 75th percentile threshold
    print(f"\n🔍 Binary Classification Evaluation (75th percentile threshold):")
    
    # Calculate 75th percentile threshold in NORMALIZED space (where model operates)
    threshold_75_norm = torch.quantile(Y_train, 0.75).item()
    
    # Also calculate what this means in denormalized space for reporting
    Y_train_denorm = Y_train * (y_max - y_min) + y_min
    threshold_75_denorm = torch.quantile(Y_train_denorm, 0.75).item()
    
    print(f"   Threshold (normalized): {threshold_75_norm:.4f}")
    print(f"   Threshold (UPDRS scale): {threshold_75_denorm:.2f}")
    
    # Convert to binary predictions and labels using NORMALIZED threshold
    y_pred_binary = (predictions >= threshold_75_norm).float()
    y_test_binary = (Y_test >= threshold_75_norm).float()
    
    # Calculate binary classification metrics
    from sklearn.metrics import accuracy_score, f1_score, average_precision_score
    
    y_pred_binary_np = y_pred_binary.cpu().numpy().flatten()
    y_test_binary_np = y_test_binary.cpu().numpy().flatten()
    predictions_np = predictions.cpu().numpy().flatten()  # Use normalized predictions for AUPRC
    
    accuracy = accuracy_score(y_test_binary_np, y_pred_binary_np)
    f1 = f1_score(y_test_binary_np, y_pred_binary_np, zero_division=0)
    auprc = average_precision_score(y_test_binary_np, predictions_np)
    
    print(f"   Accuracy: {accuracy:.4f}")
    print(f"   F1 Score: {f1:.4f}")
    print(f"   AUPRC:    {auprc:.4f}")
    
    # Class distribution
    n_high = int(y_test_binary.sum().item())
    n_low = len(y_test_binary) - n_high
    n_pred_high = int(y_pred_binary.sum().item())
    n_pred_low = len(y_pred_binary) - n_pred_high
    print(f"   Test set actual:    {n_low} low severity, {n_high} high severity")
    print(f"   Test set predicted: {n_pred_low} low severity, {n_pred_high} high severity")

# Visualize the learned structure
print("\n🌲 Learned tree structure:")
print_tree_structure(bacon.assembler, feature_names)
visualize_tree_structure(bacon.assembler, feature_names)

print("\n✅ Regression model training complete!")
