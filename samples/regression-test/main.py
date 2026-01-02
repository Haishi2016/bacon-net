"""Regression test for BACON - Approximating synthetic continuous data."""

import sys
sys.path.insert(0, '../../')

import torch
from bacon.baconNet import baconNet
from bacon.visualization import visualize_tree_structure, print_tree_structure
import logging
from dataset import prepare_data
from bacon.transformationLayer import IdentityTransformation, NegationTransformation, PeakTransformation

logging.basicConfig(level=logging.INFO, format='%(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load synthetic regression data
x_train, y_train, x_test, y_test, feature_names = prepare_data(device)
input_size = x_train.shape[1]

print(f"📊 Dataset: {x_train.shape[0]} samples, {input_size} features")
print(f"🎯 Target range: [{y_train.min():.3f}, {y_train.max():.3f}]")
print()


trans = [
    IdentityTransformation(1), 
    NegationTransformation(1)
]

# Create BACON model for regression
# Using math.avg aggregator for continuous output
bacon = baconNet(
    input_size, 
    aggregator='lsp.full_weight',
    weight_mode='trainable',    
    use_transformation_layer=True,
    transformations=trans,
    weight_normalization='softmax',
    weight_penalty_strength=1e-3
)

# Train the model
print("🔧 Training BACON model...")
(best_model, best_accuracy) = bacon.find_best_model(
    x_train, y_train, 
    x_test, y_test,     
    use_hierarchical_permutation=True,
    hierarchical_group_size=2,
    hierarchical_bleed_ratio=0.5,
    acceptance_threshold=1.0,  # High threshold for small dataset
    attempts=5,
    max_epochs=2000,    
    save_model=True,
    binary_threshold=-1.0  # Disable binarization for regression
)

print(f"\n🏆 Best accuracy: {best_accuracy * 100:.2f}%")

# Calculate and display regression metrics
with torch.no_grad():
    predictions = bacon(x_test)
    mse = torch.mean((predictions - y_test) ** 2).item()
    mae = torch.mean(torch.abs(predictions - y_test)).item()
    rmse = torch.sqrt(torch.tensor(mse)).item()
    
    print(f"\n📈 Regression Metrics:")
    print(f"   MSE:  {mse:.6f}")
    print(f"   MAE:  {mae:.6f}")
    print(f"   RMSE: {rmse:.6f}")
    
    # Show some example predictions
    print(f"\n🔍 Sample Predictions (first 5):")
    for i in range(min(5, len(x_test))):
        print(f"   Target: {y_test[i].item():.3f}, Predicted: {predictions[i].item():.3f}, Error: {abs(y_test[i].item() - predictions[i].item()):.3f}")

# Visualize the learned structure
print("\n🌲 Learned tree structure:")
print_tree_structure(bacon.assembler, feature_names)
visualize_tree_structure(bacon.assembler, feature_names)
