"""Minimal test to identify when NaN first appears"""
import torch
import sys
sys.path.insert(0, '../../')
from bacon import baconNet

# Generate simple balanced dataset (10 variables)
# torch.manual_seed(42)
n_samples = 1000
input_size = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train = torch.randint(0, 2, (n_samples, input_size)).float().to(device)

# Balanced labels: 50% positive
y_train = torch.zeros((n_samples, 1)).to(device)
y_train[:n_samples//2] = 1
# Shuffle
perm = torch.randperm(n_samples)
X_train = X_train[perm]
y_train = y_train[perm]

print(f"Dataset: {n_samples} samples, {input_size} variables")
print(f"Positive ratio: {y_train.mean():.2%}")

# Create model with same config as ablation test
bacon = baconNet(
    input_size=input_size,
    aggregator="bool.min_max",  # Same as ablation test
    loss_weight_perm_entropy=0.1,
    loss_weight_trans_entropy=0.0,
    loss_weight_perm_sparsity=0.01,  # Very low sparsity
    use_transformation_layer=False,
    weight_mode='fixed',  # Same as ablation test
    use_class_weighting=False,  # Same as ablation test
    permutation_initial_temperature=10.0,  # Same as ablation test
    permutation_final_temperature=0.05,  # Same as ablation test
).to(device)

print("\n🔍 Starting training with NaN detection...")
print("=" * 60)

# Manual training loop to detect first NaN
optimizer = torch.optim.Adam(bacon.parameters(), lr=0.001)
bacon.train()

for epoch in range(1, 101):
    optimizer.zero_grad()
    
    # Forward pass
    outputs = bacon.assembler(X_train, targets=y_train)
    
    # Check outputs
    if torch.isnan(outputs).any():
        print(f"\n❌ FIRST NaN DETECTED at epoch {epoch} in MODEL OUTPUTS")
        print(f"   Output stats: min={outputs.min()}, max={outputs.max()}, mean={outputs.mean()}")
        print(f"   NaN count: {torch.isnan(outputs).sum()}/{outputs.numel()}")
        break
    
    # Compute loss
    main_loss = torch.nn.functional.binary_cross_entropy(outputs, y_train)
    
    if torch.isnan(main_loss):
        print(f"\n❌ FIRST NaN DETECTED at epoch {epoch} in BCE LOSS")
        print(f"   Output stats: min={outputs.min()}, max={outputs.max()}, mean={outputs.mean()}")
        break
    
    # Backward pass
    main_loss.backward()
    
    # Check gradients
    nan_grads = []
    for name, param in bacon.named_parameters():
        if param.grad is not None and torch.isnan(param.grad).any():
            nan_grads.append(name)
    
    if nan_grads:
        print(f"\n❌ FIRST NaN DETECTED at epoch {epoch} in GRADIENTS")
        print(f"   Parameters with NaN gradients: {nan_grads}")
        break
    
    optimizer.step()
    
    # Check parameters after update
    nan_params = []
    for name, param in bacon.named_parameters():
        if torch.isnan(param).any():
            nan_params.append(name)
    
    if nan_params:
        print(f"\n❌ FIRST NaN DETECTED at epoch {epoch} in PARAMETERS (after optimizer step)")
        print(f"   Parameters with NaN: {nan_params}")
        break
    
    if epoch % 10 == 0:
        accuracy = ((outputs > 0.5).float() == y_train).float().mean()
        print(f"Epoch {epoch}/100, Loss: {main_loss.item():.4f}, Accuracy: {accuracy:.2%}")

print("\n✅ Training completed without NaN" if epoch == 100 else "\n❌ Training failed with NaN")
