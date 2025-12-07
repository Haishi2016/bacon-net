"""
Test script for parameterized transformations.

This tests the new ParameterizedTransformation architecture with:
1. Identity transformation (no parameters)
2. Negation transformation (no parameters)  
3. Peak transformation (learnable peak location per feature)
"""

import torch
import torch.nn as nn
from bacon.transformationLayer import (
    TransformationLayer,
    IdentityTransformation,
    NegationTransformation,
    PeakTransformation
)

def test_basic_transformations():
    """Test that basic transformations work correctly."""
    print("=" * 60)
    print("Test 1: Basic Transformations")
    print("=" * 60)
    
    # Create layer with default transformations (identity and negation)
    layer = TransformationLayer(num_features=3, device='cpu')
    
    # Create test input
    x = torch.tensor([[0.2, 0.5, 0.8]], dtype=torch.float32)
    
    print(f"Input: {x}")
    
    # Forward pass
    output = layer(x)
    print(f"Output: {output}")
    
    # Check summary
    summary = layer.get_transformation_summary()
    print("\nTransformation Summary:")
    for feat_idx, info in summary.items():
        print(f"  Feature {feat_idx}: {info['transformation']} "
              f"(prob={info['probability']:.3f})")
    
    print("\n✓ Basic transformations test passed\n")


def test_peak_transformation():
    """Test that peak transformation with learnable parameters works."""
    print("=" * 60)
    print("Test 2: Peak Transformation")
    print("=" * 60)
    
    # Create layer with peak transformation
    transforms = [
        IdentityTransformation(num_features=3),
        NegationTransformation(num_features=3),
        PeakTransformation(num_features=3)
    ]
    layer = TransformationLayer(
        num_features=3,
        transformations=transforms,
        device='cpu'
    )
    
    # Check that peak parameters were initialized
    print("Initialized parameters:")
    for name, param in layer.transform_params.items():
        print(f"  {name}: {param.data}")
    
    # Create test input
    x = torch.tensor([
        [0.1, 0.3, 0.7],
        [0.5, 0.5, 0.5],
        [0.9, 0.7, 0.3]
    ], dtype=torch.float32)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Input:\n{x}")
    
    # Forward pass
    output = layer(x)
    print(f"\nOutput shape: {output.shape}")
    print(f"Output:\n{output}")
    
    # Check summary
    summary = layer.get_transformation_summary()
    print("\nTransformation Summary:")
    for feat_idx, info in summary.items():
        params_str = ", ".join(f"{k}={v}" for k, v in info['params'].items())
        params_part = f" ({params_str})" if params_str else ""
        print(f"  Feature {feat_idx}: {info['transformation']} "
              f"(prob={info['probability']:.3f}){params_part}")
    
    print("\n✓ Peak transformation test passed\n")


def test_gradient_flow():
    """Test that gradients flow through parameterized transformations."""
    print("=" * 60)
    print("Test 3: Gradient Flow")
    print("=" * 60)
    
    # Create layer with peak transformation
    transforms = [
        IdentityTransformation(num_features=2),
        PeakTransformation(num_features=2)
    ]
    layer = TransformationLayer(
        num_features=2,
        transformations=transforms,
        device='cpu'
    )
    
    # Set logits to strongly prefer peak transformation
    with torch.no_grad():
        layer.logits[:, 1] = 5.0  # Prefer peak transformation
        layer.logits[:, 0] = -5.0
    
    # Create input and target
    x = torch.tensor([[0.3, 0.7]], dtype=torch.float32)
    target = torch.tensor([[0.8, 0.8]], dtype=torch.float32)
    
    # Forward pass
    output = layer(x)
    loss = nn.MSELoss()(output, target)
    
    print(f"Input: {x}")
    print(f"Output: {output}")
    print(f"Target: {target}")
    print(f"Loss: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    print("\nGradients:")
    for name, param in layer.transform_params.items():
        if param.grad is not None:
            print(f"  {name}.grad: {param.grad}")
        else:
            print(f"  {name}.grad: None (parameter not used)")
    
    print(f"  logits.grad: {layer.logits.grad}")
    
    # Verify peak location has gradients
    peak_param = layer.transform_params['t1_peak_loc']
    assert peak_param.grad is not None, "Peak location should have gradients"
    assert not torch.all(peak_param.grad == 0), "Peak location gradients should be non-zero"
    
    print("\n✓ Gradient flow test passed\n")


def test_training_step():
    """Test a simple training scenario where peak should move toward optimal value."""
    print("=" * 60)
    print("Test 4: Simple Training")
    print("=" * 60)
    
    # Create layer with peak transformation
    transforms = [
        IdentityTransformation(num_features=1),
        PeakTransformation(num_features=1)
    ]
    layer = TransformationLayer(
        num_features=1,
        transformations=transforms,
        temperature=0.1,  # Low temperature for sharp selection
        device='cpu'
    )
    
    # Set logits to strongly prefer peak transformation
    with torch.no_grad():
        layer.logits[0, 1] = 10.0
        layer.logits[0, 0] = -10.0
    
    # Optimizer
    optimizer = torch.optim.Adam(layer.parameters(), lr=0.1)
    
    # Training data: inputs around 0.3, want output close to 1.0
    # This should train peak to move toward 0.3
    x_train = torch.tensor([[0.3], [0.28], [0.32], [0.35]], dtype=torch.float32)
    y_train = torch.tensor([[1.0], [1.0], [1.0], [1.0]], dtype=torch.float32)
    
    print("Training to make output=1.0 when input≈0.3")
    print("Peak transformation: 1 - |x - t|, so optimal t≈0.3\n")
    
    # Initial peak location
    initial_peak = layer.transform_params['t1_peak_loc'].item()
    print(f"Initial peak location: {initial_peak:.3f}")
    
    # Train for a few steps
    for epoch in range(50):
        optimizer.zero_grad()
        output = layer(x_train)
        loss = nn.MSELoss()(output, y_train)
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            peak_loc = layer.transform_params['t1_peak_loc'].item()
            print(f"Epoch {epoch:2d}: loss={loss.item():.4f}, peak_loc={peak_loc:.3f}")
    
    # Final peak location
    final_peak = layer.transform_params['t1_peak_loc'].item()
    print(f"\nFinal peak location: {final_peak:.3f}")
    print(f"Expected: ~0.3 (mean of training inputs)")
    print(f"Peak moved: {abs(final_peak - initial_peak):.3f}")
    
    # Verify peak moved toward 0.3
    assert abs(final_peak - 0.3) < abs(initial_peak - 0.3), \
        "Peak should move closer to optimal value 0.3"
    
    print("\n✓ Training test passed\n")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("TESTING PARAMETERIZED TRANSFORMATIONS")
    print("=" * 60 + "\n")
    
    test_basic_transformations()
    test_peak_transformation()
    test_gradient_flow()
    test_training_step()
    
    print("=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)
