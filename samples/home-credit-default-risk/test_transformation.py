import sys
sys.path.insert(0, '../../')  # Use insert(0, ...) to prioritize local version
import torch
from bacon.baconNet import baconNet

print("Testing transformation layer integration...")

device = torch.device("cpu")  # Use CPU for testing

# Test that baconNet accepts transformation parameters
try:
    bacon = baconNet(
        input_size=10,
        use_transformation_layer=True,
        transformation_temperature=1.0,
        transformation_use_gumbel=False
    )
    print("SUCCESS: baconNet accepts transformation layer parameters")
    
    # Test forward pass
    x = torch.rand(5, 10).to(device)
    bacon.assembler.to(device)
    output = bacon(x)
    print(f"SUCCESS: Forward pass works, output shape: {output.shape}")
    
    # Test transformation layer exists
    if bacon.assembler.transformation_layer is not None:
        print("SUCCESS: Transformation layer is initialized")
        print(f"  - Number of features: {bacon.assembler.transformation_layer.num_features}")
        print(f"  - Number of transformations: {bacon.assembler.transformation_layer.num_transforms}")
    else:
        print("ERROR: Transformation layer is None")
        
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

# Test without transformation layer (backward compatibility)
print("\nTesting backward compatibility (transformation layer disabled)...")
try:
    bacon_no_transform = baconNet(input_size=10)
    print("SUCCESS: baconNet works without transformation layer")
    
    if bacon_no_transform.assembler.transformation_layer is None:
        print("SUCCESS: Transformation layer is None when disabled")
    else:
        print("ERROR: Transformation layer should be None when disabled")
        
except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()

print("\nAll tests completed!")
