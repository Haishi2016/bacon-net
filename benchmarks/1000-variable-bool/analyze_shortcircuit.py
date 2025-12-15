"""Analyze short-circuit behavior in generated boolean expressions."""

import sys
sys.path.insert(0, '../../')

from bacon.utils import generate_classic_boolean_data
import torch
import numpy as np

print("🔍 Analyzing short-circuit behavior...\n")

for num_vars in [10, 50, 100, 500, 1000]:
    print(f"\n{'='*60}")
    print(f"Testing with {num_vars} variables:")
    print(f"{'='*60}")
    
    # Generate data
    x_train, y_train, metadata = generate_classic_boolean_data(
        num_vars=num_vars,
        repeat_factor=1000 if num_vars > 10 else 1,
        randomize=True if num_vars > 10 else False
    )
    
    # Analyze feature importance by flipping each bit
    print(f"\n📊 Feature importance analysis:")
    
    important_vars = []
    
    for var_idx in range(min(20, num_vars)):  # Check first 20 variables
        # Flip this variable and see if output changes
        x_flipped = x_train.clone()
        x_flipped[:, var_idx] = 1 - x_flipped[:, var_idx]
        
        # Recompute labels with flipped variable
        y_flipped = []
        ops = metadata['ops']
        for row in x_flipped:
            result = bool(row[0])
            for i in range(1, num_vars):
                if ops[i - 1] == "and":
                    result = result and bool(row[i])
                else:
                    result = result or bool(row[i])
            y_flipped.append(int(result))
        
        y_flipped = torch.tensor(y_flipped, dtype=torch.float32).reshape(-1, 1)
        
        # Count how many outputs changed
        changes = (y_flipped != y_train).sum().item()
        change_pct = 100 * changes / len(y_train)
        
        if change_pct > 1.0:  # Variable matters if it changes >1% of outputs
            important_vars.append(var_idx)
        
        if var_idx < 10:  # Show first 10
            status = "✅ IMPORTANT" if change_pct > 1.0 else "❌ irrelevant"
            print(f"   Var {var_idx:3d}: {change_pct:6.2f}% outputs changed - {status}")
    
    print(f"\n📈 Summary:")
    print(f"   Important variables: {len(important_vars)}/{num_vars} ({100*len(important_vars)/num_vars:.1f}%)")
    print(f"   Expression operators: {metadata['ops'][:10]}{'...' if len(metadata['ops']) > 10 else ''}")
    
    # Check for trivial expressions (all outputs same)
    unique_outputs = torch.unique(y_train)
    if len(unique_outputs) == 1:
        print(f"   ⚠️  WARNING: Expression is trivial (all outputs = {unique_outputs[0].item():.0f})")
    
    # Check output distribution
    output_mean = y_train.mean().item()
    print(f"   Output distribution: {output_mean*100:.1f}% True, {(1-output_mean)*100:.1f}% False")

print("\n" + "="*60)
print("Analysis complete!")
