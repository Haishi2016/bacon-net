import sys
sys.path.insert(0, '../../')

import torch
import itertools
from bacon.baconNet import baconNet
from bacon.utils import generate_classic_boolean_data
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_simple_negation_data(device):
    """Generate a simple expression: (NOT A and B) or C"""
    num_samples = 100
    data = []
    labels = []
    
    for _ in range(num_samples):
        a = torch.rand(1).item() > 0.5
        b = torch.rand(1).item() > 0.5
        c = torch.rand(1).item() > 0.5
        
        # Expression: (NOT A and B) or C
        result = ((not a) and b) or c
        
        data.append([float(a), float(b), float(c)])
        labels.append(float(result))
    
    return (
        torch.tensor(data, dtype=torch.float32, device=device),
        torch.tensor(labels, dtype=torch.float32, device=device).unsqueeze(1),
        {
            "expression": "(NOT A and B) or C",
            "var_names": ["A", "B", "C"],
            "negated": [True, False, False]  # A is negated
        }
    )

# Generate data
x, y, info = generate_simple_negation_data(device)

print("🧠 Simple Negation Test")
print(f"➗ Expression: {info['expression']}")
print(f"🔢 Variables: {', '.join(info['var_names'])}")
print(f"🔄 Negated: A\n")

# Train model
bacon = baconNet(
    3, 
    aggregator='bool.min_max', 
    weight_mode='fixed', 
    loss_amplifier=1000, 
    normalize_andness=False,
    use_transformation_layer=True,
    transformation_temperature=1.0,
    transformation_use_gumbel=False
)

(best_model, best_accuracy) = bacon.find_best_model(
    x, y, x, y, 
    acceptance_threshold=0.95, 
    attempts=10, 
    max_epochs=3000, 
    save_model=False
)

print(f"\n🏆 Best accuracy: {best_accuracy * 100:.2f}%\n")

# Analyze transformations
if bacon.assembler.transformation_layer is not None:
    trans_summary = bacon.assembler.transformation_layer.get_transformation_summary()
    
    print("📊 Learned Transformations:")
    for i, var_name in enumerate(info['var_names']):
        trans_type = trans_summary[i]['transformation']
        confidence = trans_summary[i]['probability'] * 100
        expected = "negation" if info['negated'][i] else "identity"
        status = "✅" if trans_type == expected else "❌"
        print(f"   {status} {var_name}: {trans_type:10s} (confidence: {confidence:5.1f}%) [Expected: {expected}]")

# VERIFY LOGICAL EQUIVALENCE
print("\n🔍 Logical Equivalence Verification")
print("=" * 70)

# Test all 8 possible combinations (2^3)
all_inputs = list(itertools.product([0, 1], repeat=3))

# Original expression results
original_outputs = []
for a, b, c in all_inputs:
    result = ((not a) and b) or c  # (NOT A and B) or C
    original_outputs.append(result)

# Learned model results
learned_inputs = torch.tensor(all_inputs, dtype=torch.float32, device=device)
with torch.no_grad():
    bacon.assembler.eval()
    learned_outputs = bacon.assembler(learned_inputs).squeeze().cpu().numpy()
    learned_outputs = (learned_outputs > 0.5).astype(int)

# Compare each case
print("\nA B C | Original | Learned | Match")
print("-" * 40)
for i, (a, b, c) in enumerate(all_inputs):
    orig = int(original_outputs[i])
    learned = learned_outputs[i]
    match = "✅" if orig == learned else "❌"
    print(f"{a} {b} {c} |    {orig}     |    {learned}    | {match}")

# Summary
matches = sum(1 for orig, learned in zip(original_outputs, learned_outputs) if orig == learned)
equivalence_rate = matches / len(all_inputs) * 100

print("\n" + "=" * 70)
print(f"✅ Tested {len(all_inputs)} input combinations")
print(f"✅ Logical equivalence: {matches}/{len(all_inputs)} ({equivalence_rate:.1f}%)")

if equivalence_rate == 100:
    print("✅ VERIFIED: Learned expression is logically equivalent to original!")
    print("   The model discovered a correct representation, even if transformations differ.")
else:
    print(f"⚠️  WARNING: Only {equivalence_rate:.1f}% equivalent")

print("=" * 70)
