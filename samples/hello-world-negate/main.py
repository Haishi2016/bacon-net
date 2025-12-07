# Note: required to import baconNet from local folder
import sys
sys.path.insert(0, '../../')

import torch
from bacon.baconNet import baconNet
from bacon.visualization import visualize_tree_structure, print_tree_structure
import logging
import random
import itertools


logging.basicConfig(level=logging.INFO, format='%(message)s')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def generate_boolean_data_with_negation(num_vars=3, repeat_factor=100, randomize=True, negation_prob=0.3, device=None):
    """
    Generate a dataset for boolean expressions that may include negations.
    
    Args:
        num_vars (int): Number of boolean variables to use
        repeat_factor (int): How many times to repeat the truth table (or number of samples if randomized)
        randomize (bool): If True, randomize the order of variable combinations
        negation_prob (float): Probability that a variable will be negated in the expression
        device (torch.device): Device to store the generated tensors
        
    Returns:
        tuple: (X, Y, metadata_dict)
    """
    logging.info("🧠 Generating boolean data with negation...")
    assert num_vars >= 2, "Need at least 2 variables for expression."
    
    data = []
    labels = []
    
    # Generate random operators between variables
    ops = [random.choice(["and", "or"]) for _ in range(num_vars - 1)]
    
    # Generate variable names: A-Z, then emojis for 26+
    # This supports up to 26 + many emojis (avoiding ASCII range issues)
    def get_var_name(idx):
        if idx < 26:
            return chr(ord('A') + idx)
        else:
            # Use emoji range starting from index 26
            # Common emoji ranges that work well: 🌀-🌿, 🍀-🍺, 🎀-🎯, 🏀-🏺, 🐀-🐿, 👀-👿, 💀-💿
            emoji_idx = idx - 26
            emoji_start = 0x1F300  # 🌀
            return chr(emoji_start + emoji_idx)
    
    var_names = [get_var_name(i) for i in range(num_vars)]
    
    # Decide which variables should be negated
    is_negated = [random.random() < negation_prob for _ in range(num_vars)]
    
    # Build symbolic expression and evaluation expression
    def var_expr(idx):
        """Get the variable expression (with or without negation)"""
        base_var = var_names[idx]
        if is_negated[idx]:
            return f"NOT {base_var}", f"(1 - x[{idx}])"
        else:
            return base_var, f"x[{idx}]"
    
    # Build the expression
    symbolic_parts = []
    
    for i in range(num_vars):
        base_var = var_names[i]
        if is_negated[i]:
            symbolic_parts.append(f"NOT {base_var}")
        else:
            symbolic_parts.append(base_var)
    
    # Combine with operators to build symbolic expression (for display)
    symbolic_expr = symbolic_parts[0]
    for i in range(1, num_vars):
        op = ops[i - 1]
        symbolic_expr = f"({symbolic_expr} {op} {symbolic_parts[i]})"
    
    # For evaluation, compute directly instead of building nested string expression
    def evaluate_expr(x):
        """Evaluate the expression directly without building a nested string"""
        # Start with first variable
        if is_negated[0]:
            result = 1 - x[0]
        else:
            result = x[0]
        
        # Apply operators sequentially
        for i in range(1, num_vars):
            if is_negated[i]:
                operand = 1 - x[i]
            else:
                operand = x[i]
            
            if ops[i - 1] == "and":
                result = result and operand
            else:  # "or"
                result = result or operand
        
        return int(result)
    
    if randomize:
        logging.info("⚡ Randomized input generation mode enabled.")
        num_samples = repeat_factor
        for _ in range(num_samples):
            x = [random.randint(0, 1) for _ in range(num_vars)]
            y = evaluate_expr(x)
            data.append(x)
            labels.append([y])
    else:
        # Generate all possible combinations
        base_cases = list(itertools.product([0, 1], repeat=num_vars))
        for _ in range(repeat_factor):
            for x in base_cases:
                y = evaluate_expr(list(x))
                data.append(list(x))
                labels.append([y])
    
    return (
        torch.tensor(data, dtype=torch.float32, device=device),
        torch.tensor(labels, dtype=torch.float32, device=device),
        {
            "expression_text": symbolic_expr,
            "ops": ops,
            "num_vars": num_vars,
            "var_names": var_names,
            "is_negated": is_negated
        }
    )


# Generate data with negation
input_size = 1000
group_size = 200
x, y, expr_info = generate_boolean_data_with_negation(
    input_size, 
    repeat_factor=100, 
    randomize=True, 
    negation_prob=0.4,  # 40% chance each variable is negated
    device=device
)

print(f"➗ Expression: {expr_info['expression_text']}")
print(f"🔢 Variables: {', '.join(expr_info['var_names'])}")
negated_vars = [expr_info['var_names'][i] for i, neg in enumerate(expr_info['is_negated']) if neg]
if negated_vars:
    print(f"🔄 Negated variables: {', '.join(negated_vars)}")
else:
    print(f"🔄 No negated variables")

# Create BACON model with transformation layer
# The transformation layer can learn to negate inputs, which should help
# discover the correct expression
#
# For larger input sizes (10+), we limit max_permutations to speed up training
# while still allowing the model to find good solutions
#
# early_stop_threshold_large_inputs controls when to stop training after finding
# a good permutation. Lower values = more training = higher accuracy.
# Default 0.1 is a good balance. Use 0.05 for higher accuracy, 0.2 for faster training.
#
# Adaptive reheating helps escape local optima:
# - reheat_plateau_window: epochs to check for plateau (smaller = more aggressive)
# - reheat_improvement_threshold: min improvement to avoid reheat (smaller = more aggressive)
# - reheat_cooldown: min epochs between reheats (prevents oscillation)
# - reheat_temperature: temperature when reheating (higher = more exploration)
#
# Temperature annealing (when using Sinkhorn permutations):
# - permutation_initial_temperature: start hot for broad exploration (default 5.0)
# - permutation_final_temperature: end cool for hard commitment (default 0.1)
# - transformation_initial_temperature: start cooler since transformation is simpler (default 1.0)
# - transformation_final_temperature: end same as permutation (default 0.1)
bacon = baconNet(
    input_size, 
    aggregator='bool.min_max', 
    weight_mode='fixed', 
    loss_amplifier=1000, 
    normalize_andness=False,
    use_transformation_layer=True,  # Enable transformation layer
    transformation_temperature=1.0,
    transformation_use_gumbel=False,
    max_permutations=100 if input_size >= 10 else None,  # Speed up for large inputs
    early_stop_threshold_large_inputs=0.05,  # Balance between speed and accuracy
    # Adaptive reheating for steep landscapes
    reheat_plateau_window=100,      # Aggressive: check last 100 epochs
    reheat_improvement_threshold=0.5,  # Aggressive: reheat if < 0.5 improvement
    reheat_cooldown=200,            # Allow reheating every 200 epochs
    reheat_temperature=10.0,        # High temp for strong exploration
    # Temperature annealing schedule
    permutation_initial_temperature=5.0,   # Start hot for permutation exploration
    permutation_final_temperature=0.1,     # Cool to hard permutation
    transformation_initial_temperature=1.0,  # Start cooler (simpler: 2^n vs n! states)
    transformation_final_temperature=0.1   # Same final temp
)

print("\n🎯 Training with transformation layer enabled...")
print("   The model can learn to negate features if needed.\n")

# Train the model using hierarchical permutation strategy
# For 10 inputs with group_size=3: 10÷3 = 4 groups → 4! = 24 coarse permutations
# Each coarse permutation initializes the full 10×10 matrix with block structure
# This gives us 24 structured attempts instead of random initialization
#
# HIERARCHICAL PERMUTATION STRATEGY:
#   - Enumerate ALL hard permutations at coarse level (feasible: 3!=6, 4!=24, 5!=120)
#   - Each coarse perm gives block-structured initialization of full matrix
#   - Train each for subset of epochs
#   - Much better coverage of permutation space than random restarts
#
# Example for 10 inputs, group_size=3:
#   - 4 groups (10÷3 rounded up)
#   - 4! = 24 possible coarse permutations
#   - Each trains for epochs_per_attempt
#   - Total computation = 24 * epochs_per_attempt
#
# Reduced epochs: Wrong coarse permutations won't converge anyway,
# so give each less time to avoid wasting computation on bad choices

max_epochs_value = min(input_size * 1000, 10000) if input_size >= 10 else min(input_size * 1500, 12000)
epochs_per_coarse_perm = 4000  # Each coarse permutation gets 1000 epochs (reduced from 2000)

(best_model, best_accuracy) = bacon.find_best_model(
    x, y, x, y, 
    acceptance_threshold=0.95, 
    max_epochs=max_epochs_value,  # Not used in hierarchical mode
    save_model=False,
    use_hierarchical_permutation=True,
    hierarchical_group_size=group_size,
    hierarchical_epochs_per_attempt=epochs_per_coarse_perm
)

print(f"\n🏆 Best accuracy: {best_accuracy * 100:.2f}%")

# Analyze transformation layer
if bacon.assembler.transformation_layer is not None:
    print("\n🔄 Transformation Layer Analysis")
    print("=" * 70)
    
    trans_summary = bacon.assembler.transformation_layer.get_transformation_summary()
    selected_transforms = bacon.assembler.transformation_layer.get_selected_transformations()
    
    print(f"\n📊 Learned Transformations:")
    for i, var_name in enumerate(expr_info['var_names']):
        trans_type = trans_summary[i]['transformation']
        confidence = trans_summary[i]['probability'] * 100
        
        # Check if the model learned the correct transformation
        expected_negation = expr_info['is_negated'][i]
        learned_negation = (trans_type == 'negation')
        
        status = "✅" if learned_negation == expected_negation else "❌"
        
        print(f"   {status} {var_name}: {trans_type:10s} (confidence: {confidence:5.1f}%) ", end="")
        
        if expected_negation:
            print(f"[Expected: negation]")
        else:
            print(f"[Expected: identity]")
    
    # Summary
    correct_count = sum(
        1 for i in range(input_size) 
        if (trans_summary[i]['transformation'] == 'negation') == expr_info['is_negated'][i]
    )
    
    print(f"\n📈 Transformation Accuracy: {correct_count}/{input_size} variables correct")
    print("=" * 70)

# Print tree structure with transformations applied
print("\n🌳 Logical Tree Structure:")
print_tree_structure(bacon.assembler, expr_info['var_names'], classic_boolean=True)

# Verify logical equivalence
print("\n🔍 Logical Equivalence Verification")
print("=" * 70)

# Generate all possible input combinations
import itertools
num_test_cases = 2 ** input_size
print(f"📊 Testing all {num_test_cases} possible input combinations...")

# Skip verification if too many combinations
if num_test_cases > 2048:
    print(f"⚠️  Skipping verification: {num_test_cases} combinations is too many")
    print(f"   (For {input_size} inputs, exhaustive testing would take too long)")
else:
    all_inputs = list(itertools.product([0, 1], repeat=input_size))
    
    # Evaluate original expression
    print("🔄 Evaluating original expression...")
    original_outputs = []
    for test_input in all_inputs:
        # Pass the input as 'x' for the eval expression
        x = list(test_input)
        result = eval(expr_info['eval_expr'], {"__builtins__": {}}, {"x": x})
        original_outputs.append(result)
    
    # Evaluate learned model
    print("🔄 Evaluating learned model...")
    learned_inputs = torch.tensor(all_inputs, dtype=torch.float32, device=device)
    with torch.no_grad():
        bacon.assembler.eval()
        learned_outputs = bacon.assembler(learned_inputs).squeeze().cpu().numpy()
        learned_outputs = (learned_outputs > 0.5).astype(int)
    
    # Compare
    print("🔄 Comparing results...")
    matches = sum(1 for orig, learned in zip(original_outputs, learned_outputs) if orig == learned)
    equivalence_rate = matches / num_test_cases * 100
    
    print(f"✅ Tested {num_test_cases} input combinations")
    print(f"✅ Logical equivalence: {matches}/{num_test_cases} ({equivalence_rate:.1f}%)")
    
    if equivalence_rate == 100:
        print("✅ VERIFIED: Learned expression is logically equivalent to original!")
        print("   (Even though some transformations differ, the permutation compensates)")
    else:
        print(f"⚠️  WARNING: Only {equivalence_rate:.1f}% equivalent - model may have errors")

print("=" * 70)

# Visualize
visualize_tree_structure(bacon.assembler, expr_info['var_names'])
