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
    
    # Generate variable names
    var_names = [chr(ord('A') + i) for i in range(num_vars)]
    
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
    eval_parts = []
    
    for i in range(num_vars):
        sym, ev = var_expr(i)
        symbolic_parts.append(sym)
        eval_parts.append(ev)
    
    # Combine with operators
    symbolic_expr = symbolic_parts[0]
    eval_expr = eval_parts[0]
    
    for i in range(1, num_vars):
        op = ops[i - 1]
        symbolic_expr = f"({symbolic_expr} {op} {symbolic_parts[i]})"
        eval_expr = f"({eval_expr} {op} {eval_parts[i]})"
    
    if randomize:
        logging.info("⚡ Randomized input generation mode enabled.")
        num_samples = repeat_factor
        for _ in range(num_samples):
            x = [random.randint(0, 1) for _ in range(num_vars)]
            y = int(eval(eval_expr))
            data.append(x)
            labels.append([y])
    else:
        # Generate all possible combinations
        base_cases = list(itertools.product([0, 1], repeat=num_vars))
        for _ in range(repeat_factor):
            for x in base_cases:
                y = int(eval(eval_expr))
                data.append(list(x))
                labels.append([y])
    
    return (
        torch.tensor(data, dtype=torch.float32, device=device),
        torch.tensor(labels, dtype=torch.float32, device=device),
        {
            "expression_text": symbolic_expr,
            "eval_expr": eval_expr,
            "ops": ops,
            "num_vars": num_vars,
            "var_names": var_names,
            "is_negated": is_negated
        }
    )


# Generate data with negation
input_size = 3
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
bacon = baconNet(
    input_size, 
    aggregator='bool.min_max', 
    weight_mode='fixed', 
    loss_amplifier=1000, 
    normalize_andness=False,
    use_transformation_layer=True,  # Enable transformation layer
    transformation_temperature=1.0,
    transformation_use_gumbel=False
)

print("\n🎯 Training with transformation layer enabled...")
print("   The model can learn to negate features if needed.\n")

# Train the model
(best_model, best_accuracy) = bacon.find_best_model(
    x, y, x, y, 
    acceptance_threshold=0.95, 
    attempts=10, 
    max_epochs=min(input_size * 1000, 8000), 
    save_model=False
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

# Visualize
visualize_tree_structure(bacon.assembler, expr_info['var_names'])
