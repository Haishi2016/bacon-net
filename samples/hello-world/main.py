import sys
sys.path.append('../../')
import torch
from bacon.baconNet import baconNet
import logging
import itertools
import random

logging.basicConfig(level=logging.INFO,
                    format='%(message)s')

def generate_data(num_vars=5, repeat_factor=100):
    print("🧠 Generating data...")
    assert num_vars >= 2, "Need at least 2 variables for expression."
    data = []
    labels = []
    base_cases = list(itertools.product([0, 1], repeat=num_vars))

    # Step 1: generate stable ops per variable link
    ops = [random.choice(["and", "or"]) for _ in range(num_vars - 1)]

    # Step 2: generate variable names and build expression strings
    var_names = [chr(ord('A') + i) for i in range(num_vars)]
    symbolic_expr = var_names[0]
    eval_expr = "x[0]"
    for i in range(1, num_vars):
        op = ops[i - 1]
        symbolic_expr = f"({symbolic_expr} {op} {var_names[i]})"
        eval_expr = f"({eval_expr} {op} x[{i}])"

    # Step 3: evaluate the expression across the truth table
    for _ in range(repeat_factor):
        for x in base_cases:
            y = int(eval(eval_expr))
            data.append(list(x))
            labels.append([y])
    return (
        torch.tensor(data, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.float32),
        {
            "expression_text": symbolic_expr,
            "eval_expr": eval_expr,
            "ops": ops,
            "num_vars": num_vars,
            "var_names": var_names
        }
    )

x, y,  expr_info = generate_data(3, repeat_factor=100)
print(f"Expression: {expr_info['expression_text']}")
bacon = baconNet(input_size=3, freeze_loss_threshold=0.001)
(best_model, best_accuracy) = bacon.find_best_model(x, y, x, y, acceptance_threshold=0.95, attempts=100)
print(f"Best accuracy: {best_accuracy * 100:.2f}%")
bacon.print_tree_structure(expr_info['var_names'])
bacon.visualize_tree_structure(expr_info['var_names'])