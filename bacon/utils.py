import itertools
import random
import torch

def generate_classic_boolean_data(num_vars=5, repeat_factor=100, device=None):
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
        torch.tensor(data, dtype=torch.float32, device=device),
        torch.tensor(labels, dtype=torch.float32, device=device),
        {
            "expression_text": symbolic_expr,
            "eval_expr": eval_expr,
            "ops": ops,
            "num_vars": num_vars,
            "var_names": var_names
        }
    )