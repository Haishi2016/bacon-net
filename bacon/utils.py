import itertools
import random
import torch
from sklearn.utils import resample
import pandas as pd

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

def balance_classes(train_df, target_col='target', random_state=42):
    df_majority = train_df[train_df[target_col] == 1]
    df_minority = train_df[train_df[target_col] == 0]

    df_minority_upsampled = resample(
        df_minority,
        replace=True,                 # Sample with replacement
        n_samples=len(df_majority),    # Match majority class size
        random_state=random_state
    )

    df_balanced = pd.concat([df_majority, df_minority_upsampled])
    df_balanced = df_balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)

    print(f"[INFO] Class distribution after balancing:\n{df_balanced[target_col].value_counts()}")
    return df_balanced

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

def find_best_threshold(model, X_val, Y_val, metric='accuracy', steps=1000):
    model.eval()
    with torch.no_grad():
        probs = model.inference_raw(X_val).cpu().numpy().flatten()
    true_labels = Y_val.cpu().numpy().flatten()

    if metric == 'recall':
        thresholds = np.linspace(1, 0, steps)
    else:
        thresholds = np.linspace(0, 1, steps)
    best_score = -1
    best_threshold = 0.5

    for threshold in thresholds:
        preds = (probs >= threshold).astype(int)
        
        if metric == 'accuracy':
            score = accuracy_score(true_labels, preds)
        elif metric == 'precision':
            score = precision_score(true_labels, preds, zero_division=0)
        elif metric == 'recall':
            score = recall_score(true_labels, preds, zero_division=0)
        elif metric == 'f1':
            score = f1_score(true_labels, preds, zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if score > best_score:
            best_score = score
            best_threshold = threshold

    return best_threshold, best_score
