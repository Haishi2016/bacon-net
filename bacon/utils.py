import itertools
import random
import torch
from sklearn.utils import resample
import pandas as pd
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
import logging
import torch.nn.functional as F

def generate_classic_boolean_data(num_vars=5, repeat_factor=100, randomize=False, device=None):
    """ Generate a dataset for classic boolean expressions with a specified number of variables.

    Args:
        num_vars (int): Number of boolean variables (A, B, C, etc.) to use in the expression.
        repeat_factor (int): How many times to repeat the truth table for each variable combination.
        randomize (bool): If True, randomize the order of variable combinations.
        device (torch.device): Device to store the generated tensors.
    Returns:
        tuple: A tuple containing:
            - torch.Tensor: Input data tensor of shape (num_cases, num_vars).
            - torch.Tensor: Output labels tensor of shape (num_cases, 1).
            - dict: Metadata dictionary with expression details.
    """
    logging.info("🧠 Generating data...")
    assert num_vars >= 2, "Need at least 2 variables for expression."
    data = []
    labels = []
    
    # generate stable ops per variable link
    ops = [random.choice(["and", "or"]) for _ in range(num_vars - 1)]

    # generate variable names and build expression strings
    var_names = [chr(ord('A') + i) for i in range(num_vars)]
    symbolic_expr = var_names[0]
    eval_expr = "x[0]"
    for i in range(1, num_vars):
        op = ops[i - 1]
        symbolic_expr = f"({symbolic_expr} {op} {var_names[i]})"
        eval_expr = f"({eval_expr} {op} x[{i}])"

    if randomize:
        logging.info("⚡ Randomized input generation mode enabled.")
        num_samples = repeat_factor  # reinterpret repeat_factor as sample count
        for _ in range(num_samples):
            x = [random.randint(0, 1) for _ in range(num_vars)]
            y = int(eval(eval_expr))
            data.append(x)
            labels.append([y])
    else:
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
            "var_names": var_names
        }
    )


def balance_classes(train_df, target_col='target', random_state=42, replication_factor=1):
    counts_str = train_df[target_col].value_counts().to_string()
    indented_counts = '\n'.join('   ' + line for line in counts_str.split('\n'))
    logging.info(f"🔻 Class distribution before balancing and replication (×{replication_factor}):\n{indented_counts}")

    # Get class distribution
    class_counts = train_df[target_col].value_counts()
    if len(class_counts) != 2:
        raise ValueError("This function only supports binary classification.")

    # Identify majority and minority classes
    majority_class = class_counts.idxmax()
    minority_class = class_counts.idxmin()

    df_majority = train_df[train_df[target_col] == majority_class]
    df_minority = train_df[train_df[target_col] == minority_class]

    # Upsample minority class
    df_minority_upsampled = resample(
        df_minority,
        replace=True,
        n_samples=len(df_majority),
        random_state=random_state
    )

    # Combine and shuffle
    df_balanced = pd.concat([df_majority, df_minority_upsampled])
    df_balanced = df_balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Replicate the balanced dataset
    if replication_factor > 1:
        df_balanced = pd.concat([df_balanced] * replication_factor, ignore_index=True)
        df_balanced = df_balanced.sample(frac=1, random_state=random_state).reset_index(drop=True)

    counts_str = df_balanced[target_col].value_counts().to_string()
    indented_counts = '\n'.join('   ' + line for line in counts_str.split('\n'))
    logging.info(f"⚖️ Class distribution after balancing and replication (×{replication_factor}):\n{indented_counts}")
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

def analyze_bacon_tree_conjunctive_disjunctive(model, balanced_threshold=(0.35, 0.65), compound_drop_threshold=0.1):
    """
    Analyze a BACON model's left-associative tree to classify each node as Conjunctive, Disjunctive, or Dominated.
    Handles 2-element weight tensors with the selected normalization method.
    Resets compounded weight when a node is decisively contributing.
    """
    results = []
    compounded = 1.0  # Start with 1 at root

    for i, (w_raw, a_raw) in enumerate(zip(model.weights, model.biases)):
        w_tensor = w_raw.detach().cpu()

        # Normalize weights
        if model.weight_normalization == 'softmax':
            w = F.softmax(w_tensor, dim=0)
        elif model.weight_normalization == 'minmax':
            w_min = w_tensor.min()
            w_max = w_tensor.max()
            denom = w_max - w_min
            if denom.item() == 0:
                w = torch.tensor([0.5, 0.5])
            else:
                w = (w_tensor - w_min) / denom
                w = w / w.sum()
        else:
            # No normalization or fixed weights
            w = torch.sigmoid((w_tensor - 0.5) * 4)  # mimic sharp sigmoid if used

        left_weight = w[0].item()
        right_weight = w[1].item()

        # Normalize andness bias
        if model.normalize_andness:
            bias_a = torch.sigmoid(a_raw.detach().cpu()) * 3 - 1
        else:
            bias_a = a_raw.detach().cpu()

        bias_a = bias_a.item()

        # Effective compounded weight for feature entering at right input
        feature_compounded_weight = compounded * right_weight

        # Decision logic
        if feature_compounded_weight < compound_drop_threshold:
            conclusion = "Dominated (structure suppresses)"
        elif right_weight < balanced_threshold[0]:
            conclusion = "Dominated (left dominates)"
        elif right_weight > balanced_threshold[1]:
            conclusion = "Conjunctive" if bias_a > 0.5 else "Disjunctive"
            compounded = 1.0
        else:
            conclusion = "Conjunctive" if bias_a > 0.5 else "Disjunctive (balanced)"
            compounded = 1.0

        results.append({
            'Node': f'Node{i}',
            'w (right)': round(right_weight, 4),
            '1-w (left)': round(left_weight, 4),
            'bias (a)': round(bias_a, 4),
            'compounded_weight': round(feature_compounded_weight, 4),
            'Conclusion': conclusion
        })

        # Update compounded logic
        if feature_compounded_weight >= compound_drop_threshold:
            compounded = 1.0
        else:
            compounded *= left_weight

    return pd.DataFrame(results)

class SigmoidScaler(BaseEstimator, TransformerMixin):
    def __init__(self, alpha=1.0, beta=0.0):
        self.alpha = alpha
        self.beta = beta
    def fit(self, X, y=None):
        X = np.asarray(X)
        self.mean_ = X.mean(axis=0)
        self.std_ = X.std(axis=0)
        self.std_[self.std_ == 0] = 1.0  # avoid division by zero
        return self

    def transform(self, X):
        X = np.asarray(X)
        X_centered = (X - self.mean_) / self.std_
        return 1 / (1 + np.exp(-self.alpha * (X_centered - self.beta)))
