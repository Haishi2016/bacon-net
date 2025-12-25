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
    # A-Z for first 26, then 1000+ unique emojis for the rest
    emojis = [
        # Geometric shapes & symbols (20)
        '🔴', '🟠', '🟡', '🟢', '🔵', '🟣', '🟤', '⚫', '⚪', '🔶', '🔷', '🔸', '🔹', '🔺', '🔻', '💠', '🔘', '🔳', '🔲', '◼️',
        # Stars & weather (40)
        '⭐', '🌟', '✨', '💫', '🌙', '🌛', '🌜', '🌚', '🌝', '🌞', '☀️', '⛅', '⛈️', '🌤️', '🌦️', '🌧️', '🌨️', '🌩️', '🌪️', '🌫️',
        '☃️', '⛄', '❄️', '🌬️', '💨', '🌊', '💧', '💦', '☔', '⚡', '🌈', '🌅', '🌄', '🌠', '🌌', '🌃', '🌆', '🌇', '🏙️', '🌉',
        # Fruits (40)
        '🍎', '🍏', '🍊', '🍋', '🍌', '🍉', '🍇', '🍓', '🫐', '🍈', '🍒', '🍑', '🥭', '🍍', '🥥', '🥝', '🍅', '🥑', '🌶️', '🫑',
        '🥒', '🥬', '🥦', '🧄', '🧅', '🌽', '🥕', '🥔', '🍠', '🥐', '🥖', '🥨', '🥯', '🧀', '🥚', '🍳', '🧈', '🥞', '🧇', '🥓',
        # More food (40)
        '🍗', '🍖', '🌭', '🍔', '🍟', '🍕', '🫓', '🥪', '🥙', '🧆', '🌮', '🌯', '🫔', '🥗', '🥘', '🫕', '🥫', '🍝', '🍜', '🍲',
        '🍛', '🍣', '🍱', '🥟', '🦪', '🍤', '🍙', '🍚', '🍘', '🍥', '🥠', '🥮', '🍢', '🍡', '🍧', '🍨', '🍦', '🥧', '🧁', '🍰',
        # Desserts & drinks (40)
        '🎂', '🍮', '🍭', '🍬', '🍫', '🍿', '🍩', '🍪', '🌰', '🥜', '🍯', '🥛', '🍼', '🫖', '☕', '🍵', '🧃', '🥤', '🧋', '🍶',
        '🍺', '🍻', '🥂', '🍷', '🥃', '🍸', '🍹', '🧉', '🍾', '🧊', '🥄', '🍴', '🍽️', '🥣', '🥡', '🥢', '🧂', '🫗', '🍱', '🥟',
        # Animals & nature (80)
        '🐶', '🐱', '🐭', '🐹', '🐰', '🦊', '🐻', '🐼', '🐨', '🐯', '🦁', '🐮', '🐷', '🐽', '🐸', '🐵', '🙈', '🙉', '🙊', '🐒',
        '🐔', '🐧', '🐦', '🐤', '🐣', '🐥', '🦆', '🦅', '🦉', '🦇', '🐺', '🐗', '🐴', '🦄', '🐝', '🪱', '🐛', '🦋', '🐌', '🐞',
        '🐜', '🪰', '🪲', '🪳', '🦟', '🦗', '🕷️', '🕸️', '🦂', '🐢', '🐍', '🦎', '🦖', '🦕', '🐙', '🦑', '🦐', '🦞', '🦀', '🐡',
        '🐠', '🐟', '🐬', '🐳', '🐋', '🦈', '🐊', '🐅', '🐆', '🦓', '🦍', '🦧', '🦣', '🐘', '🦛', '🦏', '🐪', '🐫', '🦒', '🦘',
        # More animals (40)
        '🦬', '🐃', '🐂', '🐄', '🐎', '🐖', '🐏', '🐑', '🦙', '🐐', '🦌', '🐕', '🐩', '🦮', '🐕‍🦺', '🐈', '🐈‍⬛', '🪶', '🐓', '🦃',
        '🦤', '🦚', '🦜', '🦢', '🦩', '🕊️', '🐇', '🦝', '🦨', '🦡', '🦫', '🦦', '🦥', '🐁', '🐀', '🐿️', '🦔', '🐾', '🐉', '🐲',
        # Plants & flowers (40)
        '🌵', '🎄', '🌲', '🌳', '🌴', '🪵', '🌱', '🌿', '☘️', '🍀', '🎍', '🪴', '🎋', '🍃', '🍂', '🍁', '🪺', '🪹', '🌾', '💐',
        '🌷', '🌹', '🥀', '🌺', '🌸', '🌼', '🌻', '🏵️', '🌕', '🌖', '🌗', '🌘', '🌑', '🌒', '🌓', '🌔', '🪐', '⭐', '🌟', '💫',
        # Objects & activities (80)
        '⚽', '⚾', '🥎', '🏀', '🏐', '🏈', '🏉', '🎾', '🥏', '🎳', '🏏', '🏑', '🏒', '🥍', '🏓', '🏸', '🥊', '🥋', '🥅', '⛳',
        '⛸️', '🎣', '🤿', '🎽', '🎿', '🛷', '🥌', '🎯', '🪀', '🪁', '🎱', '🔮', '🪄', '🧿', '🪬', '🎮', '🕹️', '🎰', '🎲', '🧩',
        '🧸', '🪅', '🪩', '🪆', '♠️', '♥️', '♦️', '♣️', '♟️', '🃏', '🀄', '🎴', '🎭', '🖼️', '🎨', '🧵', '🪡', '🧶', '🪢', '👓',
        '🕶️', '🥽', '🥼', '🦺', '👔', '👕', '👖', '🧣', '🧤', '🧥', '🧦', '👗', '👘', '🥻', '🩱', '🩲', '🩳', '👙', '👚', '👛',
        # More objects (80)
        '👜', '👝', '🛍️', '🎒', '🩴', '👞', '👟', '🥾', '🥿', '👠', '👡', '🩰', '👢', '👑', '👒', '🎩', '🎓', '🧢', '🪖', '⛑️',
        '📿', '💄', '💍', '💎', '🔇', '🔈', '🔉', '🔊', '📢', '📣', '📯', '🔔', '🔕', '🎼', '🎵', '🎶', '🎙️', '🎚️', '🎛️', '🎤',
        '🎧', '📻', '🎷', '🪗', '🎸', '🎹', '🎺', '🎻', '🪕', '🥁', '🪘', '📱', '📲', '☎️', '📞', '📟', '📠', '🔋', '🪫', '🔌',
        '💻', '🖥️', '🖨️', '⌨️', '🖱️', '🖲️', '💽', '💾', '💿', '📀', '🧮', '🎥', '🎞️', '📽️', '🎬', '📺', '📷', '📸', '📹', '📼',
        # Tools & tech (80)
        '🔍', '🔎', '🕯️', '💡', '🔦', '🏮', '🪔', '📔', '📕', '📖', '📗', '📘', '📙', '📚', '📓', '📒', '📃', '📜', '📄', '📰',
        '🗞️', '📑', '🔖', '🏷️', '💰', '🪙', '💴', '💵', '💶', '💷', '💸', '💳', '🧾', '💹', '✉️', '📧', '📨', '📩', '📤', '📥',
        '📦', '📫', '📪', '📬', '📭', '📮', '🗳️', '✏️', '✒️', '🖋️', '🖊️', '🖌️', '🖍️', '📝', '💼', '📁', '📂', '🗂️', '📅', '📆',
        '🗒️', '🗓️', '📇', '📈', '📉', '📊', '📋', '📌', '📍', '📎', '🖇️', '📏', '📐', '✂️', '🗃️', '🗄️', '🗑️', '🔒', '🔓', '🔏',
        # Misc symbols (74+)
        '🔐', '🔑', '🗝️', '🔨', '🪓', '⛏️', '⚒️', '🛠️', '🗡️', '⚔️', '🔫', '🪃', '🏹', '🛡️', '🪚', '🔧', '🪛', '🔩', '⚙️', '🗜️',
        '⚖️', '🦯', '🔗', '⛓️', '🪝', '🧰', '🧲', '🪜', '⚗️', '🧪', '🧫', '🧬', '🔬', '🔭', '📡', '💉', '🩸', '💊', '🩹', '🩼',
        '🩺', '🩻', '🚪', '🛗', '🪞', '🪟', '🛏️', '🛋️', '🪑', '🚽', '🪠', '🚿', '🛁', '🪤', '🪒', '🧴', '🧷', '🧹', '🧺', '🧻',
        '🪣', '🧼', '🫧', '🪥', '🧽', '🧯', '🛒', '🚬', '⚰️', '🪦', '⚱️', '🗿', '🪧', '🪪'
    ]
    
    def get_var_name(i):
        if i < 26:
            return chr(ord('A') + i)
        elif i - 26 < len(emojis):
            return emojis[i - 26]
        else:
            # Fallback for >1000 variables: cycle emojis with numbers
            emoji_index = (i - 26) % len(emojis)
            repeat = (i - 26) // len(emojis)
            return emojis[emoji_index] + str(repeat + 1)
    
    var_names = [get_var_name(i) for i in range(num_vars)]
    symbolic_expr = var_names[0]
    for i in range(1, num_vars):
        op = ops[i - 1]
        symbolic_expr = f"({symbolic_expr} {op} {var_names[i]})"
    
    # Evaluation function that avoids deeply nested eval() for large inputs
    def evaluate_expression(x):
        """Evaluate boolean expression iteratively to avoid stack overflow.
        
        Matches the eval() behavior with nested parentheses: ((((A and B) or C) and D)...)
        This means RIGHTMOST variables have HIGHEST priority (evaluated first conceptually).
        We build from right to left to match the original parentheses structure.
        """
        # Start from the rightmost variable and work backwards
        result = bool(x[num_vars - 1])
        for i in range(num_vars - 2, -1, -1):  # Go backwards from second-to-last to first
            if ops[i] == "and":
                result = bool(x[i]) and result
            else:  # "or"
                result = bool(x[i]) or result
        return int(result)

    if randomize:
        logging.info("⚡ Randomized input generation mode enabled.")
        num_samples = repeat_factor  # reinterpret repeat_factor as sample count
        for _ in range(num_samples):
            x = [random.randint(0, 1) for _ in range(num_vars)]
            y = evaluate_expression(x)
            data.append(x)
            labels.append([y])
    else:
        base_cases = list(itertools.product([0, 1], repeat=num_vars))
        for _ in range(repeat_factor):
            for x in base_cases:
                y = evaluate_expression(x)
                data.append(list(x))
                labels.append([y])
    return (
        torch.tensor(data, dtype=torch.float32, device=device),
        torch.tensor(labels, dtype=torch.float32, device=device),
        {
            "expression_text": symbolic_expr,
            "ops": ops,
            "num_vars": num_vars,
            "var_names": var_names
        }
    )


def generate_paired_boolean_data(num_vars=4, repeat_factor=100, randomize=False, device=None):
    """Generate dataset for a two-layer paired boolean expression.

    Layer 1: Pair adjacent variables (A,B), (C,D), ... with random AND/OR per pair.
    Layer 2: Fold the pair-outputs with random AND/OR between them (left fold).

    Args:
        num_vars (int): Number of boolean variables.
        repeat_factor (int): Repeats/allocation similar to generate_classic_boolean_data.
        randomize (bool): If True, sample inputs randomly; else use full truth table repeated.
        device (torch.device): Device to store tensors.
    Returns:
        tuple[Tensor, Tensor, dict]: inputs X [N,num_vars], labels y [N,1], metadata dict.
    """
    logging.info("🧠 Generating paired-structure boolean data...")
    assert num_vars >= 2, "Need at least 2 variables for expression."

    var_names = [chr(ord('A') + i) for i in range(num_vars)]

    # Layer 1: pairwise ops
    pair_syms = []
    pair_evals = []
    pair_ops = []
    j = 0
    while j < num_vars:
        if j + 1 < num_vars:
            op = random.choice(["and", "or"])  # per-pair op
            pair_ops.append(op)
            pair_syms.append(f"({var_names[j]} {op} {var_names[j+1]})")
            pair_evals.append(f"(x[{j}] {op} x[{j+1}])")
            j += 2
        else:
            # odd leftover passes through
            pair_syms.append(var_names[j])
            pair_evals.append(f"x[{j}]")
            j += 1

    # Layer 2: fold pair outputs
    fold_ops = []
    symbolic_expr = pair_syms[0]
    eval_expr = pair_evals[0]
    for i in range(1, len(pair_syms)):
        op2 = random.choice(["and", "or"])  # op between pairs
        fold_ops.append(op2)
        symbolic_expr = f"({symbolic_expr} {op2} {pair_syms[i]})"
        eval_expr = f"({eval_expr} {op2} {pair_evals[i]})"

    data = []
    labels = []
    if randomize:
        logging.info("⚡ Randomized input generation mode enabled (paired).")
        num_samples = repeat_factor
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
            "pair_ops": pair_ops,
            "fold_ops": fold_ops,
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


def analyze_feature_importance_with_pruning(
    model, 
    X, 
    Y, 
    feature_names, 
    threshold=0.5,
    baseline_enabled=False,
    baseline_drop_threshold=0.05,
    device=None
):
    """Analyze feature importance through cumulative pruning.
    
    Args:
        model: Trained baconNet model with frozen structure
        X: Input tensor
        Y: Target tensor
        feature_names: List of feature names
        threshold: Classification threshold for accuracy calculation
        baseline_enabled: If True, detect and skip baseline features
        baseline_drop_threshold: Minimum accuracy drop to consider a feature as baseline
        device: torch device
        
    Returns:
        dict: {
            'accuracies': List of accuracies after pruning 0, 1, 2, ... features,
            'baseline_features': List of feature indices that form the baseline (empty if baseline_enabled=False),
            'baseline_feature_names': List of feature names in baseline,
            'num_features_pruned': Number of features actually pruned (excluding baseline)
        }
    """
    if device is None:
        device = X.device
    
    assembler = model.assembler
    num_features = len(feature_names)
    
    # Calculate baseline accuracy
    with torch.no_grad():
        baseline_output = assembler(X)
        baseline_accuracy = ((baseline_output > threshold).float() == Y).float().mean().item()
    
    accuracies = [baseline_accuracy]
    baseline_features = []
    
    # Save original weights
    original_weights = [w.data.clone() for w in assembler.weights]
    
    # Detect baseline if enabled
    if baseline_enabled:
        print("🔍 Detecting baseline features...")
        for i in range(num_features):
            # Restore weights
            for j, w in enumerate(assembler.weights):
                w.data.copy_(original_weights[j])
            assembler.pruned_aggregators.clear()
            
            # Check if pruning feature i causes significant drop
            if i < len(assembler.weights):
                assembler.prune_features(i)
                
                with torch.no_grad():
                    pruned_output = assembler(X)
                    pruned_accuracy = ((pruned_output > threshold).float() == Y).float().mean().item()
                    drop = baseline_accuracy - pruned_accuracy
                
                if drop >= baseline_drop_threshold:
                    baseline_features.append(i)
                    feature_idx = assembler.locked_perm[i].item()
                    print(f"   ✅ Baseline feature {i}: {feature_names[feature_idx]} (drop: {drop*100:.2f}%)")
                else:
                    # Baseline chain breaks at first non-significant feature
                    break
        
        if len(baseline_features) > 0:
            print(f"📌 Baseline detected: {len(baseline_features)} features will NOT be pruned")
        else:
            print("📌 No baseline detected")
    
    # Cumulative pruning analysis (skip baseline features)
    print("\n🔬 Cumulative pruning analysis:")
    for i in range(1, num_features):
        # Restore original weights
        for j, w in enumerate(assembler.weights):
            w.data.copy_(original_weights[j])
        assembler.pruned_aggregators.clear()
        
        # Apply cumulative pruning: prune features 0 through i-1, EXCEPT baseline
        for k in range(i):
            if k not in baseline_features:  # Skip baseline features
                assembler.prune_features(k)
        
        with torch.no_grad():
            pruned_output = assembler(X)
            pruned_accuracy = ((pruned_output > threshold).float() == Y).float().mean().item()
            accuracies.append(pruned_accuracy)
            
            baseline_note = f" (baseline: {len(baseline_features)})" if baseline_enabled and len(baseline_features) > 0 else ""
            print(f"   Accuracy after pruning {i} feature(s){baseline_note}: {pruned_accuracy * 100:.2f}%")
    
    # Restore original weights
    for j, w in enumerate(assembler.weights):
        w.data.copy_(original_weights[j])
    assembler.pruned_aggregators.clear()
    
    baseline_feature_names = [feature_names[assembler.locked_perm[i].item()] for i in baseline_features]
    
    return {
        'accuracies': accuracies,
        'baseline_features': baseline_features,
        'baseline_feature_names': baseline_feature_names,
        'num_features_pruned': num_features - len(baseline_features) - 1  # -1 for the last feature
    }


def export_tree_structure_to_json(model, feature_names=None):
    """Export the binary tree structure to a JSON-serializable dictionary.
    
    Args:
        model: The binaryTreeLogicNet model
        feature_names (list, optional): List of feature names. If None, uses feature0, feature1, etc.
        
    Returns:
        dict: JSON-serializable dictionary representing the tree structure
    """
    import json
    
    if feature_names is not None and len(feature_names) < model.num_leaves:
        raise ValueError(f"Feature name count {len(feature_names)} doesn't match number of leaves {model.num_leaves}")
    
    # Get feature names with permutation applied
    if feature_names:
        if model.locked_perm is not None:
            leaf_names = [feature_names[i] for i in model.locked_perm.tolist()]
        else:
            leaf_names = feature_names
    else:
        leaf_names = [f"feature{i}" for i in range(model.num_leaves)]
    
    # Apply transformations to leaf names
    if hasattr(model, 'transformation_layer') and model.transformation_layer is not None:
        selected_transforms = model.transformation_layer.get_selected_transformations()
        # Get transformation names from the transformation objects
        transformation_names = []
        for transform in model.transformation_layer.transformations:
            name = transform.__class__.__name__.replace('Transformation', '').lower()
            transformation_names.append(name)
        
        original_leaf_names = leaf_names.copy()
        leaf_names = []
        transformations_applied = []
        for i, name in enumerate(original_leaf_names):
            transform_idx = selected_transforms[i].item()
            transform_name = transformation_names[transform_idx]
            transformations_applied.append(transform_name)
            if transform_name == 'negation':  # negation
                leaf_names.append(f"NOT {name}")
            else:
                leaf_names.append(name)
    else:
        transformations_applied = ["identity"] * len(leaf_names)
    
    # Extract weights and biases
    if model.weight_mode == 'fixed' or model.weight_normalization == 'minmax':
        weights = [w.detach().cpu().tolist() if hasattr(w, 'detach') else w for w in model.weights]
    else:
        weights = [F.softmax(w.detach().cpu(), dim=0).tolist() for w in model.weights]
    
    # Calculate andness values (a-values)
    a_vals = [(torch.sigmoid(b) * 3 - 1).item() for b in model.biases]
    
    # Build tree structure based on layout
    effective_layout = getattr(model, 'tree_layout', 'left')
    
    tree_structure = {
        "model_type": "binaryTreeLogicNet",
        "layout": effective_layout,
        "num_features": model.num_leaves,
        "num_layers": model.num_layers,
        "features": []
    }
    
    # Add feature information with transformations
    for i, (name, orig_name, transform) in enumerate(zip(leaf_names, 
                                                           feature_names if feature_names else leaf_names, 
                                                           transformations_applied)):
        tree_structure["features"].append({
            "index": i,
            "original_name": orig_name if feature_names else f"feature{i}",
            "display_name": name,
            "transformation": transform
        })
    
    # Build node structure based on layout
    if effective_layout == 'left':
        # Left-associative tree
        nodes = []
        for i in range(model.num_layers):
            node = {
                "layer": i,
                "aggregator_index": i,
                "andness": round(a_vals[i], 6),
                # "operator": "AND" if a_vals[i] >= 0.5 else "OR",
                "weights": {
                    "left": round(weights[i][0], 6),
                    "right": round(weights[i][1], 6)
                }
            }
            
            if i == 0:
                node["left_input"] = {"type": "feature", "index": 0, "name": leaf_names[0]}
                node["right_input"] = {"type": "feature", "index": 1, "name": leaf_names[1]}
            else:
                node["left_input"] = {"type": "aggregator", "layer": i-1}
                node["right_input"] = {"type": "feature", "index": i+1, "name": leaf_names[i+1]}
            
            nodes.append(node)
        
        tree_structure["nodes"] = nodes
        
    elif effective_layout == 'balanced':
        # Balanced binary tree
        def build_balanced_structure(start, end, idx):
            if start == end:
                return {"type": "feature", "index": start, "name": leaf_names[start]}, idx
            
            mid = (start + end) // 2
            left_tree, idx = build_balanced_structure(start, mid, idx)
            right_tree, idx = build_balanced_structure(mid + 1, end, idx)
            
            node = {
                "type": "aggregator",
                "layer": idx,
                "andness": round(a_vals[idx], 6),
                "operator": "AND" if a_vals[idx] >= 0.5 else "OR",
                "weights": {
                    "left": round(weights[idx][0], 6),
                    "right": round(weights[idx][1], 6)
                },
                "left_input": left_tree,
                "right_input": right_tree
            }
            
            return node, idx + 1
        
        root, _ = build_balanced_structure(0, model.num_leaves - 1, 0)
        tree_structure["root"] = root
        
    elif effective_layout == 'paired':
        # Paired tree: pair features first, then aggregate pairs
        nodes = []
        idx = 0
        pairs = []
        j = 0
        
        # First phase: pair adjacent features
        while j < model.num_leaves:
            if j + 1 < model.num_leaves:
                node = {
                    "layer": idx,
                    "aggregator_index": idx,
                    "phase": "pairing",
                    "andness": round(a_vals[idx], 6),
                    "operator": "AND" if a_vals[idx] >= 0.5 else "OR",
                    "weights": {
                        "left": round(weights[idx][0], 6),
                        "right": round(weights[idx][1], 6)
                    },
                    "left_input": {"type": "feature", "index": j, "name": leaf_names[j]},
                    "right_input": {"type": "feature", "index": j+1, "name": leaf_names[j+1]}
                }
                pairs.append({"type": "aggregator", "index": idx})
                nodes.append(node)
                idx += 1
                j += 2
            else:
                pairs.append({"type": "feature", "index": j, "name": leaf_names[j]})
                j += 1
        
        # Second phase: fold pairs left-associatively
        for k in range(1, len(pairs)):
            node = {
                "layer": idx,
                "aggregator_index": idx,
                "phase": "folding",
                "andness": round(a_vals[idx], 6),
                "operator": "AND" if a_vals[idx] >= 0.5 else "OR",
                "weights": {
                    "left": round(weights[idx][0], 6),
                    "right": round(weights[idx][1], 6)
                },
                "left_input": pairs[k-1] if k == 1 else {"type": "aggregator", "index": idx-1},
                "right_input": pairs[k]
            }
            nodes.append(node)
            idx += 1
        
        tree_structure["nodes"] = nodes
    
    return tree_structure


def save_tree_structure_to_json(model, filename, feature_names=None):
    """Export the binary tree structure to a JSON file.
    
    Args:
        model: The binaryTreeLogicNet model
        filename (str): Path to save the JSON file
        feature_names (list, optional): List of feature names
        
    Returns:
        str: Path to the saved file
    """
    import json
    
    tree_structure = export_tree_structure_to_json(model, feature_names)
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(tree_structure, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Tree structure saved to: {filename}")
    return filename
