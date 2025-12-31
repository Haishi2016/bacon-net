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
    
    # CRITICAL: Clear any existing pruning state from previous runs
    assembler.pruned_aggregators.clear()
    
    # Calculate baseline accuracy
    with torch.no_grad():
        baseline_output = assembler(X)
        baseline_accuracy = ((baseline_output > threshold).float() == Y).float().mean().item()
    
    accuracies = [baseline_accuracy]
    baseline_features = []
    
    # Save original weights
    original_weights = [w.data.clone() for w in assembler.weights]
    
    # IMPORTANT: In a left-associative tree, there are num_features-1 aggregators
    # Feature at position i uses aggregator i-1 to combine with previous result
    # So we can prune features 0 through num_features-1 (all features)
    num_aggregators = len(assembler.weights)
    max_prunable_feature = num_aggregators  # This equals num_features - 1
    
    # Detect baseline structurally
    if baseline_enabled:
        print("🔍 Detecting baseline features (structural)...")
        
        # In a left-associative tree, the first two features (indices 0 and 1) 
        # ALWAYS form the baseline - they establish the initial conjunctive or disjunctive relationship
        # This is a structural property, not dependent on accuracy drops
        if num_features >= 2:
            baseline_features = [0, 1]
            
            # Determine the baseline type from the first aggregator's bias
            first_bias = assembler.biases[0].item()
            if assembler.normalize_andness:
                first_bias = torch.sigmoid(torch.tensor(first_bias)) * 3 - 1
                first_bias = first_bias.item()
            
            baseline_type = "Conjunctive" if first_bias > 0.5 else "Disjunctive"
            
            feature_0_idx = assembler.locked_perm[0].item()
            feature_1_idx = assembler.locked_perm[1].item()
            
            print(f"   📌 Baseline: Features 0 and 1 ({feature_names[feature_0_idx]} + {feature_names[feature_1_idx]})")
            print(f"   📌 Baseline type: {baseline_type} (bias a={first_bias:.4f})")
            print(f"   📌 These two features establish the initial logical relationship")
            print(f"📌 Baseline detected: {len(baseline_features)} features will NOT be pruned")
        else:
            print("📌 No baseline detected (fewer than 2 features)")
    
    # Cumulative pruning analysis (skip baseline features)
    print("\n🔬 Cumulative pruning analysis:")
    
    # Determine where to start pruning
    start_feature = max(baseline_features) + 1 if baseline_features else 0
    
    # Track actual number of features pruned (excluding baseline)
    # Note: Only iterate up to max_prunable_feature (num_aggregators) since there are num_features-1 aggregators
    for i in range(start_feature, max_prunable_feature + 1):
        # Restore original weights
        for j, w in enumerate(assembler.weights):
            w.data.copy_(original_weights[j])
        assembler.pruned_aggregators.clear()
        
        # Apply cumulative pruning: prune features from start_feature through i, EXCEPT baseline
        for k in range(start_feature, i + 1):
            if k not in baseline_features:  # Skip baseline features (shouldn't happen with start_feature logic)
                assembler.prune_features(k)
        
        with torch.no_grad():
            pruned_output = assembler(X)
            pruned_accuracy = ((pruned_output > threshold).float() == Y).float().mean().item()
            accuracies.append(pruned_accuracy)
            
            # Calculate actual number of pruned features (excluding baseline)
            num_pruned = i - start_feature + 1
            baseline_note = f" (baseline protected: {len(baseline_features)} features)" if baseline_enabled and len(baseline_features) > 0 else ""
            print(f"   Pruning feature {i} ({feature_names[assembler.locked_perm[i].item()]}): accuracy = {pruned_accuracy * 100:.2f}% (total pruned: {num_pruned}){baseline_note}")
    
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


def analyze_feature_importance_with_growing(
    model, 
    X, 
    Y, 
    feature_names, 
    threshold=0.5,
    device=None
):
    """Analyze feature importance through incremental growth from baseline.
    
    Start with baseline (node0, node1, aggregator0) and incrementally grow the tree
    towards the root by restoring one feature/aggregator at a time.
    
    Args:
        model: Trained baconNet model with frozen structure
        X: Input tensor
        Y: Target tensor
        feature_names: List of feature names
        threshold: Classification threshold for accuracy calculation
        device: torch device
        
    Returns:
        dict: {
            'accuracies': List of accuracies as tree grows from baseline,
            'feature_order': List of feature indices in growth order
        }
    """
    if device is None:
        device = X.device
    
    assembler = model.assembler
    num_features = len(feature_names)
    
    if num_features < 2:
        print("⚠️ Growing analysis requires at least 2 features")
        return {'accuracies': [], 'feature_order': []}
    
    # Save original weights and biases
    original_weights = [w.data.clone() for w in assembler.weights]
    original_biases = [b.data.clone() for b in assembler.biases]
    
    num_aggregators = len(assembler.weights)
    accuracies = []
    feature_order = list(range(num_features))
    
    print("\n🌱 Incremental growing analysis:")
    print(f"   Starting with baseline: Features 0 and 1 ({feature_names[assembler.locked_perm[0].item()]} + {feature_names[assembler.locked_perm[1].item()]})")
    
    # Step 1: Evaluate baseline (node0, node1, aggregator0 only)
    # Set all aggregators except agg0 to neutrality: bias=0.5, left=1, right=0
    with torch.no_grad():
        # Restore all to original first
        for j, w in enumerate(assembler.weights):
            w.data.copy_(original_weights[j])
        for j in range(len(assembler.biases)):
            if assembler.biases[j].shape == original_biases[j].shape:
                assembler.biases[j].data.copy_(original_biases[j])
            else:
                assembler.biases[j].data = original_biases[j].clone()
        
        # Set aggregators 1 onwards to neutrality (pass left input through)
        for j in range(1, num_aggregators):
            assembler.weights[j].data = torch.tensor([1.0, 0.0], dtype=torch.float32, device=device)
            # Set bias to 0.5 (logical neutrality)
            if assembler.normalize_andness:
                # If normalize_andness, bias is logit-transformed: logit(0.5) = 0
                if assembler.biases[j].numel() == 1:
                    assembler.biases[j].data.fill_(0.0)
                else:
                    assembler.biases[j].data = torch.tensor(0.0, dtype=torch.float32, device=device)
            else:
                if assembler.biases[j].numel() == 1:
                    assembler.biases[j].data.fill_(0.5)
                else:
                    assembler.biases[j].data = torch.tensor(0.5, dtype=torch.float32, device=device)
        
        baseline_output = assembler(X)
        baseline_accuracy = ((baseline_output > threshold).float() == Y).float().mean().item()
        accuracies.append(baseline_accuracy)
        print(f"   Baseline (features 0-1): accuracy = {baseline_accuracy * 100:.2f}%")
    
    # Step 2: Incrementally grow by adding features 2 onwards
    for i in range(2, num_features):
        with torch.no_grad():
            # Restore all to original
            for j, w in enumerate(assembler.weights):
                w.data.copy_(original_weights[j])
            for j in range(len(assembler.biases)):
                if assembler.biases[j].shape == original_biases[j].shape:
                    assembler.biases[j].data.copy_(original_biases[j])
                else:
                    assembler.biases[j].data = original_biases[j].clone()
            
            # Feature i uses aggregator i-1
            # We want to include features 0 through i
            # So restore aggregators 0 through i-1 to original values
            # Set aggregators i onwards to neutrality (pass left input through)
            for j in range(i, num_aggregators):
                assembler.weights[j].data = torch.tensor([1.0, 0.0], dtype=torch.float32, device=device)
                # Set bias to 0.5 (logical neutrality)
                if assembler.normalize_andness:
                    # If normalize_andness, bias is logit-transformed: logit(0.5) = 0
                    if assembler.biases[j].numel() == 1:
                        assembler.biases[j].data.fill_(0.0)
                    else:
                        assembler.biases[j].data = torch.tensor(0.0, dtype=torch.float32, device=device)
                else:
                    if assembler.biases[j].numel() == 1:
                        assembler.biases[j].data.fill_(0.5)
                    else:
                        assembler.biases[j].data = torch.tensor(0.5, dtype=torch.float32, device=device)
            
            grown_output = assembler(X)
            grown_accuracy = ((grown_output > threshold).float() == Y).float().mean().item()
            accuracies.append(grown_accuracy)
            
            feature_name = feature_names[assembler.locked_perm[i].item()]
            print(f"   Growing to feature {i} ({feature_name}): accuracy = {grown_accuracy * 100:.2f}% (total features: {i+1})")
    
    # Restore original weights and biases
    with torch.no_grad():
        for j, w in enumerate(assembler.weights):
            w.data.copy_(original_weights[j])
        for j in range(len(assembler.biases)):
            if assembler.biases[j].shape == original_biases[j].shape:
                assembler.biases[j].data.copy_(original_biases[j])
            else:
                assembler.biases[j].data = original_biases[j].clone()
    
    return {
        'accuracies': accuracies,
        'feature_order': feature_order
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
        
        original_leaf_names = leaf_names.copy()  # Keep permuted but untransformed names
        leaf_names = []
        transformations_applied = []
        transformation_params = []
        
        for i, name in enumerate(original_leaf_names):
            transform_idx = selected_transforms[i].item()
            transform_name = transformation_names[transform_idx]
            transform_obj = model.transformation_layer.transformations[transform_idx]
            transformations_applied.append(transform_name)
            
            # Get learned parameters for this transformation
            params = {}
            if hasattr(transform_obj, 'get_param_summary'):
                # Extract params for this transformation from ParameterDict
                # Parameters are stored as "t{idx}_{param_name}"
                transform_params_dict = {}
                for key, value in model.transformation_layer.transform_params.items():
                    if key.startswith(f"t{transform_idx}_"):
                        param_name = key[len(f"t{transform_idx}_"):]
                        transform_params_dict[param_name] = value
                
                if transform_params_dict:
                    params = transform_obj.get_param_summary(transform_params_dict, i)
            transformation_params.append(params)
            
            # Format display name based on transformation type
            if transform_name == 'negation':
                leaf_names.append(f"NOT {name}")
            elif transform_name == 'peak':
                peak_loc = params.get('peak_location', '?')
                leaf_names.append(f"PEAK({name}, t={peak_loc})")
            elif transform_name == 'valley':
                valley_loc = params.get('valley_location', '?')
                leaf_names.append(f"VALLEY({name}, t={valley_loc})")
            elif transform_name == 'step_up':
                threshold = params.get('threshold', '?')
                leaf_names.append(f"STEP_UP({name}, t={threshold})")
            elif transform_name == 'step_down':
                threshold = params.get('threshold', '?')
                leaf_names.append(f"STEP_DOWN({name}, t={threshold})")
            else:  # identity or unknown
                leaf_names.append(name)
    else:
        original_leaf_names = leaf_names.copy()  # No transformations, but keep copy
        transformations_applied = ["identity"] * len(leaf_names)
        transformation_params = [{}] * len(leaf_names)
    
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
        "original_feature_order": feature_names if feature_names else [f"feature{i}" for i in range(model.num_leaves)],
        "locked_perm": model.locked_perm.tolist() if model.locked_perm is not None else None,
        "features": []
    }
    
    # Add feature information with transformations
    # original_leaf_names are the permuted feature names (before transformation display)
    # leaf_names are the display names (after transformation, e.g., "NOT age")
    # transformations_applied are the transformation types
    # transformation_params are the learned parameters
    for i in range(len(leaf_names)):
        feature_info = {
            "index": i,
            "original_name": original_leaf_names[i],  # Permuted but untransformed name
            "display_name": leaf_names[i],  # With transformation applied
            "transformation": transformations_applied[i]
        }
        # Add learned parameters if available
        if transformation_params[i]:
            feature_info["transformation_params"] = transformation_params[i]
        tree_structure["features"].append(feature_info)
    
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


# ============================================================================
# BACON Network Distillation - Generate Standalone Inference Code
# ============================================================================

def _generate_aggregator_library(aggregator_type):
    """Generate standalone Python code for aggregator functions.
    
    Args:
        aggregator_type (str): Type of aggregator (e.g., 'lsp.half_weight')
        
    Returns:
        str: Python code for the aggregator implementation
    """
    if aggregator_type == 'lsp.half_weight':
        return '''
def lsp_half_weight_r(a):
    """Compute r parameter for LSP half-weight aggregator."""
    delta = 0.5 - a
    numerator = (0.25 +
                 1.65811 * delta +
                 2.15388 * delta ** 2 + 
                 8.2844 * delta ** 3 +
                 6.16764 * delta ** 4)
    denominator = a * (1 - a)
    epsilon = 1e-6
    if abs(denominator) < epsilon:
        denominator = epsilon if denominator >= 0 else -epsilon
    return numerator / denominator


def lsp_half_weight_aggregate(x, y, a, w0, w1):
    """LSP half-weight aggregator for two inputs.
    
    Args:
        x: First input (0 to 1)
        y: Second input (0 to 1)
        a: Andness parameter (-1 to 2)
        w0: Weight for first input
        w1: Weight for second input
        
    Returns:
        Aggregated value (0 to 1)
    """
    epsilon = 1e-6
    
    # Clamp inputs to valid range
    x = max(epsilon, min(1 - epsilon, x))
    y = max(epsilon, min(1 - epsilon, y))
    a = max(-1.0 + epsilon, min(2.0 - epsilon, a))
    
    # Rule 0: a == 2 (full conjunction)
    if abs(a - 2) < epsilon:
        return 1.0 if (abs(x - 1) < epsilon and abs(y - 1) < epsilon) else 0.0
    
    # Rule 1: 0.75 <= a < 2 (strong conjunction)
    elif a >= 0.75:
        import math
        return (x ** (2*w0) * y ** (2*w1)) ** (math.sqrt(3 / (2 - a)) - 1)
    
    # Rule 2: 0.5 < a < 0.75 (weak conjunction)
    elif a > 0.5:
        import math
        R = lsp_half_weight_r(0.75)
        return (3 - 4*a) * (w0*x + w1*y) + (4*a - 2) * (x ** (2*w0) * y ** (2*w1)) ** (math.sqrt(3 / (2 - a)) - 1)
    
    # Rule 3: a == 0.5 (arithmetic mean)
    elif abs(a - 0.5) < epsilon:
        return w0*x + w1*y
    
    # Rule 4: -1 <= a < 0.5 (disjunction, use De Morgan)
    elif a >= -1:
        return 1 - lsp_half_weight_aggregate(1-x, 1-y, max(-1.0 + epsilon, min(2.0 - epsilon, 1-a)), w0, w1)
    
    else:
        raise ValueError(f"Invalid andness value: {a}. Must be in [-1, 2].")
'''
    elif aggregator_type == 'lsp.full_weight':
        return '''
def lsp_full_weight_aggregate(x, y, a, w0, w1):
    """LSP full-weight aggregator (simplified placeholder).
    
    Args:
        x: First input (0 to 1)
        y: Second input (0 to 1)
        a: Andness parameter (-1 to 2)
        w0: Weight for first input
        w1: Weight for second input
        
    Returns:
        Aggregated value (0 to 1)
    """
    # Simplified implementation - extend as needed
    epsilon = 1e-6
    x = max(epsilon, min(1 - epsilon, x))
    y = max(epsilon, min(1 - epsilon, y))
    a = max(-1.0 + epsilon, min(2.0 - epsilon, a))
    
    # Weighted geometric mean for conjunction, weighted arithmetic for disjunction
    if a >= 0.5:
        import math
        return (x ** w0 * y ** w1) ** (1 + (a - 0.5) * 2)
    else:
        return 1 - lsp_full_weight_aggregate(1-x, 1-y, 1-a, w0, w1)
'''
    elif aggregator_type.startswith('math.'):
        return '''
def math_aggregate(x, y, a, w0, w1):
    """Mathematical aggregator (arithmetic/geometric mean).
    
    Args:
        x: First input (0 to 1)
        y: Second input (0 to 1)
        a: Andness parameter (ignored for math aggregators)
        w0: Weight for first input
        w1: Weight for second input
        
    Returns:
        Aggregated value (0 to 1)
    """
    # Weighted arithmetic mean
    return w0 * x + w1 * y
'''
    else:
        # Generic fallback
        return '''
def generic_aggregate(x, y, a, w0, w1):
    """Generic aggregator fallback (weighted average).
    
    Args:
        x: First input (0 to 1)
        y: Second input (0 to 1)
        a: Andness parameter
        w0: Weight for first input
        w1: Weight for second input
        
    Returns:
        Aggregated value (0 to 1)
    """
    return w0 * x + w1 * y
'''


def _generate_transformation_library():
    """Generate standalone Python code for transformation functions."""
    return '''
def apply_identity(x):
    """Identity transformation."""
    return x


def apply_negation(x):
    """Negation transformation."""
    return 1.0 - x


def apply_peak(x, center=0.5, sharpness=2.0):
    """Peak transformation - bell curve centered at 'center'."""
    return 1.0 - abs(x - center) ** sharpness


def apply_valley(x, center=0.5, sharpness=2.0):
    """Valley transformation - inverted bell curve."""
    return abs(x - center) ** sharpness


def apply_step_up(x, threshold=0.5, sharpness=10.0):
    """Step up transformation - sigmoid-like step."""
    import math
    return 1.0 / (1.0 + math.exp(-sharpness * (x - threshold)))


def apply_step_down(x, threshold=0.5, sharpness=10.0):
    """Step down transformation - inverted sigmoid."""
    import math
    return 1.0 / (1.0 + math.exp(sharpness * (x - threshold)))
'''


def _generate_vectorized_transformation_library():
    """Generate vectorized transformation functions for batch mode."""
    return '''
def apply_identity_vec(x):
    """Vectorized identity transformation."""
    return x


def apply_negation_vec(x):
    """Vectorized negation transformation."""
    return 1.0 - x


def apply_peak_vec(x, center=0.5, sharpness=2.0):
    """Vectorized peak transformation - bell curve centered at 'center'."""
    return 1.0 - np.abs(x - center) ** sharpness


def apply_valley_vec(x, center=0.5, sharpness=2.0):
    """Vectorized valley transformation - inverted bell curve."""
    return np.abs(x - center) ** sharpness


def apply_step_up_vec(x, threshold=0.5, sharpness=10.0):
    """Vectorized step up transformation - sigmoid-like step."""
    return 1.0 / (1.0 + np.exp(-sharpness * (x - threshold)))


def apply_step_down_vec(x, threshold=0.5, sharpness=10.0):
    """Vectorized step down transformation - inverted sigmoid."""
    return 1.0 / (1.0 + np.exp(sharpness * (x - threshold)))
'''


def distill_bacon_to_code(json_file, output_file, aggregator_type='lsp.half_weight', mode='instance'):
    """Distill a BACON model from JSON to standalone executable Python code.
    
    This generates a self-contained Python file that can perform inference
    without requiring the BACON framework. The generated code includes:
    - Necessary aggregator implementations
    - Transformation functions
    - Model inference function
    
    Args:
        json_file (str): Path to the JSON file containing the model structure
        output_file (str): Path where the generated Python code will be saved
        aggregator_type (str): Type of aggregator used in the model
        mode (str): Generation mode - 'instance' (zero dependencies) or 'batch' (NumPy required)
        
    Returns:
        str: Path to the generated Python file
    """
    import json
    
    # Load model structure
    with open(json_file, 'r', encoding='utf-8') as f:
        model_data = json.load(f)
    
    # Determine aggregator function name
    if aggregator_type == 'lsp.half_weight':
        agg_func_name = 'lsp_half_weight_aggregate'
    elif aggregator_type == 'lsp.full_weight':
        agg_func_name = 'lsp_full_weight_aggregate'
    elif aggregator_type.startswith('math.'):
        agg_func_name = 'math_aggregate'
    else:
        agg_func_name = 'generic_aggregate'
    
    # Start building the code
    code_parts = []
    
    # Header
    if mode == 'batch':
        code_parts.append('''"""
Distilled BACON Network - Standalone Inference Code (Batch Mode)
Generated automatically from trained BACON model.

This file can perform batch inference with NumPy for vectorized operations.
Dependency: NumPy
"""

import math
import numpy as np
''')
    else:
        code_parts.append('''"""
Distilled BACON Network - Standalone Inference Code (Instance Mode)
Generated automatically from trained BACON model.

This file is self-contained and can perform inference without any dependencies.
Zero dependencies: Uses only Python standard library (math module).
"""

import math
''')
    
    # Add aggregator library
    code_parts.append('\n# ============================================================================')
    code_parts.append('# Aggregator Functions')
    code_parts.append('# ============================================================================\n')
    code_parts.append(_generate_aggregator_library(aggregator_type))
    
    # Add transformation library
    code_parts.append('\n# ============================================================================')
    code_parts.append('# Transformation Functions')
    code_parts.append('# ============================================================================\n')
    code_parts.append(_generate_transformation_library())
    
    # For batch mode, also add vectorized versions
    if mode == 'batch':
        code_parts.append('\n# ============================================================================')
        code_parts.append('# Vectorized Transformation Functions (for Batch Mode)')
        code_parts.append('# ============================================================================\n')
        code_parts.append(_generate_vectorized_transformation_library())
    
    # Generate inference function based on tree layout
    code_parts.append('\n# ============================================================================')
    code_parts.append('# Model Inference')
    code_parts.append('# ============================================================================\n')
    
    layout = model_data.get('layout', 'left')
    features = model_data.get('features', [])
    original_feature_order = model_data.get('original_feature_order', [feat['original_name'] for feat in features])
    locked_perm = model_data.get('locked_perm', None)
    
    # Generate the predict function based on mode
    if mode == 'batch':
        code_parts.append(f'''
def predict(input_array):
    """Perform batch inference on input data.
    
    Args:
        input_array: NumPy array of shape (n_samples, {len(original_feature_order)}) 
                     or single sample of shape ({len(original_feature_order)},)
                     Features in ORIGINAL dataset order: {original_feature_order}
        
    Returns:
        NumPy array of predictions (0 to 1), shape (n_samples,) or scalar for single sample
    """
    input_array = np.atleast_2d(input_array)
    if input_array.shape[1] != {len(original_feature_order)}:
        raise ValueError(f"Expected {len(original_feature_order)} features, got {{input_array.shape[1]}}")
    
    # Apply permutation and transformations (vectorized)
    features = []
''')
        
        # For each position in the permuted order (batch mode with vectorized functions)
        for feat in features:
            perm_idx = feat['index']
            orig_name = feat['original_name']
            input_idx = original_feature_order.index(orig_name)
            transform = feat['transformation']
            
            if transform == 'identity':
                code_parts.append(f"    features.append(apply_identity_vec(input_array[:, {input_idx}]))  # {feat['display_name']}")
            elif transform == 'negation':
                code_parts.append(f"    features.append(apply_negation_vec(input_array[:, {input_idx}]))  # {feat['display_name']}")
            elif transform == 'peak':
                code_parts.append(f"    features.append(apply_peak_vec(input_array[:, {input_idx}]))  # {feat['display_name']}")
            elif transform == 'valley':
                code_parts.append(f"    features.append(apply_valley_vec(input_array[:, {input_idx}]))  # {feat['display_name']}")
            elif transform == 'stepup':
                code_parts.append(f"    features.append(apply_step_up_vec(input_array[:, {input_idx}]))  # {feat['display_name']}")
            elif transform == 'stepdown':
                code_parts.append(f"    features.append(apply_step_down_vec(input_array[:, {input_idx}]))  # {feat['display_name']}")
            else:
                code_parts.append(f"    features.append(apply_identity_vec(input_array[:, {input_idx}]))  # {feat['display_name']} (unknown: {transform})")
        
        code_parts.append('\n    # Convert to array for easier indexing\n')
        code_parts.append('    features = np.array(features)  # Shape: (n_features, n_samples)\n')
        code_parts.append('\n    # Aggregate through the tree (vectorized)\n')
        
    else:  # instance mode
        code_parts.append(f'''
def predict(input_array):
    """Perform inference on input data.
    
    Args:
        input_array: List or array of {len(original_feature_order)} input features in ORIGINAL dataset order
                     Feature order: {original_feature_order}
        
    Returns:
        float: Prediction value (0 to 1)
    """
    if len(input_array) != {len(original_feature_order)}:
        raise ValueError(f"Expected {len(original_feature_order)} features, got {{len(input_array)}}")
    
    # Apply permutation and transformations
    features = []
''')
        
        # For each position in the permuted order (instance mode)
        for feat in features:
            perm_idx = feat['index']
            orig_name = feat['original_name']
            input_idx = original_feature_order.index(orig_name)
            transform = feat['transformation']
            
            if transform == 'identity':
                code_parts.append(f"    features.append(apply_identity(input_array[{input_idx}]))  # {feat['display_name']}")
            elif transform == 'negation':
                code_parts.append(f"    features.append(apply_negation(input_array[{input_idx}]))  # {feat['display_name']}")
            elif transform == 'peak':
                code_parts.append(f"    features.append(apply_peak(input_array[{input_idx}]))  # {feat['display_name']}")
            elif transform == 'valley':
                code_parts.append(f"    features.append(apply_valley(input_array[{input_idx}]))  # {feat['display_name']}")
            elif transform == 'stepup':
                code_parts.append(f"    features.append(apply_step_up(input_array[{input_idx}]))  # {feat['display_name']}")
            elif transform == 'stepdown':
                code_parts.append(f"    features.append(apply_step_down(input_array[{input_idx}]))  # {feat['display_name']}")
            else:
                code_parts.append(f"    features.append(apply_identity(input_array[{input_idx}]))  # {feat['display_name']} (unknown: {transform})")
        
        code_parts.append('\n    # Aggregate through the tree\n')
    
    # Generate aggregation code based on layout
    if layout == 'left':
        nodes = model_data.get('nodes', [])
        if mode == 'batch':
            # Batch mode: vectorized aggregation
            for i, node in enumerate(nodes):
                layer = node['layer']
                andness = node['andness']
                w0 = node['weights']['left']
                w1 = node['weights']['right']
                
                if layer == 0:
                    left_idx = node['left_input']['index']
                    right_idx = node['right_input']['index']
                    code_parts.append(f"    agg_{layer} = np.array([{agg_func_name}(features[{left_idx}, i], features[{right_idx}, i], {andness}, {w0}, {w1}) for i in range(features.shape[1])])  # Layer {layer}")
                else:
                    right_idx = node['right_input']['index']
                    code_parts.append(f"    agg_{layer} = np.array([{agg_func_name}(agg_{layer-1}[i], features[{right_idx}, i], {andness}, {w0}, {w1}) for i in range(features.shape[1])])  # Layer {layer}")
            
            code_parts.append(f'\n    return agg_{len(nodes)-1} if input_array.ndim > 1 else agg_{len(nodes)-1}[0]\n')
        else:
            # Instance mode: single sample aggregation
            for i, node in enumerate(nodes):
                layer = node['layer']
                andness = node['andness']
                w0 = node['weights']['left']
                w1 = node['weights']['right']
                
                if layer == 0:
                    left_idx = node['left_input']['index']
                    right_idx = node['right_input']['index']
                    code_parts.append(f"    agg_{layer} = {agg_func_name}(features[{left_idx}], features[{right_idx}], {andness}, {w0}, {w1})  # Layer {layer}")
                else:
                    right_idx = node['right_input']['index']
                    code_parts.append(f"    agg_{layer} = {agg_func_name}(agg_{layer-1}, features[{right_idx}], {andness}, {w0}, {w1})  # Layer {layer}")
            
            code_parts.append(f'\n    return agg_{len(nodes)-1}\n')
    
    elif layout == 'balanced':
        # For balanced tree, we need to recursively build the aggregation
        code_parts.append('    # TODO: Balanced tree aggregation - implement recursive structure\n')
        code_parts.append('    raise NotImplementedError("Balanced tree layout not yet implemented in distillation")\n')
    
    elif layout == 'paired':
        # For paired tree, handle in two phases
        nodes = model_data.get('nodes', [])
        pairing_nodes = [n for n in nodes if n.get('phase') == 'pairing']
        folding_nodes = [n for n in nodes if n.get('phase') == 'folding']
        
        code_parts.append('    # Phase 1: Pair adjacent features\n')
        pair_results = []
        for i, node in enumerate(pairing_nodes):
            left_idx = node['left_input']['index']
            right_idx = node['right_input']['index']
            andness = node['andness']
            w0 = node['weights']['left']
            w1 = node['weights']['right']
            code_parts.append(f"    pair_{i} = {agg_func_name}(features[{left_idx}], features[{right_idx}], {andness}, {w0}, {w1})")
            pair_results.append(f"pair_{i}")
        
        # Handle unpaired feature if exists
        if len(features) % 2 == 1:
            code_parts.append(f"    pair_{len(pairing_nodes)} = features[{len(features)-1}]  # Unpaired feature")
            pair_results.append(f"pair_{len(pairing_nodes)}")
        
        code_parts.append('\n    # Phase 2: Fold pairs left-associatively\n')
        for i, node in enumerate(folding_nodes):
            andness = node['andness']
            w0 = node['weights']['left']
            w1 = node['weights']['right']
            if i == 0:
                code_parts.append(f"    fold_{i} = {agg_func_name}({pair_results[0]}, {pair_results[1]}, {andness}, {w0}, {w1})")
            else:
                code_parts.append(f"    fold_{i} = {agg_func_name}(fold_{i-1}, {pair_results[i+1]}, {andness}, {w0}, {w1})")
        
        if len(folding_nodes) > 0:
            code_parts.append(f'\n    return fold_{len(folding_nodes)-1}\n')
        else:
            code_parts.append(f'\n    return {pair_results[0]}\n')
    
    # Add example usage
    if mode == 'batch':
        code_parts.append('''

if __name__ == "__main__":
    # Example usage for batch mode
    import sys
    
    if len(sys.argv) > 1:
        # Read input from command line (single sample)
        try:
            input_values = np.array([[float(x) for x in sys.argv[1:]]])
            result = predict(input_values)
            print(f"Prediction: {result:.6f}")
        except Exception as e:
            print(f"Error: {e}")
            print(f"Usage: python {sys.argv[0]} <value1> <value2> ... <valueN>")
    else:
        # Demo with random batch input
        n_samples = 5
        demo_input = np.random.random((n_samples, ''' + str(len(features)) + '''))
        print(f"Demo batch input (shape {demo_input.shape}):")
        print(demo_input)
        results = predict(demo_input)
        print(f"\\nBatch predictions:")
        print(results)
''')
    else:
        code_parts.append('''

if __name__ == "__main__":
    # Example usage for instance mode
    import sys
    
    if len(sys.argv) > 1:
        # Read input from command line
        try:
            input_values = [float(x) for x in sys.argv[1:]]
            result = predict(input_values)
            print(f"Prediction: {result:.6f}")
        except Exception as e:
            print(f"Error: {e}")
            print(f"Usage: python {sys.argv[0]} <value1> <value2> ... <valueN>")
    else:
        # Demo with random input
        import random
        demo_input = [random.random() for _ in range(''' + str(len(features)) + ''')]
        print(f"Demo input: {demo_input}")
        result = predict(demo_input)
        print(f"Prediction: {result:.6f}")
''')
    
    # Write to file
    final_code = '\n'.join(code_parts)
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(final_code)
    
    print(f"✅ Distilled model saved to: {output_file}")
    print(f"   - Aggregator: {aggregator_type}")
    print(f"   - Mode: {mode} ({'zero dependencies' if mode == 'instance' else 'requires NumPy'})")
    print(f"   - Layout: {layout}")
    print(f"   - Features: {len(features)}")
    print(f"   - File size: {len(final_code)} characters")
    
    return output_file
