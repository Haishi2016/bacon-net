import networkx as nx
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
def left_associative_layout(G, root):
    pos = {}
    def dfs(node, depth=0, y=0):
        children = list(G.successors(node))
        if not children:
            pos[node] = (y, -depth)  # Leaf node: x by y order
            return y + 1
        else:
            y = dfs(children[0], depth + 1, y)
            y = dfs(children[1], depth + 1, y)
            pos[node] = ((pos[children[0]][0] + pos[children[1]][0]) / 2, -depth)
            return y
    dfs(root)
    return pos

def visualize_tree_structure(model, labels=None):
    if labels is not None and len(labels) < model.num_leaves:
        raise ValueError("Label count does not match number of leaves")

    if labels:
        if model.locked_perm is not None:
            leaf_names = [labels[i] for i in model.locked_perm.tolist()]
        else:
            leaf_names = labels
    else:
        leaf_names = [f"Leaf {i+1}" for i in range(model.num_leaves)]

    node_dict = {}
    node_labels = {}
    weight_map = {}
    leaf_nodes = set()

    # Build the tree structure and assign labels
    for i in range(model.num_layers):
        a = model.biases[i].item()
        w = model.weights[i].detach().cpu().numpy()

        left = f"Node{i}" if i > 0 else leaf_names[0]
        right = leaf_names[i + 1]
        parent = f"Node{i+1}"

        node_dict[parent] = (left, right)
        node_labels[parent] = f"{a:.2f}"

        weight_map[(parent, left)] = w[0]
        weight_map[(parent, right)] = w[1]

        # Keep track of all possible leaf nodes
        if left in leaf_names:
            leaf_nodes.add(left)    
        if right in leaf_names:
            leaf_nodes.add(right)

    # Add labels for leaf nodes
    for leaf in leaf_nodes:
        if leaf not in node_labels:
            node_labels[leaf] = leaf

    # Build the graph
    G = nx.DiGraph()
    def add_edges(node):
        if node in node_dict:
            l, r = node_dict[node]
            G.add_edge(node, l, weight=weight_map.get((node, l), 1.0))
            G.add_edge(node, r, weight=weight_map.get((node, r), 1.0))
            add_edges(l)
            add_edges(r)
        else:
            G.add_node(node)

    root = f"Node{model.num_layers}"
    add_edges(root)

    # Use the left-associative layout
    pos = left_associative_layout(G, root)
    edge_labels = {e: f"{w:.2f}" for e, w in nx.get_edge_attributes(G, "weight").items()}

    # Draw
    plt.figure(figsize=(14, 8))
    nx.draw(G, pos, with_labels=True, labels=node_labels, node_color='lightblue', node_size=2000, font_size=9)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
    plt.title("BACON Tree Structure (Left-Associative)")
    plt.axis("off")
    plt.margins(0.1)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show()

def print_tree_structure(model, labels=None, classic_boolean=False):
    # print(f" FC_OUT Weight: {model.fc_out.weight.item():.2f}")
    # print(f" FC_OUT Bias: {model.fc_out.bias.item():.2f}")
    """Print a left-associative logic tree showing weights and biases."""
    if labels is not None and len(labels) < model.num_leaves:
        raise ValueError(f"Label count {len(labels)} doesn't match number of leaves {model.num_leaves}")

    if labels:
        if model.locked_perm is not None:
            leaf_names = [labels[i] for i in model.locked_perm.tolist()]
        else:
            leaf_names = labels
    else:
        leaf_names = [f"feature{i+1}" for i in range(model.num_leaves)]

    max_label_length = max(len(name) for name in leaf_names)
    label_width = max_label_length + 2

    def fmt_label(name):
        return f"[{name}]".rjust(label_width + 2)

    print("\n🧠 Logical Aggregation Tree (Left-Associative):\n")

    previous_w = None
    weights = [w.detach().cpu().numpy() for w in model.weights]
    a = [(torch.sigmoid(b) * 3 - 1).item() for b in model.biases]
    indent = 2
    for i in range(model.num_layers):
        if i == 0:
            print(fmt_label(leaf_names[0]) + f"─{weights[0][0]:.2f}".rjust(5) + "────┐")
            new_leaf = leaf_names[1]
        else:
            new_leaf = leaf_names[i + 1]
        if classic_boolean:
            if a[i] >= 0.5:
                operator = "[ AND ]"
            else:
                operator = "[ O R ]"
        else:
            operator = f"[a={a[i]:.8f}]"
        if i < model.num_layers - 1:
            print(fmt_label(new_leaf) + f"─{weights[i][1]:.2f}".rjust(5) + "─" * indent +  operator + f"─{weights[i+1][0]:.2f}".rjust(5) + "────┐")
        else:
            print(fmt_label(new_leaf) + f"─{weights[i][1]:.2f}".rjust(5) + "─" * indent +  f"{operator}──OUTPUT")
        indent += 15

def plot_sorted_predictions_with_labels(model, X_test, Y_test, threshold=0.5):
    model.eval()
    with torch.no_grad():
        outputs = model.inference_raw(X_test).cpu().numpy().flatten()
    true_labels = Y_test.cpu().numpy().flatten()

    # Sort outputs and true labels together
    sorted_indices = np.argsort(outputs)
    sorted_outputs = outputs[sorted_indices]
    sorted_labels = true_labels[sorted_indices]

    plt.figure(figsize=(12, 6))
    for i, (score, label) in enumerate(zip(sorted_outputs, sorted_labels)):
        color = 'red' if label == 1 else 'blue'
        plt.plot(i, score, marker='o', color=color)

    plt.axhline(y=threshold, color='green', linestyle='dotted', label=f'Threshold = {threshold}')
    plt.title("Sorted Model Prediction Scores (Colored by True Label)")
    plt.xlabel("Sample Index (sorted)")
    plt.ylabel("Predicted Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_sorted_predictions_with_errors(model, X_test, Y_test, threshold=0.5, confidence_margin=0.2):
    """
    Plots sorted prediction scores with classification outcomes and highlights high-confidence correct predictions.

    Args:
        model: Trained model.
        X_test (torch.Tensor): Test features.
        Y_test (torch.Tensor): Ground truth labels (0 or 1).
        threshold (float): Decision threshold.
        confidence_margin (float): Margin for high confidence (default 20%).
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.lines import Line2D

    model.eval()
    with torch.no_grad():
        probs = model.inference_raw(X_test).cpu().numpy().flatten()
    true_labels = Y_test.cpu().numpy().flatten()

    # Sort outputs for visualization
    sorted_idx = np.argsort(probs)
    sorted_probs = probs[sorted_idx]
    sorted_labels = true_labels[sorted_idx]
    preds = (sorted_probs >= threshold).astype(int)

    plt.figure(figsize=(14, 7))

    for i, (score, label, pred) in enumerate(zip(sorted_probs, sorted_labels, preds)):
        # Determine high-confidence correct predictions
        if label == 1 and pred == 1 and score >= (1 - confidence_margin):
            plt.plot(i, score, marker='o', color='black', markersize=4)  # High-confidence TP
        elif label == 0 and pred == 0 and score <= confidence_margin:
            plt.plot(i, score, marker='^', color='gray', markersize=4)  # High-confidence TN
        elif label == 1 and pred == 1:
            plt.plot(i, score, marker='o', color='black', markersize=2)  # Normal TP
        elif label == 0 and pred == 0:
            plt.plot(i, score, marker='^', color='gray', markersize=2)  # Normal TN
        elif label == 0 and pred == 1:
            plt.plot(i, score, marker='x', color='orange', markersize=17, markeredgewidth=2)  # False Positive
        elif label == 1 and pred == 0:
            plt.plot(i, score, marker='x', color='red', markersize=17, markeredgewidth=2)  # False Negative

    # Threshold line
    plt.axhline(y=threshold, color='black', linestyle='dotted', linewidth=1, label=f'Threshold = {threshold:.2f}')
    
    low_conf_indices = np.where(sorted_probs <= confidence_margin)[0]
    high_conf_indices = np.where(sorted_probs >= (1 - confidence_margin))[0]

     # Draw two vertical bands (low and high confidence zones)
    if len(low_conf_indices) > 0:
        plt.axvspan(low_conf_indices[0] - 0.5, low_conf_indices[-1] + 0.5, color='blue', alpha=0.1, label='High-Confidence Negative Zone')

    if len(high_conf_indices) > 0:
        plt.axvspan(high_conf_indices[0] - 0.5, high_conf_indices[-1] + 0.5, color='green', alpha=0.1, label='High-Confidence Positive Zone')

    # Custom legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label='True Positive', markerfacecolor='black', markersize=8),
        Line2D([0], [0], marker='^', color='w', label='True Negative', markerfacecolor='gray', markersize=8),
        Line2D([0], [0], marker='x', color='orange', label='False Positive', markersize=8, markeredgewidth=2),
        Line2D([0], [0], marker='x', color='red', label='False Negative', markersize=8, markeredgewidth=2),
        Line2D([0], [0], color='black', linestyle='dotted', label='Threshold'),
        Line2D([0], [0], color='green', linewidth=8, alpha=0.2, label='High-Confidence Positive Zone'),
        Line2D([0], [0], color='blue', linewidth=8, alpha=0.2, label='High-Confidence Negative Zone')

    ]
    plt.legend(handles=legend_elements, loc='lower right')

    plt.title("Sorted Model Prediction Scores with High Confidence Bands")
    plt.xlabel("Sample Index (sorted by predicted score)")
    plt.ylabel("Predicted Score")
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()





def print_metrics(model, X, Y_true, threshold=0.5):
    model.eval()
    with torch.no_grad():
        probs = model.inference_raw(X).cpu().numpy().flatten()
    true_labels = Y_true.cpu().numpy().flatten()
    preds = (probs >= threshold).astype(int)

    acc = accuracy_score(true_labels, preds)
    prec = precision_score(true_labels, preds, zero_division=0)
    rec = recall_score(true_labels, preds, zero_division=0)
    f1 = f1_score(true_labels, preds, zero_division=0)

    print(f"Metrics at threshold {threshold:.2f}:")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-score : {f1:.4f}")

def plot_precision_vs_threshold(model, X_val, Y_val, steps=1000):
    model.eval()
    with torch.no_grad():
        probs = model.inference_raw(X_val).cpu().numpy().flatten()
    true_labels = Y_val.cpu().numpy().flatten()

    thresholds = np.linspace(1, 0, steps)
    precisions = []

    for threshold in thresholds:
        preds = (probs >= threshold).astype(int)
        prec = precision_score(true_labels, preds, zero_division=0)
        precisions.append(prec)

    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions, marker='.', label="Precision")
    plt.xlabel("Threshold")
    plt.ylabel("Precision")
    plt.title("Precision vs Threshold")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_feature_sensitivity(model, X_tensor, X_tensor_extended, feature_name, feature_names):
    """
    Plots model output vs. a single feature value, sorted by that feature.
    Also overlays the feature value itself for comparison.

    Args:
        model: Trained model with .inference_raw().
        X_tensor (torch.Tensor): Input features [N, D].
        feature_name (str): Name of the feature to analyze.
        feature_names (list of str): All feature names, matching X_tensor columns.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    assert feature_name in feature_names, f"Feature '{feature_name}' not found in feature_names list."
    feature_index = feature_names.index(feature_name)

    # Get feature values and predictions
    feature_values = X_tensor_extended[:, feature_index].cpu().numpy()
    model.eval()
    with torch.no_grad():
        outputs = model.inference_raw(X_tensor).cpu().numpy().flatten()

    # Sort by feature value
    sorted_idx = np.argsort(feature_values)
    sorted_feature = feature_values[sorted_idx]
    sorted_outputs = outputs[sorted_idx]

    # Plot both feature value and model output
    plt.figure(figsize=(8, 5))
    plt.plot(sorted_feature, sorted_outputs, label="Model Output", linestyle='-', marker='.')
    # plt.plot(sorted_feature, sorted_feature, label="Raw Feature", linestyle='--', alpha=0.6)
    # plt.fill_between(sorted_feature, sorted_feature, sorted_outputs, color='gray', alpha=0.2, label="Difference")

    plt.xlabel(feature_name)
    plt.ylabel("Score")
    plt.title(f"Model Response vs. {feature_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_multi_feature_sensitivity(model, X_tensor, feature_names_or, all_feature_names, is_or=True):
    """
    Plots model output vs. soft logical AND/OR of selected features,
    sorted by the first selected feature. Also shows difference.

    Args:
        model: Trained model with .inference_raw().
        X_tensor (torch.Tensor): Input features [N, D].
        feature_names_or (list of str): Feature names to compute logic over.
        all_feature_names (list of str): All feature column names.
        is_or (bool): Use soft-OR if True, soft-AND if False.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    assert all(fn in all_feature_names for fn in feature_names_or), \
        "Some feature names not found in all_feature_names."

    # Get feature indices and extract selected feature values
    indices = [all_feature_names.index(fn) for fn in feature_names_or]
    selected_features = X_tensor[:, indices].cpu().numpy()

    # Compute soft AND or soft OR
    if is_or:
        soft_logic = np.max(selected_features, axis=1)  # soft OR
    else:
        soft_logic = np.min(selected_features, axis=1)          # soft AND

    primary_feature = selected_features[:, 0]
    combined_logic = soft_logic

    # Lexicographic sort: first by combined logic, then by primary feature
    sort_idx = np.lexsort((combined_logic, primary_feature))


    # Get model predictions
    model.eval()
    with torch.no_grad():
        model_output = model.inference_raw(X_tensor).cpu().numpy().flatten()

    # Compute difference
    diff = model_output - soft_logic

    # Plot
    plt.figure(figsize=(10, 6))
    # plt.plot(primary_feature[sort_idx], soft_logic[sort_idx], label="Soft Logic", linestyle='--')
    # plt.plot(primary_feature[sort_idx], model_output[sort_idx], label="Model Output", linestyle='-')

    # Model Output
    plt.scatter(primary_feature[sort_idx], model_output[sort_idx], s=8, label="Model Output", alpha=0.7)

    # Soft Logic
    plt.scatter(primary_feature[sort_idx], soft_logic[sort_idx], s=8, label="Soft Logic", alpha=0.7)


    plt.fill_between(primary_feature[sort_idx],
                     soft_logic[sort_idx], model_output[sort_idx],
                     color='gray', alpha=0.2, label="Difference")

    logic_type = "OR" if is_or else "AND"
    feature_label = f"{logic_type}({', '.join(feature_names_or)})"

    plt.xlabel(f"{feature_names_or[0]} (sorted)")
    plt.ylabel("Score")
    plt.title(f"Model Output vs. Soft {logic_type}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_multi_feature_as_1(model, X_tensor, feature_names_or, all_feature_names):
    """
    Plots model output sorted by the logical OR across multiple input features.

    Args:
        model: Trained model.
        X_tensor (torch.Tensor): Input features [N, D].
        feature_names_or (list of str): Feature names to compute OR across.
        all_feature_names (list of str): All feature column names for mapping to tensor indices.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    assert all(fn in all_feature_names for fn in feature_names_or), \
        "Some OR feature names not found in all_feature_names."

    # Get indices
    indices = [all_feature_names.index(fn) for fn in feature_names_or]
    selected_features = X_tensor[:, indices].cpu().numpy()

    # Compute OR across columns (row-wise)
    mask_1 = np.all(selected_features == 1, axis=1).astype(int)  # shape [N]

    # Use OR mask plus row sum as sort key for tie-breaking
    feature_sum = selected_features.sum(axis=1)
    sort_key = mask_1 + feature_sum

    # Get predictions
    model.eval()
    with torch.no_grad():
        outputs = model.inference_raw(X_tensor).cpu().numpy().flatten()

    # Sort
    sorted_idx = np.argsort(sort_key)
    sorted_or_score = sort_key[sorted_idx]
    sorted_outputs = outputs[sorted_idx]

    # Plot
    plt.figure(figsize=(8, 5))
    feature_label = " ∨ ".join(feature_names_or)
    plt.plot(sorted_or_score, sorted_outputs, marker='o', linestyle='-')
    plt.xlabel(f"OR({feature_label})")
    plt.ylabel("Model Output (Score)")
    plt.title(f"Model Response vs. OR({feature_label})")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_feature_correlation(X_tensor, feature_name1, feature_name2, feature_names):
    import matplotlib.pyplot as plt
    import numpy as np

    # Check feature names
    assert feature_name1 in feature_names, f"Feature '{feature_name1}' not found."
    assert feature_name2 in feature_names, f"Feature '{feature_name2}' not found."

    idx1 = feature_names.index(feature_name1)
    idx2 = feature_names.index(feature_name2)

    feature1 = X_tensor[:, idx1].cpu().numpy()
    feature2 = X_tensor[:, idx2].cpu().numpy()

    # Plot
    plt.figure(figsize=(7, 5))
    plt.scatter(feature1, feature2, alpha=0.6, edgecolors='k')
    plt.xlabel(feature_name1)
    plt.ylabel(feature_name2)
    plt.title(f"Correlation between {feature_name1} and {feature_name2}")
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()
