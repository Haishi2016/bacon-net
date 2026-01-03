import networkx as nx
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import pandas as pd
import seaborn as sns
import torch.nn.functional as F

import torch
import torch.nn.functional as F

import networkx as nx
import matplotlib.pyplot as plt
import torch

def _left_associative_layout(G, root):
    """Same geometry as your previous helper: leaves laid out left→right, parents centered."""
    pos = {}
    def dfs(node, depth=0, y=0):
        children = list(G.successors(node))
        if not children:
            pos[node] = (y, -depth)
            return y + 1
        y = dfs(children[0], depth + 1, y)
        y = dfs(children[1], depth + 1, y)
        pos[node] = ((pos[children[0]][0] + pos[children[1]][0]) / 2, -depth)
        return y
    dfs(root)
    return pos

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

def visualize_tree_structure(model, labels=None, layout=None):
    if labels is not None and len(labels) < model.num_leaves:
        raise ValueError("Label count does not match number of leaves")

    if labels:
        if model.locked_perm is not None:
            leaf_names = [labels[i] for i in model.locked_perm.tolist()]
        else:
            leaf_names = labels
    else:
        leaf_names = [f"Leaf {i+1}" for i in range(model.num_leaves)]
    
    # Check for transformation layer and apply transformations to labels
    if hasattr(model, 'transformation_layer') and model.transformation_layer is not None:
        selected_transforms = model.transformation_layer.get_selected_transformations()
        original_leaf_names = leaf_names.copy()
        leaf_names = []
        
        # Get transformation names
        transformation_names = []
        for transform in model.transformation_layer.transformations:
            name = transform.__class__.__name__.replace('Transformation', '').lower()
            transformation_names.append(name)
        
        for i, name in enumerate(original_leaf_names):
            transform_idx = selected_transforms[i].item()
            transform_name = transformation_names[transform_idx]
            transform_obj = model.transformation_layer.transformations[transform_idx]
            
            # Get learned parameters
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
            
            # Format label based on transformation type
            if transform_name == 'negation':
                leaf_names.append(f"NOT {name}")
            elif transform_name == 'peak':
                peak_loc = params.get('peak_location', '?')
                leaf_names.append(f"PEAK({name},t={peak_loc})")
            elif transform_name == 'valley':
                valley_loc = params.get('valley_location', '?')
                leaf_names.append(f"VALLEY({name},t={valley_loc})")
            elif transform_name == 'step_up':
                threshold = params.get('threshold', '?')
                leaf_names.append(f"STEP_UP({name},t={threshold})")
            elif transform_name == 'step_down':
                threshold = params.get('threshold', '?')
                leaf_names.append(f"STEP_DOWN({name},t={threshold})")
            else:  # identity
                leaf_names.append(name)


    node_dict = {}
    node_labels = {}
    weight_map = {}
    leaf_nodes = set()

    effective_layout = layout or getattr(model, 'tree_layout', 'left')

    # Precompute a-values and normalized weights per internal node index
    a_vals = [
        (torch.sigmoid(b) * 3 - 1).item() if model.normalize_andness else b.item()
        for b in model.biases
    ]
    if model.weight_mode == 'fixed' or model.weight_normalization == 'minmax':
        w_list = model.weights
    else:
        w_list = [F.softmax(w, dim=0) for w in model.weights]

    def add_parent(parent, left, right, idx):
        node_dict[parent] = (left, right)
        node_labels[parent] = f"{a_vals[idx]:.2f}"
        w = w_list[idx]
        weight_map[(parent, left)] = w[0]
        weight_map[(parent, right)] = w[1]
        if left in leaf_names:
            leaf_nodes.add(left)
        if right in leaf_names:
            leaf_nodes.add(right)

    root = None
    if effective_layout == 'balanced':
        # Build a balanced tree structure with in-order node indexing
        idx_ref = {'i': 0}
        def build(start, end):
            if start == end:
                return leaf_names[start]
            mid = (start + end) // 2
            left = build(start, mid)
            right = build(mid + 1, end)
            parent = f"Node{idx_ref['i'] + 1}"
            add_parent(parent, left, right, idx_ref['i'])
            idx_ref['i'] += 1
            return parent
        root = build(0, model.num_leaves - 1)
    elif effective_layout == 'paired':
        # First stage: pair adjacent leaves; second stage: fold pair outputs
        idx = 0
        nodes = []
        j = 0
        while j < model.num_leaves:
            if j + 1 < model.num_leaves:
                parent = f"Node{idx + 1}"
                add_parent(parent, leaf_names[j], leaf_names[j + 1], idx)
                nodes.append(parent)
                idx += 1
                j += 2
            else:
                nodes.append(leaf_names[j])
                j += 1
        current = nodes[0]
        for k in range(1, len(nodes)):
            parent = f"Node{idx + 1}"
            add_parent(parent, current, nodes[k], idx)
            current = parent
            idx += 1
        root = current
    else:
        # Left-associative structure
        for i in range(model.num_layers):
            left = f"Node{i}" if i > 0 else leaf_names[0]
            right = leaf_names[i + 1]
            parent = f"Node{i+1}"
            add_parent(parent, left, right, i)
        root = f"Node{model.num_layers}"

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

    add_edges(root)

    # Use general DFS layout (works for all binary trees constructed above)
    pos = left_associative_layout(G, root)
    edge_labels = {e: f"{w:.2f}" for e, w in nx.get_edge_attributes(G, "weight").items()}

    # Draw
    plt.figure(figsize=(14, 8))
    nx.draw(G, pos, with_labels=True, labels=node_labels, node_color='lightblue', node_size=2000, font_size=9)
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_color='red')
    title = "Left-Associative"
    if effective_layout == 'balanced':
        title = 'Balanced'
    elif effective_layout == 'paired':
        title = 'Paired'
    plt.title(f"BACON Tree Structure ({title})")
    plt.axis("off")
    plt.margins(0.1)
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    plt.show()

def print_tree_structure(model, labels=None, classic_boolean=False, layout=None):
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
    
    # Check for transformation layer and apply transformations to labels
    if hasattr(model, 'transformation_layer') and model.transformation_layer is not None:
        selected_transforms = model.transformation_layer.get_selected_transformations()
        original_leaf_names = leaf_names.copy()
        leaf_names = []
        
        # Get transformation names
        transformation_names = []
        for transform in model.transformation_layer.transformations:
            name = transform.__class__.__name__.replace('Transformation', '').lower()
            transformation_names.append(name)
        
        for i, name in enumerate(original_leaf_names):
            transform_idx = selected_transforms[i].item()
            transform_name = transformation_names[transform_idx]
            transform_obj = model.transformation_layer.transformations[transform_idx]
            
            # Get learned parameters
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
            
            # Format label based on transformation type
            if transform_name == 'negation':
                leaf_names.append(f"NOT {name}")
            elif transform_name == 'peak':
                peak_loc = params.get('peak_location', '?')
                leaf_names.append(f"PEAK({name},t={peak_loc})")
            elif transform_name == 'valley':
                valley_loc = params.get('valley_location', '?')
                leaf_names.append(f"VALLEY({name},t={valley_loc})")
            elif transform_name == 'step_up':
                threshold = params.get('threshold', '?')
                leaf_names.append(f"STEP_UP({name},t={threshold})")
            elif transform_name == 'step_down':
                threshold = params.get('threshold', '?')
                leaf_names.append(f"STEP_DOWN({name},t={threshold})")
            else:  # identity
                leaf_names.append(name)


    max_label_length = max(len(name) for name in leaf_names)
    label_width = max_label_length + 2

    def fmt_label(name):
        return f"[{name}]".rjust(label_width + 2)

    effective_layout = layout or getattr(model, 'tree_layout', 'left')
    if effective_layout == 'balanced':
        print("\n🧠 Logical Aggregation Tree (Balanced):\n")
    elif effective_layout == 'paired':
        print("\n🧠 Logical Aggregation Tree (Paired):\n")
    else:
        print("\n🧠 Logical Aggregation Tree (Left-Associative):\n")

    # Map weights and biases to CPU for printing
    if model.weight_mode == 'fixed' or model.weight_normalization == 'minmax':
        weights = [w.detach().cpu() if hasattr(w, 'detach') else w for w in model.weights]
    else:
        weights = [F.softmax(w.detach().cpu(), dim=0) for w in model.weights]
    a_vals = [(torch.sigmoid(b) * 3 - 1).item() for b in model.biases]

    if effective_layout == 'balanced':
        # Build a balanced parenthesized expression using the same node indexing order
        def build_expr(start, end, idx):
            if start == end:
                return leaf_names[start], idx
            mid = (start + end) // 2
            left_expr, idx = build_expr(start, mid, idx)
            right_expr, idx = build_expr(mid + 1, end, idx)
            a = a_vals[idx]
            op = ("AND" if a >= 0.5 else "OR") if classic_boolean else f"a={a:.3f}"
            # weights per internal node
            w = weights[idx]
            if hasattr(w, 'tolist'):
                wl, wr = float(w[0]), float(w[1])
            else:
                wl, wr = float(w), float(1 - w)
            node_str = f"({left_expr} -{wl:.2f}- [{op}] -{wr:.2f}- {right_expr})"
            return node_str, idx + 1

        expr, _ = build_expr(0, model.num_leaves - 1, 0)
        print(expr)
        return
    elif effective_layout == 'paired':
        # Build a paired-then-fold expression using the same node indexing order as build_paired_tree
        idx = 0
        parts = []
        j = 0
        while j < model.num_leaves:
            if j + 1 < model.num_leaves:
                a = a_vals[idx]
                op = ("AND" if a >= 0.5 else "OR") if classic_boolean else f"a={a:.3f}"
                w = weights[idx]
                if hasattr(w, 'tolist'):
                    wl, wr = float(w[0]), float(w[1])
                else:
                    wl, wr = float(w), float(1 - w)
                node_str = f"([{leaf_names[j]}] -{wl:.2f}- [{op}] -{wr:.2f}- [{leaf_names[j+1]}])"
                parts.append(node_str)
                idx += 1
                j += 2
            else:
                parts.append(f"[{leaf_names[j]}]")
                j += 1

        current = parts[0]
        for k in range(1, len(parts)):
            a = a_vals[idx]
            op = ("AND" if a >= 0.5 else "OR") if classic_boolean else f"a={a:.3f}"
            w = weights[idx]
            if hasattr(w, 'tolist'):
                wl, wr = float(w[0]), float(w[1])
            else:
                wl, wr = float(w), float(1 - w)
            current = f"({current} -{wl:.2f}- [{op}] -{wr:.2f}- {parts[k]})"
            idx += 1
        print(current)
        return
    else:
        # Left-associative ASCII tree (existing behavior)
        indent = 2
        for i in range(model.num_layers):
            a = a_vals[i]
            if i == 0:
                print(fmt_label(leaf_names[0]) + f"─{weights[0][0]:.2f}".rjust(5) + "────┐")
                new_leaf = leaf_names[1]
            else:
                new_leaf = leaf_names[i + 1]
            if classic_boolean:
                operator = "[ AND ]" if a >= 0.5 else "[ O R ]"
            else:
                operator = f"[a={a:.8f}]"
            if i < model.num_layers - 1:
                print(
                    fmt_label(new_leaf)
                    + f"─{weights[i][1]:.2f}".rjust(5)
                    + "─" * indent
                    + operator
                    + f"─{weights[i+1][0]:.2f}".rjust(5)
                    + "────┐"
                )
            else:
                print(
                    fmt_label(new_leaf)
                    + f"─{weights[i][1]:.2f}".rjust(5)
                    + "─" * indent
                    + f"{operator}──OUTPUT"
                )
            indent += 15

def print_table_structure(model, labels=None):
    """
    Print the left-associative logic tree as a flat table showing features, weights, and biases at each layer.
    """
    if labels is not None and len(labels) < model.num_leaves:
        raise ValueError(f"Label count {len(labels)} doesn't match number of leaves {model.num_leaves}")

    if labels:
        if model.locked_perm is not None:
            leaf_names = [labels[i] for i in model.locked_perm.tolist()]
        else:
            leaf_names = labels
    else:
        leaf_names = [f"feature{i+1}" for i in range(model.num_leaves)]

    print("\n📋 Logical Aggregation Table (Left-Associative):\n")
    print(f"{'Layer':<6} {'Left Feature':<20} {'Right Feature':<20} {'w (left)':<10} {'a (bias)':<10} {'1-w (right)':<12}")
    print("-" * 80)

    if model.weight_mode == 'fixed' or model.weight_normalization == 'minmax':
        weights = model.weights
    else:
        weights = [F.softmax(w.detach().cpu(), dim=0) for w in model.weights]
    biases = [(torch.sigmoid(b) * 3 - 1).item() for b in model.biases]

    for i in range(model.num_layers):
        if i == 0:
            left_feature = leaf_names[0]
        else:
            left_feature = f"Node{i}"

        right_feature = leaf_names[i + 1]
        w = weights[i]
        a = biases[i]
        print(f"{i+1:<6} {left_feature:<20} {right_feature:<20} {w[0]:.4f}     {a:.4f}     {w[1]:.4f}")

    print("-" * 80)
    print("Note: 'w' applies to left input, '1-w' applies to right input.\n")


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

def plot_gcd_aggregator_3d(model, w, a, grid_points=100):
    from mpl_toolkits.mplot3d import Axes3D

    x = torch.linspace(0, 1, grid_points)
    y = torch.linspace(0, 1, grid_points)
    X, Y = torch.meshgrid(x, y, indexing='ij')

    x_flat = X.flatten()
    y_flat = Y.flatten()

    with torch.no_grad():
        output = model.aggregator.aggregate(x_flat, y_flat, a, w, 1-w)
        output_grid = output.reshape(grid_points, grid_points).cpu().numpy()

    X = X.cpu().numpy()
    Y = Y.cpu().numpy()

    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, output_grid, cmap='viridis', edgecolor='none', alpha=0.9)
    ax.set_xlabel('Left Input (x)')
    ax.set_ylabel('Right Input (y)')
    ax.set_zlabel('GCD Output')
    ax.set_title(f'3D GCD Aggregator Surface\nw={w:.2f}, 1-w={1-w:.2f}, a={a:.2f}')
    plt.show()

def plot_gcd_aggregator_3d_minimal(model, w, a, grid_points=5):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    import torch
    import numpy as np

    # Create coarse grid
    x = torch.linspace(0, 1, grid_points)
    y = torch.linspace(0, 1, grid_points)
    X, Y = torch.meshgrid(x, y, indexing='ij')

    x_flat = X.flatten()
    y_flat = Y.flatten()

    # Evaluate model
    with torch.no_grad():
        Z = model.aggregator.aggregate(x_flat, y_flat, a, w, 1 - w).reshape(grid_points, grid_points).cpu().numpy()

    X = X.cpu().numpy()
    Y = Y.cpu().numpy()

    # Create pure white facecolors with full alpha (RGBA)
    white_facecolors = np.full((grid_points, grid_points, 4), [1, 1, 1, 1])

    # Plot
    fig = plt.figure(figsize=(7, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(X, Y, Z,
                    facecolors=white_facecolors,
                    edgecolor='black',
                    linewidth=0.5,
                    antialiased=False,
                    shade=False)  # Important to disable shading

    ax.view_init(elev=20, azim=-137)  # Fix view angle

    # ax.set_xlabel('Left Input (x)')
    # ax.set_ylabel('Right Input (y)')
    # ax.set_zlabel('GCD Output')
    # ax.set_title(f'GCD Aggregator (Minimal Grid)\nw={w:.2f}, 1-w={1-w:.2f}, a={a:.2f}')
    plt.show()


def plot_feature_aggregator_response_aligned(model, X_tensor, feature_name, feature_names):
    """
    Plots feature value vs. left input, right input, and aggregator output using model's cached layer_outputs.

    Args:
        model: Your binaryTreeLogicNet instance.
        X_tensor: Input tensor [N, D].
        feature_name: Name of feature to analyze.
        feature_names: List of all feature names.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    assert feature_name in feature_names, f"Feature '{feature_name}' not found."
    feature_index = feature_names.index(feature_name)
    if feature_index == 0:
        raise ValueError("Feature 0 is always combined at Aggregator 0. No standalone aggregator.")

    model.eval()
    with torch.no_grad():
        # Trigger forward pass to ensure layer_outputs and node_outputs are populated
        _ = model(X_tensor)

        # From model.forward() logic:
        # At aggregator index (feature_index - 1):
        #   Left input = node_outputs[feature_index - 1]
        #   Right input = node_outputs[feature_index]
        #   Output = self.layer_outputs[feature_index - 1]
        leaf_values = model.input_to_leaf(X_tensor)
        node_outputs = list(leaf_values.T)

        left_input = node_outputs[feature_index - 1]
        right_input = node_outputs[feature_index]
        aggregator_output = model.layer_outputs[feature_index - 1]

    # Prepare for plotting
    feature_values = X_tensor[:, feature_index].cpu().numpy()
    sorted_idx = np.argsort(feature_values)
    sorted_feature = feature_values[sorted_idx]
    sorted_left = left_input.cpu().numpy()[sorted_idx]
    sorted_right = right_input.cpu().numpy()[sorted_idx]
    sorted_aggregator = aggregator_output.cpu().numpy()[sorted_idx]

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(sorted_feature, sorted_left, label="Left Input", linestyle='-', marker='.')
    plt.plot(sorted_feature, sorted_right, label="Right Input (Feature itself)", linestyle='-', marker='.')
    plt.plot(sorted_feature, sorted_aggregator, label="Aggregator Output", linestyle='-', marker='o')
    plt.xlabel(feature_name)
    plt.ylabel("Values")
    plt.title(f"{feature_name} vs. Aggregator {feature_index - 1} Inputs & Output")
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
    plt.plot(sorted_feature, sorted_feature, label="Raw Feature", linestyle='--', alpha=0.6)
    # plt.fill_between(sorted_feature, sorted_feature, sorted_outputs, color='gray', alpha=0.2, label="Difference")

    plt.xlabel(feature_name)
    plt.ylabel("Score")
    plt.title(f"Model Response vs. {feature_name}")
    # plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_feature_sensitivity_synthetic(model, X_tensor, feature_name, feature_names_extended, baseline='mean', steps=100):
    """
    Sweep synthetic feature values from 0 to 1 while fixing other features.
    For extended features outside the model, use X_tensor_extended for feature value, but X_tensor for inference.

    Args:
        model: Trained model with .inference_raw().
        X_tensor (torch.Tensor): Real model input features [N, D].
        X_tensor_extended (torch.Tensor): Extended features [N, D+X].
        feature_name (str): Feature to analyze (from extended feature list).
        feature_names_extended (list of str): All columns in X_tensor_extended.
        baseline (str): 'mean' or 'median'.
    """
    assert feature_name in feature_names_extended, f"Feature '{feature_name}' not found in extended feature list."
    feature_index = feature_names_extended.index(feature_name)

    # Use mean/median from real model input (safe baseline)
    if baseline == 'mean':
        baseline_input = X_tensor.mean(dim=0, keepdim=True).repeat(steps, 1)
    else:
        baseline_input = X_tensor.median(dim=0, keepdim=True).values.repeat(steps, 1)

    # Use the feature from extended tensor for plotting (synthetic sweep from 0 to 1)
    synthetic_feature_values = torch.linspace(0, 1, steps).unsqueeze(1).to(X_tensor.device)
    baseline_input[:, feature_index] = synthetic_feature_values.flatten()
    # Get model output (input is baseline_input, feature is only for plotting)
    model.eval()
    with torch.no_grad():
        outputs = model.inference_raw(baseline_input).cpu().numpy().flatten()

    # Plot using synthetic feature values
    plt.figure(figsize=(8, 5))
    plt.plot(synthetic_feature_values.cpu().numpy(), outputs, label="Model Output", linestyle='-', marker='.')
    plt.plot(synthetic_feature_values.cpu().numpy(), synthetic_feature_values.cpu().numpy(), label="Synthetic Feature", linestyle='--', alpha=0.6)
    plt.xlabel(f"{feature_name} (synthetic 0 → 1)")
    plt.ylabel("Model Output")
    plt.title(f"Model Sensitivity to {feature_name} (Synthetic Sweep)")
    plt.grid(True)
    plt.legend()
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

def plot_all_feature_correlations(X_tensor, feature_names, method="pearson"):
    """
    Plot a heatmap of pairwise feature correlations using constrained layout to avoid label cutoff.
    """
    X_np = X_tensor.cpu().numpy()
    df = pd.DataFrame(X_np, columns=feature_names)
    corr_matrix = df.corr(method=method)

    # Use constrained_layout for automatic margin handling
    fig, ax = plt.subplots(figsize=(14, 12), constrained_layout=True)
    sns.heatmap(corr_matrix, annot=False, fmt=".2f", cmap="coolwarm", square=True,
                xticklabels=feature_names, yticklabels=feature_names, cbar_kws={"shrink": 0.75}, ax=ax)

    ax.set_title(f"{method.capitalize()} Correlation Matrix of Features", fontsize=16)
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.show()

def overlay_sorted_predictions_and_feature(model, X_test, Y_test, feature_name, feature_names, threshold=0.5):
    import matplotlib.pyplot as plt
    import numpy as np

    model.eval()
    with torch.no_grad():
        outputs = model.inference_raw(X_test).cpu().numpy().flatten()

    true_labels = Y_test.cpu().numpy().flatten()
    feature_values = X_test[:, feature_names.index(feature_name)].cpu().numpy()

    # Sort by predicted scores (to match plot_sorted_predictions_with_labels)
    # sorted_indices = np.argsort(outputs)
    # sorted_outputs = outputs[sorted_indices]
    # sorted_labels = true_labels[sorted_indices]
    # sorted_feature_values = feature_values[sorted_indices]
    sorted_indices = np.argsort(feature_values)
    sorted_feature_values = feature_values[sorted_indices]
    sorted_outputs = outputs[sorted_indices]
    sorted_labels = true_labels[sorted_indices]


    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Plot predictions
    for i, (score, label) in enumerate(zip(sorted_outputs, sorted_labels)):
        color = 'red' if label == 1 else 'blue'
        ax1.plot(i, score, marker='o', color=color, alpha=0.6)

    ax1.axhline(y=threshold, color='green', linestyle='dotted', label=f'Threshold = {threshold}')
    ax1.set_xlabel("Sample Index (sorted by predicted score)")
    ax1.set_ylabel("Predicted Score", color='black')
    ax1.tick_params(axis='y', labelcolor='black')

    # Plot feature value on secondary axis
    ax2 = ax1.twinx()
    ax2.plot(range(len(sorted_feature_values)), sorted_feature_values, label=f"{feature_name} value", color='purple', linestyle='--', alpha=0.6)
    ax2.set_ylabel(f"{feature_name}", color='purple')
    ax2.tick_params(axis='y', labelcolor='purple')

    # Titles and legends
    plt.title(f"Overlay: Sorted Prediction Scores & {feature_name}")
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.grid(True, linestyle='--', linewidth=0.5)
    plt.tight_layout()
    plt.show()


def plot_feature_pruning_analysis(accuracies, baseline_features=None, title="Accuracy vs. Number of Features Pruned", filename=None):
    """Plot accuracy vs number of features pruned.
    
    Args:
        accuracies: List of accuracies [baseline, after_pruning_1, after_pruning_2, ...]
        baseline_features: List of baseline feature indices (optional, for annotation)
        title: Plot title
        filename: If provided, save plot to this file
    """
    plt.figure(figsize=(10, 5))
    x_values = list(range(len(accuracies)))
    plt.plot(x_values, [a * 100 for a in accuracies], marker='o', linewidth=2)
    
    # Annotate baseline if present
    if baseline_features and len(baseline_features) > 0:
        baseline_note = f" (Baseline: {len(baseline_features)} features not pruned)"
        plt.title(title + baseline_note)
    else:
        plt.title(title)
    
    plt.xlabel("Number of Features Pruned from Left (0 = No Pruning)")
    plt.ylabel("Accuracy (%)")
    plt.grid(True, alpha=0.3)
    plt.xticks(x_values)  # Show all tick marks
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=150)
    
    plt.show()


def plot_feature_growing_analysis(accuracies, f1_scores=None, title="Accuracy vs. Number of Features (Growing)", filename=None):
    """Plot accuracy vs number of features as tree grows from baseline.
    
    Args:
        accuracies: List of accuracies [baseline_2_features, after_adding_feature_2, after_adding_feature_3, ...]
        f1_scores: List of F1 scores (optional)
        title: Plot title
        filename: If provided, save plot to this file
    """
    plt.figure(figsize=(10, 5))
    # X-axis represents number of features: starts at 2 (baseline), then 3, 4, 5, ...
    num_features = list(range(2, 2 + len(accuracies)))
    plt.plot(num_features, [a * 100 for a in accuracies], marker='o', linewidth=2, color='green', label='Accuracy')
    
    # Plot F1 scores as dotted line if provided
    if f1_scores is not None and len(f1_scores) == len(accuracies):
        plt.plot(num_features, [f * 100 for f in f1_scores], marker='s', linewidth=2, linestyle='--', color='blue', label='F1 Score')
        plt.legend()
    
    plt.title(title)
    plt.xlabel("Number of Features (Growing from Baseline)")
    plt.ylabel("Score (%)")
    plt.grid(True, alpha=0.3)
    # Show x-axis labels every 5 samples
    xtick_positions = [n for n in num_features if (n - 2) % 5 == 0 or n == num_features[0] or n == num_features[-1]]
    plt.xticks(xtick_positions)
    plt.tight_layout()
    
    if filename:
        plt.savefig(filename, dpi=150)
    
    plt.show()

