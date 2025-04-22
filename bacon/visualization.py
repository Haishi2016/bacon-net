import networkx as nx
import matplotlib.pyplot as plt

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
    a = [b.item() for b in model.biases]
    indent = 2
    for i in range(model.num_layers):
        if i == 0:
            print(fmt_label(leaf_names[0]) + f"─{weights[0][0]:.2f}".rjust(5) + "────┐")
            new_leaf = leaf_names[1]
        else:
            new_leaf = leaf_names[i + 1]
        if classic_boolean:
            if a[i] >= 0.25:
                operator = "[ AND ]"
            else:
                operator = "[ O R ]"
        else:
            operator = f"[a={a[i]:.2f}]"
        if i < model.num_layers - 1:
            print(fmt_label(new_leaf) + f"─{weights[i][1]:.2f}".rjust(5) + "─" * indent +  operator + f"─{weights[i+1][0]:.2f}".rjust(5) + "────┐")
        else:
            print(fmt_label(new_leaf) + f"─{weights[i][1]:.2f}".rjust(5) + "─" * indent +  f"{operator}──OUTPUT")
        indent += 15
