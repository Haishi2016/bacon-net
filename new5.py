import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Generalized GCD operator with an andness parameter a
def gcd_operator(a, x, y):
    return a * min(x, y) + (1 - a) * max(x, y)

# Limit variable occurrences
MAX_VAR_OCCURRENCES = 2  # Adjust this to limit variable appearances

def count_variables(tree):
    """Count occurrences of each variable in a tree."""
    if isinstance(tree, str):  # Leaf node
        return {tree: 1}
    
    a, left, right = tree
    left_count = count_variables(left)
    right_count = count_variables(right)

    # Merge counts
    counts = {}
    for var in set(left_count.keys()).union(right_count.keys()):
        counts[var] = left_count.get(var, 0) + right_count.get(var, 0)
    return counts

# Generate a constrained random binary tree
def generate_random_tree(variables, depth=0, max_depth=3):
    """Generate a tree ensuring variable constraints."""
    if depth >= max_depth or len(variables) == 1:
        return random.choice(variables)

    left = generate_random_tree(variables, depth + 1, max_depth)
    right = generate_random_tree(variables, depth + 1, max_depth)
    a = random.uniform(0, 1)

    return (a, left, right)

# Mutate tree while ensuring variable constraints
def mutate_tree(tree, mutation_rate=0.2):
    """Mutate a tree while ensuring variable limits."""
    if isinstance(tree, str):  # Leaf node
        return tree

    a, left, right = tree

    # Mutate 'a' slightly
    if random.random() < mutation_rate:
        a = max(0, min(1, a + random.uniform(-0.1, 0.1)))

    # Swap subtrees
    if random.random() < mutation_rate:
        left, right = right, left

    return (a, mutate_tree(left, mutation_rate), mutate_tree(right, mutation_rate))

# Prune redundant subtrees
def prune_tree(tree):
    """Recursively simplify the tree."""
    if isinstance(tree, str):
        return tree

    a, left, right = tree
    left = prune_tree(left)
    right = prune_tree(right)

    # Remove duplicate subexpressions
    if left == right:
        return left

    # If a variable appears more than allowed, replace it
    var_counts = count_variables(tree)
    for var, count in var_counts.items():
        if count > MAX_VAR_OCCURRENCES:
            return random.choice([left, right])  # Prune excess occurrences

    return (a, left, right)

# Convert tree to human-readable format
def tree_to_expression(tree):
    """Convert tree structure to readable expression."""
    if isinstance(tree, str):
        return tree
    
    a, left, right = tree
    left_expr = tree_to_expression(left)
    right_expr = tree_to_expression(right)
    
    return f"{a:.2f} * min({left_expr}, {right_expr}) + {1-a:.2f} * max({left_expr}, {right_expr})"

# Genetic Algorithm
def evolve_trees(variables, dataset, generations=50, population_size=20, mutation_rate=0.2):
    """Evolve trees and track complexity."""
    population = [generate_random_tree(variables) for _ in range(population_size)]
    complexity_history = []

    for gen in range(generations):
        # Evaluate fitness
        fitness_scores = [(tree, -np.mean([(evaluate_tree(tree, x) - y) ** 2 for x, y in dataset])) for tree in population]
        fitness_scores.sort(key=lambda x: x[1], reverse=True)

        best_tree = fitness_scores[0][0]
        complexity_history.append(tree_depth(best_tree))

        print(f"Generation {gen+1}: Best fitness = {fitness_scores[0][1]:.4f}")
        print(f"  Best Tree: {tree_to_expression(best_tree)}\n")

        # Selection and reproduction
        survivors = [tree for tree, _ in fitness_scores[:population_size // 2]]
        new_population = survivors.copy()
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(survivors, 2)
            child = mutate_tree((0.5, parent1, parent2), mutation_rate)
            new_population.append(child)

        population = new_population

    # Prune the best tree
    return prune_tree(best_tree), complexity_history

# Tree depth function
def tree_depth(tree):
    """Compute depth of tree."""
    if isinstance(tree, str):
        return 0
    return 1 + max(tree_depth(tree[1]), tree_depth(tree[2]))

# Evaluate tree function
def evaluate_tree(tree, values):
    """Evaluate the tree."""
    if isinstance(tree, str):
        return values.get(tree, 0)
    a, left, right = tree
    return gcd_operator(a, evaluate_tree(left, values), evaluate_tree(right, values))

# Plot evolution of tree complexity
def plot_complexity(complexity_history):
    """Plot complexity over generations."""
    plt.plot(complexity_history, marker='o')
    plt.xlabel("Generation")
    plt.ylabel("Tree Depth")
    plt.title("Evolution of Tree Complexity")
    plt.show()

# Visualize tree structure
def draw_tree(tree, graph=None, parent=None, label="root"):
    """Generate a visual representation of the tree using networkx."""
    if graph is None:
        graph = nx.DiGraph()

    node_label = label if isinstance(tree, str) else f"{tree[0]:.2f}"
    graph.add_node(node_label)

    if parent is not None:
        graph.add_edge(parent, node_label)

    if not isinstance(tree, str):
        draw_tree(tree[1], graph, node_label, "L")
        draw_tree(tree[2], graph, node_label, "R")

    return graph

# Example Usage
if __name__ == "__main__":
    variables = ["x1", "x2", "x3", "x4"]
    dataset = [
        ({"x1": 0.1, "x2": 0.7, "x3": 0.4, "x4": 0.8}, 0.5),
        ({"x1": 0.9, "x2": 0.6, "x3": 0.3, "x4": 0.2}, 0.7),
        ({"x1": 0.4, "x2": 0.5, "x3": 0.7, "x4": 0.1}, 0.6),
    ]

    best_tree, complexity_history = evolve_trees(variables, dataset)
    plot_complexity(complexity_history)

    print("\n✅ Best Evolved Expression:", tree_to_expression(best_tree))
    graph = draw_tree(best_tree)
    plt.figure(figsize=(10, 6))
    nx.draw(graph, with_labels=True, node_color="lightblue", edge_color="gray", node_size=2000, font_size=10)
    plt.show()
