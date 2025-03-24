import random
import numpy as np
import itertools

# Generalized GCD operator with an andness parameter a
def gcd_operator(a, x, y):
    """Generalized GCD function with an 'andness' parameter."""
    return a * min(x, y) + (1 - a) * max(x, y)

# Variable constraints (each variable must appear within this range)
VARIABLES = ["x1", "x2", "x3", "x4"]
VARIABLE_RANGES = {var: (1, 2) for var in VARIABLES}
DEPTH_PENALTY = 0.01  # Slight penalty for deeper trees
VAR_PENALTY = 0.2  # Strong penalty for violating min/max variable appearances

# Generate all pairwise initial trees
def generate_initial_population(variables):
    """Generate the initial population using pairwise variable combinations."""
    pairs = list(itertools.permutations(variables, 2))  # All pairwise permutations
    trees = [(random.uniform(0.0, 1.0), left, right) for left, right in pairs]  # Initial trees
    return trees

# Grow the tree by appending new variables **only to the right**
def grow_tree_right(tree, variables):
    """Expand a tree by appending a new variable to the right sequentially."""
    existing_vars = extract_variables(tree)
    available_vars = [v for v in variables if v not in existing_vars]

    if not available_vars:
        return tree  # No more variables to add

    new_var = random.choice(available_vars)  # Pick a new variable
    a = random.uniform(0.0, 1.0)  # Random weight

    return (a, tree, new_var)  # Append new variable **to the right**

# Count occurrences of each variable in a tree
def count_variables(tree):
    """Count occurrences of each variable in a tree."""
    if isinstance(tree, str):  # Leaf node
        return {tree: 1}
    
    a, left, right = tree
    left_count = count_variables(left)
    right_count = count_variables(right)

    counts = {}
    for var in set(left_count.keys()).union(right_count.keys()):
        counts[var] = left_count.get(var, 0) + right_count.get(var, 0)
    return counts

# Generate trees that grow by appending to the right
def generate_constrained_tree(variables):
    """Generate a sequential binary tree where new variables are added to the right."""
    trees = generate_initial_population(variables)  # Generate initial pairwise trees
    
    expanded_population = []
    for tree in trees:
        for _ in range(len(variables) - 2):  # Grow until all variables are included
            tree = grow_tree_right(tree, variables)
        expanded_population.append(tree)

    return expanded_population

# Mutate trees while preserving structure
def mutate_tree(tree, mutation_rate=0.2):
    """Mutate a tree while allowing full flexibility in logic discovery."""
    if isinstance(tree, str):  # Leaf node
        return tree

    a, left, right = tree

    # Mutate `a` within full range
    if random.random() < mutation_rate:
        a = max(0, min(1, a + random.uniform(-0.3, 0.3)))  # Allow full flexibility

    return (a, mutate_tree(left, mutation_rate), mutate_tree(right, mutation_rate))

# Extract all variables from a tree
def extract_variables(tree):
    """Extract all variables from a tree."""
    if isinstance(tree, str):
        return {tree}
    a, left, right = tree
    return extract_variables(left) | extract_variables(right)

# Compute fitness based on correctness
def fitness(tree, dataset):
    """Compute fitness based only on correctness, without biasing AND or OR."""
    errors = []
    for inputs, expected_output in dataset:
        predicted_output = evaluate_tree(tree, inputs)
        errors.append((predicted_output - expected_output) ** 2)

    mse = np.mean(errors)
    depth_penalty = DEPTH_PENALTY * tree_depth(tree)  # Penalize deeper trees

    return -(mse + depth_penalty)  # Negative because GA maximizes fitness

# Evaluate tree function
def evaluate_tree(tree, values):
    """Evaluate the tree."""
    if isinstance(tree, str):
        return values.get(tree, 0)
    a, left, right = tree
    return gcd_operator(a, evaluate_tree(left, values), evaluate_tree(right, values))

# Compute tree depth
def tree_depth(tree):
    """Compute depth of tree."""
    if isinstance(tree, str):
        return 0
    return 1 + max(tree_depth(tree[1]), tree_depth(tree[2]))

# Genetic Algorithm
def evolve_trees(variables, dataset, generations=500, population_size=20, mutation_rate=0.2):
    """Evolve trees and track complexity."""
    population = generate_constrained_tree(variables)

    for gen in range(generations):
        # Evaluate fitness
        fitness_scores = [(tree, fitness(tree, dataset)) for tree in population]
        fitness_scores.sort(key=lambda x: x[1], reverse=True)

        best_tree = fitness_scores[0][0]

        print(f"Generation {gen+1}: Best fitness = {fitness_scores[0][1]:.4f}")
        print(f"  Best Tree: {tree_to_expression(best_tree)}\n")

        # Selection and reproduction (Elitism: keep the best tree)
        survivors = [tree for tree, _ in fitness_scores[:population_size // 2]]
        new_population = survivors.copy()
        while len(new_population) < population_size:
            parent = random.choice(survivors)
            child = mutate_tree(parent, mutation_rate)
            new_population.append(child)

        population = new_population

    return best_tree

# Recursively simplify gcd(a, x, x) → x inside the tree
def simplify_tree(tree):
    """Recursively simplify the tree, replacing gcd(a, x, x) with x."""
    if isinstance(tree, str):  # Base case: if it's a variable, return it
        return tree

    a, left, right = tree

    # **Recursively simplify left and right subtrees first**
    left = simplify_tree(left)
    right = simplify_tree(right)

    # **If both sides are the same, replace gcd(a, x, x) → x**
    if left == right:
        return left  # Collapse redundant `gcd(x, x)`

    return (a, left, right)  # Return simplified tree

# Convert tree to human-readable format after simplifying
def tree_to_expression(tree):
    """Convert a simplified tree structure to a readable expression."""
    if isinstance(tree, str):
        return tree  # Base case: return variable name

    a, left, right = tree

    return f"gcd({a:.2f}, {tree_to_expression(left)}, {tree_to_expression(right)})"

# Generate additional samples using Gaussian noise
def augment_dataset(dataset, num_augmented=100, std_dev=0.01):
    """Augment dataset by adding Gaussian noise to input values."""
    augmented_data = []
    for _ in range(num_augmented):
        base_sample, label = random.choice(dataset)
        new_sample = {key: np.clip(np.random.normal(value, std_dev), 0, 1) for key, value in base_sample.items()}
        augmented_data.append((new_sample, label))

    return dataset + augmented_data

# Example Usage
if __name__ == "__main__":
    variables = ["x1", "x2", "x3"]
    dataset = [
        ({"x1": 0, "x2": 0, "x3": 0}, 0),
        ({"x1": 0, "x2": 0, "x3": 1}, 0),
        ({"x1": 0, "x2": 1, "x3": 0}, 0),
        ({"x1": 0, "x2": 1, "x3": 1}, 1),
        ({"x1": 1, "x2": 0, "x3": 0}, 0),
        ({"x1": 1, "x2": 0, "x3": 1}, 1),
        ({"x1": 1, "x2": 1, "x3": 0}, 0),
        ({"x1": 1, "x2": 1, "x3": 1}, 1),
    ]
    augmented_dataset = augment_dataset(dataset, num_augmented=100)
    best_tree = evolve_trees(variables, augmented_dataset)
    best_tree = simplify_tree(best_tree)  
    print("\n✅ Best Evolved Expression:", tree_to_expression(best_tree))
