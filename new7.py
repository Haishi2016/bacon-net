import random
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# Generalized GCD operator with an andness parameter a
def gcd_operator(a, x, y):
    return a * min(x, y) + (1 - a) * max(x, y)

# Limit variable occurrences
MAX_VAR_OCCURRENCES = 1  # Each variable can appear only once
DEPTH_PENALTY = 0.01  # Slight penalty for depth, allowing evolution of AND
VARIABLE_USAGE_BONUS = 0.05  # Reward trees that use both variables

# Count variable occurrences
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

# Generate a constrained random binary tree
def generate_random_tree(variables, depth=0, max_depth=3):
    """Generate a tree ensuring variable constraints."""
    if depth >= max_depth or len(variables) == 1:
        return random.choice(variables)

    left = generate_random_tree(variables, depth + 1, max_depth)
    right = generate_random_tree(variables, depth + 1, max_depth)
    a = random.uniform(0.8, 1.0)  # Favoring "min" to represent AND

    return (a, left, right)

# Mutate tree while ensuring variable constraints
def mutate_tree(tree, mutation_rate=0.2):
    """Mutate a tree while ensuring variable limits."""
    if isinstance(tree, str):  # Leaf node
        return tree

    a, left, right = tree

    # Mutate 'a' slightly, but keep it favoring AND
    if random.random() < mutation_rate:
        a = max(0.8, min(1, a + random.uniform(-0.05, 0.05)))  # Favoring AND

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
    
    return f"gcd({a:.2f}, {left_expr}, {right_expr})"

# Compute tree depth
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

# Fitness function with depth penalty and variable usage bonus
def fitness(tree, dataset):
    """Compute fitness with penalty for deep trees and reward for using both variables."""
    errors = []
    for inputs, expected_output in dataset:
        predicted_output = evaluate_tree(tree, inputs)
        errors.append((predicted_output - expected_output) ** 2)

    mse = np.mean(errors)
    depth_penalty = DEPTH_PENALTY * tree_depth(tree)  # Penalize deeper trees

    # Reward trees that use both x1 and x2
    var_counts = count_variables(tree)
    variable_bonus = VARIABLE_USAGE_BONUS if len(var_counts) > 1 else -0.1

    return -(mse + depth_penalty - variable_bonus)  # Negative because GA maximizes fitness

# Genetic Algorithm
def evolve_trees(variables, dataset, generations=1000, population_size=20, mutation_rate=0.2):
    """Evolve trees and track complexity."""
    population = [generate_random_tree(variables) for _ in range(population_size)]
    complexity_history = []

    for gen in range(generations):
        # Evaluate fitness
        fitness_scores = [(tree, fitness(tree, dataset)) for tree in population]
        fitness_scores.sort(key=lambda x: x[1], reverse=True)

        best_tree = fitness_scores[0][0]
        complexity_history.append(tree_depth(best_tree))

        print(f"Generation {gen+1}: Best fitness = {fitness_scores[0][1]:.4f}")
        print(f"  Best Tree: {tree_to_expression(best_tree)}\n")

        # Selection and reproduction (Elitism: keep the best tree)
        survivors = [tree for tree, _ in fitness_scores[:population_size // 2]]
        new_population = survivors.copy()
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(survivors, 2)
            child = mutate_tree((0.9, parent1, parent2), mutation_rate)  # Favoring AND
            new_population.append(child)

        population = new_population

    # Prune the best tree
    return prune_tree(best_tree), complexity_history

# Example Usage
if __name__ == "__main__":
    variables = ["x1", "x2"]
    dataset = [
        ({"x1": 1, "x2": 1}, 1),
        ({"x1": 0, "x2": 1}, 0),
        ({"x1": 1, "x2": 0}, 0),
        ({"x1": 0, "x2": 0}, 0),
    ]

    best_tree, complexity_history = evolve_trees(variables, dataset)

    print("\n✅ Best Evolved Expression:", tree_to_expression(best_tree))
