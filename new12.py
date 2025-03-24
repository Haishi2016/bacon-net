import random
import numpy as np
import itertools

# Generalized GCD operator with an andness parameter a
def gcd_operator(a, x, y):
    return a * min(x, y) + (1 - a) * max(x, y)

# Variable constraints (each variable must appear within this range)
VARIABLES = ["x1", "x2", "x3", "x4"]
VARIABLE_RANGES = {var: (1, 2) for var in VARIABLES}
DEPTH_PENALTY = 0.01  # Slight penalty for deeper trees
VAR_PENALTY = 0.2  # Strong penalty for violating min/max variable appearances

def generate_initial_population(variables):
    """Generate the initial population using pairwise variable combinations."""
    pairs = list(itertools.permutations(variables, 2))  # All pairwise permutations
    trees = [(random.uniform(0.0, 1.0), left, right) for left, right in pairs]  # Initial trees
    return trees

def grow_tree_right(tree, variables):
    """Expand a tree by appending a new variable to the right."""
    existing_vars = extract_variables(tree)
    available_vars = [v for v in variables if v not in existing_vars]

    if not available_vars:
        return tree  # No more variables to add

    new_var = random.choice(available_vars)  # Pick a new variable
    a = random.uniform(0.0, 1.0)  # Random weight

    # Append the new variable **to the right**
    return (a, tree, new_var)

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

def generate_constrained_tree(variables, required_counts=None, depth=0, max_depth=3):
    """Generate a tree ensuring variable occurrences without forcing any specific feature."""
    if required_counts is None:
        required_counts = {var: random.randint(rng[0], rng[1]) for var, rng in VARIABLE_RANGES.items()}

    remaining_vars = [v for v, count in required_counts.items() if count > 0]
    if depth >= max_depth or len(remaining_vars) == 1:
        var = random.choice(remaining_vars)
        required_counts[var] -= 1
        return var

    left_counts = required_counts.copy()
    right_counts = required_counts.copy()

    # Choose any variable dynamically
    split_var = random.choice(remaining_vars)
    left_counts[split_var] -= 1
    right_counts[split_var] -= 1

    left = generate_constrained_tree(variables, left_counts, depth + 1, max_depth)
    right = generate_constrained_tree(variables, right_counts, depth + 1, max_depth)

    # Allow `a` to be fully random to discover AND, OR, or hybrid logic
    a = random.uniform(0.0, 1.0)

    return (a, left, right)


def mutate_tree(tree, mutation_rate=0.2):
    """Mutate a tree while allowing full flexibility in logic discovery."""
    if isinstance(tree, str):  # Leaf node
        return tree

    a, left, right = tree

    # Allow a full mutation range for a (no AND/OR bias)
    if random.random() < mutation_rate:
        a = max(0, min(1, a + random.uniform(-0.3, 0.3)))  # Allow full flexibility

    # Swap subtrees occasionally
    if random.random() < mutation_rate:
        left, right = right, left

    return (a, mutate_tree(left, mutation_rate), mutate_tree(right, mutation_rate))


def extract_nodes(tree):
    """Extract all nodes from the tree (used for analyzing 'a' values)."""
    if isinstance(tree, str):  # Leaf node
        return []

    a, left, right = tree
    return [(a, left, right)] + extract_nodes(left) + extract_nodes(right)


def fitness(tree, dataset):
    """Compute fitness based only on correctness, without biasing AND or OR."""
    errors = []
    for inputs, expected_output in dataset:
        predicted_output = evaluate_tree(tree, inputs)
        errors.append((predicted_output - expected_output) ** 2)

    mse = np.mean(errors)
    depth_penalty = DEPTH_PENALTY * tree_depth(tree)  # Penalize deeper trees

    # **No preference for AND or OR, just correctness**
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
    population = [generate_constrained_tree(variables) for _ in range(population_size)]
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
            child = mutate_tree((0.9, parent1, parent2), mutation_rate)
            new_population.append(child)

        population = new_population

    return best_tree, complexity_history

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




# Function to generate additional samples using Gaussian noise
def augment_dataset(dataset, num_augmented=100, std_dev=0.01):
    augmented_data = []
    for _ in range(num_augmented):
        base_sample, label = random.choice(dataset)  # Pick a random existing sample

        # Apply Gaussian noise
        new_x1 = np.clip(np.random.normal(base_sample["x1"], std_dev), 0, 1)  # Ensure x1 stays in [0,1]
        new_x2 = np.clip(np.random.normal(base_sample["x2"], std_dev), 0, 1)  # Ensure x2 stays in [0,1]
        new_x3 = np.clip(np.random.normal(base_sample["x3"], std_dev), 0, 1)  # Ensure x2 stays in [0,1]

        # Store augmented sample
        augmented_data.append(({"x1": new_x1, "x2": new_x2, "3": new_x3}, label))

    return dataset + augmented_data  # Combine original with augmented data


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
    # Generate augmented dataset
    augmented_dataset = augment_dataset(dataset, num_augmented=100)
    best_tree, complexity_history = evolve_trees(variables, augmented_dataset)
    best_tree = simplify_tree(best_tree)  # Simplify the best tree
    print("\n✅ Best Evolved Expression:", tree_to_expression(best_tree))
