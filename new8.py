import random
import numpy as np

# Generalized GCD operator with an andness parameter a
def gcd_operator(a, x, y):
    return a * min(x, y) + (1 - a) * max(x, y)

# Variable constraints (each variable must appear within this range)
VARIABLE_RANGES = {"x1": (0, 2), "x2": (1, 2)}  # Example: x1 can appear 0-2 times, x2 must appear 1-2 times
DEPTH_PENALTY = 0.01  # Slight penalty for deeper trees

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

# Generate a constrained random binary tree
def generate_constrained_tree(variables, depth=0, max_depth=3, required_counts=None):
    """Generate a tree ensuring variable occurrences within the given range."""
    if required_counts is None:
        required_counts = {var: random.randint(rng[0], rng[1]) for var, rng in VARIABLE_RANGES.items()}

    # If we reach the maximum depth or only one variable is left to be placed, return a valid variable
    remaining_vars = [v for v, count in required_counts.items() if count > 0]
    if depth >= max_depth or len(remaining_vars) == 1:
        var = random.choice(remaining_vars)
        required_counts[var] -= 1
        return var

    left_counts = required_counts.copy()
    right_counts = required_counts.copy()

    # Select a variable to split, prioritizing those still above their minimum requirement
    splittable_vars = [v for v in variables if required_counts[v] > 1]
    if not splittable_vars:  # If no variables can be split, choose from remaining ones
        splittable_vars = remaining_vars

    split_var = random.choice(splittable_vars)
    left_counts[split_var] -= 1
    right_counts[split_var] -= 1

    left = generate_constrained_tree(variables, depth + 1, max_depth, left_counts)
    right = generate_constrained_tree(variables, depth + 1, max_depth, right_counts)
    a = random.uniform(0.8, 1.0)  # Favoring "min" to represent AND

    return (a, left, right)

# Example Usage
if __name__ == "__main__":
    variables = ["x1", "x2"]

    # Test tree generation to check for errors
    for _ in range(5):
        tree = generate_constrained_tree(variables)
        print(tree)
