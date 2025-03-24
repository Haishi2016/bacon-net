import random
import numpy as np

# Generalized GCD operator with an andness parameter a
def gcd_operator(a, x, y):
    return a * min(x, y) + (1 - a) * max(x, y)

# Generate a random binary tree with parameters
def generate_random_tree(variables, depth=3):
    """Generate a random binary tree with random 'andness' values (a)."""
    if depth == 0 or len(variables) == 1:
        return random.choice(variables)  # Return a variable as a leaf node

    left = generate_random_tree(variables, depth-1)
    right = generate_random_tree(variables, depth-1)
    a = random.uniform(0, 1)  # Initialize andness parameter randomly

    return (a, left, right)

# Evaluate a tree with given input values
def evaluate_tree(tree, values):
    """Recursively evaluate the fuzzy Boolean tree with given variable values."""
    if isinstance(tree, str):  # Leaf node (variable)
        return values[tree]  # Look up the variable value
    else:
        a, left, right = tree
        return gcd_operator(a, evaluate_tree(left, values), evaluate_tree(right, values))

# Convert tree structure to human-readable expression
def tree_to_expression(tree):
    """Convert tree structure to a readable mathematical expression."""
    if isinstance(tree, str):  # Leaf node
        return tree
    
    a, left, right = tree
    left_expr = tree_to_expression(left)
    right_expr = tree_to_expression(right)
    
    return f"{a:.2f} * min({left_expr}, {right_expr}) + {1-a:.2f} * max({left_expr}, {right_expr})"

# Mutation: Change parameter a, or swap subtrees
def mutate_tree(tree, mutation_rate=0.2):
    """Mutate a tree by adjusting andness parameter a or swapping subtrees."""
    if isinstance(tree, str):  # Leaf node, no mutation
        return tree

    a, left, right = tree

    if random.random() < mutation_rate:
        # Mutate 'a' slightly
        a = max(0, min(1, a + random.uniform(-0.1, 0.1)))  # Ensure a stays in [0,1]

    if random.random() < mutation_rate:
        # Swap subtrees
        return (a, right, left)

    return (a, mutate_tree(left, mutation_rate), mutate_tree(right, mutation_rate))

# Crossover: Swap subtrees and blend a-values
def crossover_trees(tree1, tree2):
    """Perform subtree crossover and blend 'a' values."""
    if isinstance(tree1, str) or isinstance(tree2, str):  # If either is a leaf, return one
        return tree1 if random.random() < 0.5 else tree2

    a1, left1, right1 = tree1
    a2, left2, right2 = tree2

    if random.random() < 0.5:
        return ((a1 + a2) / 2, crossover_trees(left1, left2), right1)
    else:
        return ((a1 + a2) / 2, left1, crossover_trees(right1, right2))

# Fitness function: Mean Squared Error (MSE)
def fitness(tree, dataset):
    """Compute how well the tree matches expected outputs in the dataset."""
    errors = []
    for inputs, expected_output in dataset:
        predicted_output = evaluate_tree(tree, inputs)
        errors.append((predicted_output - expected_output) ** 2)
    return -np.mean(errors)  # Negative because GA maximizes fitness

# Genetic Algorithm Main Loop
def evolve_trees(variables, dataset, generations=50, population_size=20, mutation_rate=0.2):
    """Evolve binary trees using genetic algorithms, optimizing both structure and andness parameter 'a'."""
    # Initialize population
    population = [generate_random_tree(variables) for _ in range(population_size)]
    
    for gen in range(generations):
        # Evaluate fitness
        fitness_scores = [(tree, fitness(tree, dataset)) for tree in population]
        fitness_scores.sort(key=lambda x: x[1], reverse=True)  # Sort by fitness

        best_tree = fitness_scores[0][0]
        best_expression = tree_to_expression(best_tree)

        print(f"Generation {gen+1}: Best fitness = {fitness_scores[0][1]:.4f}")
        print(f"  Best Tree: {best_expression}\n")

        # Select top individuals
        survivors = [tree for tree, _ in fitness_scores[:population_size // 2]]

        # Generate new population via crossover and mutation
        new_population = survivors.copy()
        while len(new_population) < population_size:
            parent1, parent2 = random.sample(survivors, 2)
            child = crossover_trees(parent1, parent2)
            child = mutate_tree(child, mutation_rate)
            new_population.append(child)

        population = new_population  # Replace with new generation

    # Return the best tree found
    return fitness_scores[0][0]

# Example Usage
if __name__ == "__main__":
    # Define variables
    variables = ["x1", "x2", "x3", "x4"]

    # Create a dataset (inputs -> expected fuzzy output)
    dataset = [
        ({"x1": 0.1, "x2": 0.7, "x3": 0.4, "x4": 0.8}, 0.5),
        ({"x1": 0.9, "x2": 0.6, "x3": 0.3, "x4": 0.2}, 0.7),
        ({"x1": 0.4, "x2": 0.5, "x3": 0.7, "x4": 0.1}, 0.6),
    ]

    # Print target function (for reference)
    target_expression = "0.6 * min(x1, x2) + 0.4 * max(x3, x4)"  # Example target function
    print("\n🔹 Target Expression:", target_expression)

    # Run Genetic Algorithm to evolve best tree
    best_tree = evolve_trees(variables, dataset)
    
    # Print the best-evolved expression
    best_expression = tree_to_expression(best_tree)
    print("\n✅ Best Evolved Expression:", best_expression)
