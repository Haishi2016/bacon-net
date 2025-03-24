import random
import numpy as np

# Define available binary fuzzy operations
def fuzzy_and(x, y): return min(x, y)
def fuzzy_or(x, y): return max(x, y)
def fuzzy_avg(x, y): return (x + y) / 2  # Example of another possible operation

# List of available binary operators
OPERATORS = [fuzzy_and, fuzzy_or, fuzzy_avg]

# Generate a random binary tree
def generate_random_tree(variables, depth=3):
    """Generate a random binary tree using given variables and operators."""
    if depth == 0 or len(variables) == 1:
        return random.choice(variables)  # Return a variable as a leaf node
    
    left = generate_random_tree(variables, depth-1)
    right = generate_random_tree(variables, depth-1)
    operator = random.choice(OPERATORS)
    
    return (operator, left, right)

# Evaluate a tree with given input values
def evaluate_tree(tree, values):
    """Recursively evaluate the fuzzy Boolean tree with given variable values."""
    if isinstance(tree, str):  # Leaf node (variable)
        return values[tree]  # Look up the variable value
    else:
        operator, left, right = tree
        return operator(evaluate_tree(left, values), evaluate_tree(right, values))

# Mutation: Change operator or swap subtrees
def mutate_tree(tree, mutation_rate=0.2):
    """Mutate a tree by randomly changing an operator or swapping subtrees."""
    if isinstance(tree, str):  # Leaf node, no mutation
        return tree
    
    if random.random() < mutation_rate:
        # Change operator
        new_operator = random.choice(OPERATORS)
        return (new_operator, tree[1], tree[2])
    
    if random.random() < mutation_rate:
        # Swap subtrees
        return (tree[0], tree[2], tree[1])
    
    # Recur on left and right
    return (tree[0], mutate_tree(tree[1], mutation_rate), mutate_tree(tree[2], mutation_rate))

# Crossover: Swap subtrees between two parents
def crossover_trees(tree1, tree2):
    """Perform subtree crossover between two trees."""
    if isinstance(tree1, str) or isinstance(tree2, str):  # If either is a leaf, return one
        return tree1 if random.random() < 0.5 else tree2
    
    if random.random() < 0.5:
        return (tree1[0], crossover_trees(tree1[1], tree2[1]), tree1[2])
    else:
        return (tree1[0], tree1[1], crossover_trees(tree1[2], tree2[2]))

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
    """Evolve binary trees using genetic algorithms."""
    # Initialize population
    population = [generate_random_tree(variables) for _ in range(population_size)]
    
    for gen in range(generations):
        # Evaluate fitness
        fitness_scores = [(tree, fitness(tree, dataset)) for tree in population]
        fitness_scores.sort(key=lambda x: x[1], reverse=True)  # Sort by fitness

        print(f"Generation {gen+1}: Best fitness = {fitness_scores[0][1]:.4f}")

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

    # Run Genetic Algorithm to evolve best tree
    best_tree = evolve_trees(variables, dataset)
    
    print("\nBest evolved tree structure:", best_tree)
