import torch
import torch.nn as nn
import torch.optim as optim
import itertools
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

# 🔥 Generalized GCD Operator
def generalized_gcd(w1, w2, lambd):
    lambd = torch.sigmoid(lambd)  # Ensure lambda is between 0 and 1
    epsilon = 1e-6  # Small value to prevent zero exponentiation errors

    w1_safe = torch.abs(w1) + epsilon
    w2_safe = torch.abs(w2) + epsilon

    return (w1_safe ** lambd) * (w2_safe ** (1 - lambd)) + (1 - lambd) * torch.max(w1_safe, w2_safe)

# 🔹 Fully Connected Input-to-Leaf Mapping
class InputToLeafFC(nn.Module):
    def __init__(self, num_inputs, num_leaves, init_sharpness=1.0, max_sharpness=10.0, bias_shift=-2.0):
        super(InputToLeafFC, self).__init__()
        self.num_inputs = num_inputs
        self.num_leaves = num_leaves
        self.init_sharpness = init_sharpness
        self.max_sharpness = max_sharpness
        self.bias_shift = bias_shift
        self.sharpness = nn.Parameter(torch.tensor(init_sharpness, dtype=torch.float32), requires_grad=False)

        # Fully connected layer
        self.fc = nn.Linear(num_inputs, num_leaves, bias=False)

    def increase_sharpness(self, factor=1.01):
        """ Gradually increases sharpness factor over training. """
        with torch.no_grad():
            self.sharpness *= factor
            self.sharpness.clamp_(max=self.max_sharpness)  # Prevent explosion

    def forward(self, x):
        """
        Computes input-to-leaf connections and applies a sharp but trainable squashing function.
        """
        raw_outputs = self.fc(x)
        activated_values = torch.sigmoid(self.sharpness * (raw_outputs + self.bias_shift))  # 🔥 Adaptive transition
        return activated_values

# 🔹 Binary Tree Logic Network
class BinaryTreeLogicNet(nn.Module):
    def __init__(self, input_size, weight_mode="trainable"):
        super(BinaryTreeLogicNet, self).__init__()
        self.original_input_size = input_size
        self.num_leaves = input_size  # 🔹 Each input gets its own leaf initially
        self.weight_mode = weight_mode

        # 🔹 Fully Connected Input-to-Leaf Mapping
        self.input_to_leaf = InputToLeafFC(input_size, self.num_leaves)

        # Weights and Biases
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        self.num_layers = self.num_leaves - 1  # Leaf nodes feed into binary tree

        for _ in range(self.num_layers):
            self.weights.append(nn.Parameter(torch.randn(2) * 0.1))
            self.biases.append(nn.Parameter(torch.rand(1) * 0.1))

        self.fc_out = nn.Linear(1, 1)
        self.apply(self.initialize_weights)

    def initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.5)  

    def forward(self, x, return_all_layers=False):
        # 🔹 Compute input-to-leaf values
        leaf_values = self.input_to_leaf(x)
        node_outputs = list(leaf_values.T)  
        layer_outputs = []  

        for i in range(self.num_layers):
            w = self.weights[i]
            bias = self.biases[i]
            node_outputs.append(generalized_gcd(w[0] * node_outputs.pop(0), w[1] * node_outputs.pop(0), bias))
            layer_outputs.append(node_outputs[-1])

        final_output = torch.sigmoid(self.fc_out(layer_outputs[-1].unsqueeze(1)))
        return (final_output, layer_outputs) if return_all_layers else final_output

# 🔥 Generate Training Data
def generate_data(repeat_factor=100):
    data = []
    labels = []
    base_cases = list(itertools.product([0, 1], repeat=3))  # Truth table

    for _ in range(repeat_factor):
        for a, b, c in base_cases:
            y = a and (b or c)  # Compute function
            data.append([a, b, c])
            labels.append([y])  # Ensure correct shape

    return torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

# 🔥 Train Model
def train_model(model, X_train, Y_train, epochs=5000):
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, Y_train)

        if torch.isnan(loss):
            print(f"⚠️ NaN detected in loss at epoch {epoch}! Skipping update.")
            continue

        loss.backward()
        optimizer.step()

        # 🔹 Increase sharpness slightly every 1000 epochs
        if (epoch + 1) % 1000 == 0:
            model.input_to_leaf.increase_sharpness(factor=1.02)

        if (epoch + 1) % 1000 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Sharpness: {model.input_to_leaf.sharpness.item():.2f}")

    return model

# 🔹 Pick Strongest Edge Per Leaf After Training
def prune_to_strongest_edge(model):
    with torch.no_grad():
        fc_weights = model.input_to_leaf.fc.weight.numpy()
    
    strongest_connections = np.argmax(fc_weights, axis=1)  # Find strongest input per leaf
    
    pruned_weights = np.zeros_like(fc_weights)
    for i, idx in enumerate(strongest_connections):
        pruned_weights[i, idx] = 1.0  # Keep only the strongest connection

    model.input_to_leaf.fc.weight.data = torch.tensor(pruned_weights, dtype=torch.float32)

def visualize_binary_tree(model):
    G = nx.DiGraph()
    
    with torch.no_grad():
        fc_weights = model.input_to_leaf.fc.weight.numpy()
    
    vars = ["A", "B", "C"]
    
    # Step 1: Start with initial aggregators
    aggregators = {var: var for var in vars}  # Track aggregator parents
    aggregator_count = 0
    
    # Step 2: Connect leaves to aggregators
    leaf_nodes = []
    for i, row in enumerate(fc_weights):
        strongest_input = vars[np.argmax(row)]
        leaf_name = f"Leaf {i+1}"
        agg_name = f"Agg{aggregator_count}"
        
        G.add_edge(aggregators[strongest_input], agg_name)
        G.add_edge(agg_name, leaf_name)
        aggregators[strongest_input] = agg_name  # Update the chain
        leaf_nodes.append(leaf_name)
        aggregator_count += 1
    
    # Step 3: Build a hierarchical binary tree
    while len(aggregators) > 1:
        keys = list(aggregators.keys())
        new_aggregators = {}
        
        for i in range(0, len(keys), 2):
            if i + 1 < len(keys):
                parent1 = aggregators[keys[i]]
                parent2 = aggregators[keys[i + 1]]
                new_agg = f"Agg{aggregator_count}"
                
                G.add_edge(parent1, new_agg)
                G.add_edge(parent2, new_agg)
                new_aggregators[new_agg] = new_agg
                aggregator_count += 1
            else:
                # Carry forward unpaired aggregator
                new_aggregators[keys[i]] = aggregators[keys[i]]
        
        aggregators = new_aggregators
    
    # Step 4: Generate a structured tree layout
    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_size=3000, node_color="lightblue", font_size=10, edge_color="black", width=1.5, arrows=True)
    plt.title("Hierarchical Binary Tree Structure - AI Model")
    plt.show()

# 🔥 Run Training & Visualization
if __name__ == "__main__":
    model = BinaryTreeLogicNet(3, weight_mode="trainable")
    model = train_model(model, *generate_data(repeat_factor=100), epochs=5000)

    # 🔹 Select only the strongest edge per leaf after training
    prune_to_strongest_edge(model)

    # 🔹 Visualize the pruned binary tree structure
    visualize_binary_tree(model)
