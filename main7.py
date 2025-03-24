import torch
import torch.nn as nn
import torch.optim as optim
import itertools
import numpy as np

# 🔥 Generalized GCD Operator
def generalized_gcd(w1, w2, lambd):
    lambd = torch.sigmoid(lambd)  # Ensure lambda is between 0 and 1
    epsilon = 1e-6  # Small value to prevent zero exponentiation errors

    # Ensure non-negative values for exponentiation
    w1_safe = torch.abs(w1) + epsilon
    w2_safe = torch.abs(w2) + epsilon

    #return (w1_safe ** lambd) * (w2_safe ** (1 - lambd)) + (1 - lambd) * torch.max(w1_safe, w2_safe)
    #return lambd * torch.min(w1_safe, w2_safe) + (1 - lambd) * torch.max(w1_safe, w2_safe)
    # # **New: Weighted soft min/max to avoid dead gradients**
    # min_val = (w1_safe * w2_safe) ** (0.5 * lambd)  # Soft min
    # max_val = torch.max(w1_safe, w2_safe) ** (1 - 0.5 * lambd)  # Soft max

    # return lambd * min_val + (1 - lambd) * max_val
    return lambd * torch.min(w1_safe, w2_safe) + (1 - lambd) * torch.max(w1_safe, w2_safe)
    

class InputToLeafFC(nn.Module):
    def __init__(self, num_inputs, num_leaves, init_sharpness=1.0, max_sharpness=10.0, bias_shift=-2.0):
        """
        A fully connected input-to-leaf mapping with adaptive sharp activation behavior.
        - `init_sharpness`: Initial smooth activation.
        - `max_sharpness`: Maximum steepness reached over training.
        - `bias_shift`: Adjusts threshold for activation.
        """
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



# 🔹 Binary Tree Logic Network With Configurable Weights
class BinaryTreeLogicNet(nn.Module):
    def __init__(self, input_size, weight_mode="trainable", weight_value=1.0, weight_range=(0.5, 2.0), weight_choices=None):
        super(BinaryTreeLogicNet, self).__init__()
        self.original_input_size = input_size
        self.num_leaves = input_size  # 🔹 Each input gets its own leaf initially
        self.weight_mode = weight_mode
        self.weight_value = weight_value
        self.weight_range = weight_range
        self.weight_choices = torch.tensor(weight_choices, dtype=torch.float32) if weight_choices else None

        # 🔹 Fully Connected Input-to-Leaf Mapping
        self.input_to_leaf = InputToLeafFC(input_size, self.num_leaves)

        # Weights and Biases
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        self.num_layers = self.num_leaves - 1  # Leaf nodes feed into binary tree

        for _ in range(self.num_layers):
            if weight_mode == "fixed":
                self.weights.append(nn.Parameter(torch.tensor([weight_value, weight_value], dtype=torch.float32), requires_grad=False))
            elif weight_mode == "range":
                self.weights.append(nn.Parameter(torch.rand(2) * (weight_range[1] - weight_range[0]) + weight_range[0]))
            elif weight_mode == "discrete":
                self.weights.append(nn.Parameter(torch.choice(self.weight_choices, (2,)), requires_grad=True))
            else:  # "trainable"
                # self.weights.append(nn.Parameter(torch.randn(2) * 0.1))
                self.weights.append(nn.Parameter(torch.FloatTensor(2).uniform_(0.5, 1.5)))  # Avoid zero-centered values


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

    def print_tree_structure(self):
        """ Prints the binary tree structure with andness values (a) and weights on leaves going into GCD. """

        num_leaves = self.num_leaves
        internal_nodes = []
        leaf_connections = {}  # Stores weights applied to leaves before GCD

        # Simulate the tree-building process
        node_labels = [f"Leaf {i+1}" for i in range(num_leaves)]
        next_layer = node_labels.copy()

        for i in range(self.num_layers):
            # Retrieve andness value
            a_value = torch.sigmoid(self.biases[i]).item()
            
            # Retrieve weights applied in GCD operation
            w = self.weights[i].detach().numpy()
            
            # Merge two nodes with applied weights
            left = next_layer.pop(0)
            right = next_layer.pop(0)
            parent = f"Node {i+1} (andness: {a_value:.3f})"

            # Store the applied weights for leaf nodes
            if left.startswith("Leaf"):
                leaf_connections[left] = w[0]
            if right.startswith("Leaf"):
                leaf_connections[right] = w[1]

            internal_nodes.append((parent, left, right))
            next_layer.append(parent)

        # Recursively format the tree structure
        def format_tree(node, depth=0):
            if node.startswith("Leaf"):
                weight_info = f" [weight: {leaf_connections[node]:.3f}]" if node in leaf_connections else ""
                return f"{'  ' * depth}{node}{weight_info}"

            for parent, left, right in internal_nodes:
                if parent == node:
                    left_subtree = format_tree(left, depth + 1)
                    right_subtree = format_tree(right, depth + 1)
                    return f"{'  ' * depth}{parent}\n{left_subtree}\n{right_subtree}"

            return ""

        print("Binary Tree Structure:")
        print(format_tree(next_layer[0]))  # Root node




# 🔹 Prune Weak Connections
def prune_edges(model, threshold=0.1):
    """
    Removes weak input-leaf connections below a threshold.
    """
    with torch.no_grad():
        model.input_to_leaf.fc.weight *= (torch.abs(model.input_to_leaf.fc.weight) > threshold).float()

# 🔥 Generate Training Data
def generate_data(repeat_factor=100):
    data = []
    labels = []
    base_cases = list(itertools.product([0, 1], repeat=4))  # Truth table

    for _ in range(repeat_factor):
        for a, b, c, d in base_cases:
            # y = a and (b or c)  # Compute function
            y = a and b and c and d
            data.append([a, b, c, d])
            labels.append([y])  # Ensure correct shape

    return torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

def train_model(model, X_train, Y_train, epochs=50000):
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
            model.input_to_leaf.increase_sharpness(factor=1.01)

        if (epoch + 1) % 1000 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}, Sharpness: {model.input_to_leaf.sharpness.item():.2f}")

    return model


# 🔥 Train and Select Best Model
def train_and_select_best_model(weight_mode="trainable", weight_value=1.0, weight_range=(0.5, 2.0), weight_choices=None):
    X_train, Y_train = generate_data(repeat_factor=100)
    model = BinaryTreeLogicNet(X_train.shape[1], weight_mode, weight_value, weight_range, weight_choices)

    print(f"Training model with weight mode: {weight_mode}")
    model = train_model(model, X_train, Y_train, epochs=100000)

    # 🔹 Prune weak edges after training
    prune_edges(model, threshold=0.1)

    with torch.no_grad():
        outputs = model(X_train)
        loss = nn.BCELoss()(outputs, Y_train)

    return model, loss.item()

# 🔥 Print Estimated Logical Expression
def print_estimated_expression(model):
    with torch.no_grad():
        fc_weights = model.input_to_leaf.fc.weight.numpy()
    
    vars = ["A", "B", "C", "D"]
    
    for i, row in enumerate(fc_weights):
        # important_vars = [(vars[j], row[j]) for j in range(len(vars)) if abs(row[j]) >= 0.9]
        important_vars = [(vars[j], row[j]) for j in range(len(vars))]
        if important_vars:
            print(f"Leaf {i+1} receives input from:")
            for var, weight in important_vars:
                print(f"  - {var}: {weight:.3f}")  # Print variable name and weight
        else:
            print(f"Leaf {i+1} has no strong inputs")

# 🔥 Run Training & Visualization
if __name__ == "__main__":
    model, loss = train_and_select_best_model(weight_mode="fixed", weight_value=1.0)
    model.print_tree_structure()
    print(f"Final loss: {loss}")
    print_estimated_expression(model)
