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
    
def sinkhorn(log_alpha, n_iters=20, temperature=1.0):
    log_alpha = log_alpha / temperature
    A = torch.exp(log_alpha)

    for i in range(n_iters):
        A = A / A.sum(dim=1, keepdim=True)
        A = A / A.sum(dim=0, keepdim=True)

    return A

class FrozenInputToLeaf(nn.Module):
    def __init__(self, hard_assignment, num_inputs):
        super().__init__()
        self.register_buffer("P_hard", torch.zeros(len(hard_assignment), num_inputs))
        for leaf_idx, input_idx in enumerate(hard_assignment):
            self.P_hard[leaf_idx, input_idx] = 1.0

    def forward(self, x):
        return torch.matmul(x, self.P_hard.t())

class InputToLeafSinkhorn(nn.Module):
    def __init__(self, num_inputs, num_leaves, temperature=0.05, sinkhorn_iters=20):
        super(InputToLeafSinkhorn, self).__init__()
        self.num_inputs = num_inputs
        self.num_leaves = num_leaves
        self.temperature = temperature
        self.sinkhorn_iters = sinkhorn_iters

        # Learnable logits for permutation matrix
        self.logits = nn.Parameter(torch.randn(num_leaves, num_inputs))  # (leaves x inputs)

    def decrease_temperature(self, factor=0.98):
        self.temperature *= factor

    def forward(self, x):
        # x: (batch_size, num_inputs)
        batch_size = x.size(0)

        # Compute soft permutation matrix
        P = sinkhorn(self.logits, n_iters=self.sinkhorn_iters, temperature=self.temperature + 1e-6)  # (leaves x inputs)

        # Apply permutation matrix to input
        # x: (batch_size, inputs), P.T: (inputs x leaves) → (batch, leaves)
        return torch.matmul(x, P.t())



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
        self.input_to_leaf = InputToLeafSinkhorn(input_size, self.num_leaves)

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
            if i == 0:
                left = node_outputs[0]
                right = node_outputs[1]
            else:
                left = node_outputs[-1]  # previous node
                right = node_outputs[i + 1]  # next input
            node_outputs.append(generalized_gcd(w[0] * left, w[1] * right, bias))
            layer_outputs.append(node_outputs[-1])

        final_output = torch.sigmoid(self.fc_out(layer_outputs[-1].unsqueeze(1)))
        return (final_output, layer_outputs) if return_all_layers else final_output

    def print_tree_structure(self):
        """ Prints the recursive left-heavy binary tree structure correctly. """
        leaf_names = [f"Leaf {i+1}" for i in range(self.num_leaves)]
        leaf_connections = {}
        node_dict = {}
        node_labels = {}

        for i in range(self.num_layers):
            a_value = torch.sigmoid(self.biases[i]).item()
            w = self.weights[i].detach().numpy()

            if i == 0:
                left = leaf_names[0]
                right = leaf_names[1]
            else:
                left = f"Node{i}"
                right = leaf_names[i + 1]

            parent = f"Node{i+1}"
            label = f"{parent} (andness: {a_value:.3f})"
            node_dict[parent] = (left, right)
            node_labels[parent] = label

            # Track weights to leaves
            if left.startswith("Leaf"):
                leaf_connections[left] = w[0]
            if right.startswith("Leaf"):
                leaf_connections[right] = w[1]

        def format_tree(node, depth=0):
            indent = "  " * depth
            if node.startswith("Leaf"):
                w = leaf_connections.get(node, None)
                return f"{indent}{node}" + (f" [weight: {w:.3f}]" if w is not None else "")
            if node in node_dict:
                left, right = node_dict[node]
                label = node_labels.get(node, node)
                left_sub = format_tree(left, depth + 1)
                right_sub = format_tree(right, depth + 1)
                return f"{indent}{label}\n{left_sub}\n{right_sub}"
            return f"{indent}{node} [Unknown]"

        print("Binary Tree Structure:")
        root = f"Node{self.num_layers}"
        print(format_tree(root))


# 🔥 Generate Training Data
def generate_data(repeat_factor=100):
    data = []
    labels = []
    base_cases = list(itertools.product([0, 1], repeat=4))  # Truth table

    for _ in range(repeat_factor):
        for a, b, c, d in base_cases:
            y = a and (b or c) or d # Compute function
            # y = a and b and c and d
            data.append([a, b, c, d])
            labels.append([y])  # Ensure correct shape

    return torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

def train_model(model, X_train, Y_train, epochs=12000, freeze_epoch=5000):
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    criterion = nn.BCELoss()

    for epoch in range(epochs):
        if hasattr(model.input_to_leaf, "temperature") and (epoch + 1) % 1000 == 0:
            model.input_to_leaf.temperature *= 0.9

        optimizer.zero_grad()
        outputs = model(X_train)

        if torch.isnan(outputs).any():
            print("⚠️ NaN in outputs!")
        if (outputs < 0).any() or (outputs > 1).any():
            print("⚠️ Output out of range:", outputs.min().item(), outputs.max().item())

        loss = criterion(outputs, Y_train)

        if torch.isnan(loss):
            print(f"⚠️ NaN detected in loss at epoch {epoch}! Skipping update.")
            continue

        loss.backward()
        optimizer.step()

        if epoch == freeze_epoch:
            print("🧊 Freezing permutation at epoch", epoch)
            hard_perm = extract_hard_permutation(model)
            model.input_to_leaf = FrozenInputToLeaf(hard_perm, model.original_input_size)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}/{epochs} - Loss: {loss.item():.4f}")

    return model


def train_and_select_best_model(weight_mode="trainable", weight_value=1.0, weight_range=(0.5, 2.0), weight_choices=None, freeze_epoch=5000):
    X_train, Y_train = generate_data(repeat_factor=100)
    model = BinaryTreeLogicNet(X_train.shape[1], weight_mode, weight_value, weight_range, weight_choices)

    print(f"🚀 Training model with weight mode: {weight_mode}")
    model = train_model(model, X_train, Y_train, epochs=12000, freeze_epoch=freeze_epoch)

    with torch.no_grad():
        outputs = model(X_train)
        loss = nn.BCELoss()(outputs, Y_train)

    return model, loss.item()

    return model, loss.item()

def print_estimated_expression(model):
    if not hasattr(model.input_to_leaf, "logits"):
        print("🔒 Permutation is frozen. Skipping Sinkhorn inspection.\n")
        return

    with torch.no_grad():
        perm_matrix = sinkhorn(model.input_to_leaf.logits, n_iters=30, temperature=model.input_to_leaf.temperature).numpy()

    vars = ["A", "B", "C", "D"]

    for i, row in enumerate(perm_matrix):
        top_idx = np.argmax(row)
        important_vars = [(vars[j], row[j]) for j in range(len(vars)) if row[j] >= 0.2]
        print(f"Leaf {i+1} receives input from:")
        for var, weight in important_vars:
            print(f"  - {var}: {weight:.3f}")
        print(f"  → Most likely input: {vars[top_idx]}\n")

def extract_hard_permutation(model):
    if not hasattr(model.input_to_leaf, "logits"):
        print("🔒 Permutation already frozen. Returning current hard mapping.")
        # Reverse engineer hard mapping from P_hard buffer
        P = model.input_to_leaf.P_hard.cpu().numpy()
        return torch.tensor(np.argmax(P, axis=1))
    
    with torch.no_grad():
        perm_matrix = sinkhorn(model.input_to_leaf.logits, n_iters=50, temperature=model.input_to_leaf.temperature)
        return perm_matrix.argmax(dim=1)


# 🔥 Run Training & Visualization
if __name__ == "__main__":
    model, loss = train_and_select_best_model(weight_mode="fixed", weight_value=1.0)
    model.print_tree_structure()
    print(f"Final loss: {loss}")
    print_estimated_expression(model)

    # 🔍 Extract hard input-leaf mapping (works even if already frozen)
    hard_mapping = extract_hard_permutation(model)
    input_names = ["A", "B", "C", "D"]
    print("\n🧠 Hard Input-to-Leaf Assignment (based on argmax):")
    for leaf_idx, input_idx in enumerate(hard_mapping):
        print(f"Leaf {leaf_idx + 1} ← Input {input_idx + 1} ({input_names[input_idx]})")
