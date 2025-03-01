import torch
import torch.nn as nn
import torch.optim as optim
import itertools
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# 🔥 Generalized GCD Operator (Fixed NaN Issues)
def generalized_gcd(w1, w2, lambd):
    lambd = torch.sigmoid(lambd)  # Ensure lambda is between 0 and 1
    epsilon = 1e-6  # Small value to prevent zero exponentiation errors

    # Ensure non-negative values for exponentiation
    w1_safe = torch.abs(w1) + epsilon
    w2_safe = torch.abs(w2) + epsilon

    return (w1_safe ** lambd) * (w2_safe ** (1 - lambd)) + (1 - lambd) * torch.max(w1_safe, w2_safe)

# 🔥 Binary Tree Logic Network With Configurable Weights
class BinaryTreeLogicNet(nn.Module):
    def __init__(self, weight_mode="trainable", weight_value=1.0, weight_range=(0.5, 2.0), weight_choices=None):
        """
        weight_mode: How weights are controlled:
            - "fixed" : Weights are fixed to a given value (weight_value)
            - "range" : Weights are trainable but clamped within a range (weight_range)
            - "discrete" : Weights are constrained to a set of values (weight_choices)
            - "trainable" : Fully flexible, unconstrained training
        """
        super(BinaryTreeLogicNet, self).__init__()
        self.weight_mode = weight_mode
        self.weight_value = weight_value
        self.weight_range = weight_range
        self.weight_choices = torch.tensor(weight_choices, dtype=torch.float32) if weight_choices else None

        # Initialize Weights
        if weight_mode == "fixed":
            self.w1 = torch.tensor([weight_value, weight_value], dtype=torch.float32, requires_grad=False)
            self.w2 = torch.tensor([weight_value, weight_value], dtype=torch.float32, requires_grad=False)
        else:
            self.w1 = nn.Parameter(torch.randn(2) * 0.1)
            self.w2 = nn.Parameter(torch.randn(2) * 0.1)

        # Learnable GCD Bias
        self.gcd_bias1 = nn.Parameter(torch.rand(1) * 0.1)
        self.gcd_bias2 = nn.Parameter(torch.rand(1) * 0.1)
        self.fc_out = nn.Linear(1, 1)

        # Apply better weight initialization
        self.apply(self.initialize_weights)

    def initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Implements binary tree logic using the GCD operator with configurable weights.
        """
        if self.weight_mode == "fixed":
            w1, w2 = self.w1, self.w2  
        elif self.weight_mode == "range":
            w1 = torch.clamp(self.w1, self.weight_range[0], self.weight_range[1])
            w2 = torch.clamp(self.w2, self.weight_range[0], self.weight_range[1])
        elif self.weight_mode == "discrete":
            with torch.no_grad():
                self.w1.copy_(self.weight_choices[torch.argmin(torch.abs(self.weight_choices.unsqueeze(0) - self.w1.unsqueeze(1)), dim=1)] )
                self.w2.copy_(self.weight_choices[torch.argmin(torch.abs(self.weight_choices.unsqueeze(0) - self.w2.unsqueeze(1)), dim=1)] )
            w1, w2 = self.w1, self.w2
        else:  # "trainable"
            w1, w2 = self.w1, self.w2  

        # First hidden layer: Compute GCD for first two inputs
        h1 = generalized_gcd(w1[0] * x[:, 0], w1[1] * x[:, 1], self.gcd_bias1)
        if torch.isnan(h1).any():
            print("❌ NaN detected in h1!")

        # Second hidden layer: Compute GCD for H1 and the third input
        h2 = generalized_gcd(w2[0] * h1, w2[1] * x[:, 2], self.gcd_bias2)
        if torch.isnan(h2).any():
            print("❌ NaN detected in h2!")

        # Output (Use Sigmoid for Stability)
        out = torch.sigmoid(self.fc_out(h2.unsqueeze(1)))  
        if torch.isnan(out).any():
            print("❌ NaN detected in output!")

        return out

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

# 🔥 Train Model with Debugging
def train_model(model, X_train, Y_train, epochs=5000):
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    criterion = nn.BCELoss()  # Use BCELoss instead of BCEWithLogitsLoss

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, Y_train)

        if torch.isnan(loss):
            print(f"⚠️ NaN detected in loss at epoch {epoch}! Skipping update.")
            continue

        loss.backward()

        # Debugging: Print gradient flow
        # for name, param in model.named_parameters():
        #    if param.grad is not None:
        #        print(f"Gradient for {name}: {param.grad.norm().item()}")

        optimizer.step()

        if (epoch + 1) % 1000 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    return model

# 🔥 Train and Select Best Model
def train_and_select_best_permutation(weight_mode="trainable", weight_value=1.0, weight_range=(0.5, 2.0), weight_choices=None):
    best_perm, best_loss, best_model = None, float('inf'), None
    X_train_base, Y_train = generate_data(repeat_factor=100)

    for perm in itertools.permutations(range(3)):
        X_train = X_train_base[:, list(perm)]
        temp_model = BinaryTreeLogicNet(weight_mode, weight_value, weight_range, weight_choices)

        print(f"Training model with permutation: {perm} (Weight Mode: {weight_mode})")
        temp_model = train_model(temp_model, X_train, Y_train, epochs=5000)

        with torch.no_grad():
            outputs = temp_model(X_train)
            loss = nn.BCELoss()(outputs, Y_train)

        if loss.item() < best_loss:
            best_loss, best_perm, best_model = loss.item(), perm, temp_model

    return best_model, best_perm, best_loss

# 🔥 Check Model Initialization
def debug_model_initialization(model, X_train):
    with torch.no_grad():
        sample_outputs = model(X_train[:10])
        print("Initial outputs:", sample_outputs.squeeze().tolist())

# def classify_gate(weights, bias):
#     """
#     Classifies a node as AND or OR based on learned bias.
#     """
#     # Ensure weights and bias are Tensors
#     if isinstance(weights, np.ndarray):
#         weights = torch.tensor(weights, dtype=torch.float32)
#     if isinstance(bias, np.ndarray):
#         bias = torch.tensor(bias, dtype=torch.float32)

#     lambd = torch.sigmoid(bias).item()  # Convert bias to valid lambda (0 to 1)
#     gcd_value = generalized_gcd(weights[0], weights[1], bias)  # Compute gate type

#     return "AND" if gcd_value < np.sqrt(abs(weights[0] * weights[1])) else "OR"

def classify_gate(weights, bias, threshold=0.5):
    """
    Classifies a node as AND or OR based on learned bias.
    Uses a threshold on sigmoid(mean(bias)) to determine the logic type.
    """
    if isinstance(weights, np.ndarray):
        weights = torch.tensor(weights, dtype=torch.float32)
    if isinstance(bias, np.ndarray):
        bias = torch.tensor(bias, dtype=torch.float32)

    # Ensure bias is a scalar by taking the mean if it's a tensor with multiple elements
    if bias.numel() > 1:
        bias = bias.mean()

    lambd = torch.sigmoid(bias).item()  # Convert bias to valid lambda (0 to 1)

    return "AND" if lambd > threshold else "OR"



def print_estimated_expression(model, best_perm):
    if model is None:
        raise ValueError("Cannot extract logic: No trained model was provided!")

    # Get trained weights and biases
    with torch.no_grad():
        w1 = model.w1.numpy()
        w2 = model.w2.numpy()
        gcd_bias1 = torch.tensor(model.gcd_bias1.numpy(), dtype=torch.float32)
        gcd_bias2 = torch.tensor(model.gcd_bias2.numpy(), dtype=torch.float32)

    # Map permutation indices to variable names
    vars = ["A", "B", "C"]
    A, B, C = vars[best_perm[0]], vars[best_perm[1]], vars[best_perm[2]]

    # Determine gate types using both weights and biases
    h1_gate = classify_gate(w1, gcd_bias1)  # First hidden layer gate
    h2_gate = classify_gate(w2, gcd_bias2)  # Second hidden layer gate

    # Build logical expression
    if h1_gate == "AND":
        h1_expr = f"({A} AND {B})"
    else:
        h1_expr = f"({A} OR {B})"

    if h2_gate == "AND":
        final_expr = f"({h1_expr} AND {C})"
    else:
        final_expr = f"({h1_expr} OR {C})"

    print(f"📢 Estimated Logical Expression: {final_expr}")


# 🔥 Run Training & Visualization
if __name__ == "__main__":
    # model, best_perm, best_loss = train_and_select_best_permutation(
    #     weight_mode="trainable"
    # )
    model, best_perm, best_loss = train_and_select_best_permutation(
        weight_mode="fixed",
        weight_value=1.0,
    )
    print(f"Best permutation found: {best_perm}, loss {best_loss}")
    print_estimated_expression(model, best_perm)
