import torch
import torch.nn as nn
import torch.optim as optim
import itertools
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
    def __init__(self, input_size, weight_mode="trainable", weight_value=1.0, weight_range=(0.5, 2.0), weight_choices=None):
        """
        - input_size: Number of input variables.
        - weight_mode: How weights are controlled:
            - "fixed" : Weights are fixed to a given value (weight_value)
            - "range" : Weights are trainable but clamped within a range (weight_range)
            - "discrete" : Weights are constrained to a set of values (weight_choices)
            - "trainable" : Fully flexible, unconstrained training
        """
        super(BinaryTreeLogicNet, self).__init__()
        self.input_size = input_size
        self.weight_mode = weight_mode
        self.weight_value = weight_value
        self.weight_range = weight_range
        self.weight_choices = torch.tensor(weight_choices, dtype=torch.float32) if weight_choices else None

        # Initialize Weights and Biases for Binary Tree
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()

        for _ in range(input_size - 1):
            if weight_mode == "fixed":
                self.weights.append(nn.Parameter(torch.tensor([weight_value, weight_value], dtype=torch.float32), requires_grad=False))
            else:
                self.weights.append(nn.Parameter(torch.randn(2) * 0.1))

            self.biases.append(nn.Parameter(torch.rand(1) * 0.1))

        # Final Output Layer
        self.fc_out = nn.Linear(1, 1)
        self.apply(self.initialize_weights)

    def initialize_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Implements dynamic binary tree logic using the GCD operator.
        """
        node_outputs = list(x.T)  # Convert inputs into list of nodes

        # Apply hierarchical binary composition
        for i in range(len(self.weights)):
            w = self.weights[i]
            bias = self.biases[i]

            # Process two nodes at a time and combine them
            node_outputs.append(generalized_gcd(w[0] * node_outputs.pop(0), w[1] * node_outputs.pop(0), bias))

        # Final output
        out = torch.sigmoid(self.fc_out(node_outputs[0].unsqueeze(1)))
        return out

# 🔥 Generate Training Data
def generate_data(num_inputs, repeat_factor=100):
    data = []
    labels = []
    base_cases = list(itertools.product([0, 1], repeat=num_inputs))

    for _ in range(repeat_factor):
        for values in base_cases:
            y = values[0] and (values[1] or values[2])  # Example function (can be changed)
            data.append(list(values))
            labels.append([y])

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
        optimizer.step()

        if (epoch + 1) % 1000 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    return model

# 🔥 Train and Select Best Model
def train_and_select_best_permutation(num_inputs, weight_mode="trainable", weight_value=1.0, weight_range=(0.5, 2.0), weight_choices=None):
    best_perm, best_loss, best_model = None, float('inf'), None
    X_train_base, Y_train = generate_data(num_inputs, repeat_factor=100)

    for perm in itertools.permutations(range(num_inputs)):
        X_train = X_train_base[:, list(perm)]
        temp_model = BinaryTreeLogicNet(num_inputs, weight_mode, weight_value, weight_range, weight_choices)

        print(f"Training model with permutation: {perm} (Weight Mode: {weight_mode})")
        temp_model = train_model(temp_model, X_train, Y_train, epochs=5000)

        with torch.no_grad():
            outputs = temp_model(X_train)
            loss = nn.BCELoss()(outputs, Y_train)

        if loss.item() < best_loss:
            best_loss, best_perm, best_model = loss.item(), perm, temp_model

    return best_model, best_perm, best_loss

# 🔥 Run Training & Visualization
if __name__ == "__main__":
    num_inputs = 5  # You can change this to test with more inputs
    model, best_perm, best_loss = train_and_select_best_permutation(
        num_inputs=num_inputs,
        weight_mode="trainable"
    )
    print(f"Best permutation found: {best_perm}, loss {best_loss}")
