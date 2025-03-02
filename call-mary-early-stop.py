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
    def __init__(self, input_size, repeat_factor_per_var=2, weight_mode="trainable", weight_value=1.0, weight_range=(0.5, 2.0), weight_choices=None):
        super(BinaryTreeLogicNet, self).__init__()
        self.original_input_size = input_size
        self.expanded_input_size = input_size * repeat_factor_per_var  
        self.weight_mode = weight_mode
        self.weight_value = weight_value
        self.weight_range = weight_range
        self.weight_choices = torch.tensor(weight_choices, dtype=torch.float32) if weight_choices else None

        # Initialize Weights and Biases for Binary Tree
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()

        for _ in range(self.expanded_input_size - 1):
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
        node_outputs = list(x.T)  

        # Apply hierarchical binary composition
        for i in range(len(self.weights)):
            w = self.weights[i]
            bias = self.biases[i]
            node_outputs.append(generalized_gcd(w[0] * node_outputs.pop(0), w[1] * node_outputs.pop(0), bias))

        # Final output
        out = torch.sigmoid(self.fc_out(node_outputs[0].unsqueeze(1)))
        return out

# 🔥 Generate Training Data
def generate_data(num_inputs, repeat_factor_per_var=2, repeat_factor=100):
    """
    Generates training data for the expression: ((!A AND B) AND C) OR (C AND D)
    """
    data = []
    labels = []
    base_cases = list(itertools.product([0, 1], repeat=num_inputs))  

    for _ in range(repeat_factor):
        for values in base_cases:
            A, B, C = values  
            # y = ((not A and B) and C) or (C and A)
            y = A and B or C  
            expanded_values = list(values) * repeat_factor_per_var  
            data.append(expanded_values)
            labels.append([y])

    return torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

# 🔥 Train Model with Early Stopping
def train_model(model, X_train, Y_train, epochs=5000, early_stopping_threshold=0.000001):
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    criterion = nn.BCELoss()

    prev_loss = float("inf")
    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, Y_train)

        if torch.isnan(loss):
            print(f"⚠️ NaN detected in loss at epoch {epoch}! Skipping update.")
            continue

        loss.backward()
        optimizer.step()

        loss_reduction = prev_loss - loss.item()
        prev_loss = loss.item()

        if loss_reduction < early_stopping_threshold:
            print(f"🛑 Early stopping at epoch {epoch+1}, loss reduction: {loss_reduction:.6f}")
            return None

        if (epoch + 1) % 1000 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    return model

# 🔥 Train and Select Best Model
def train_and_select_best_permutation(num_inputs, repeat_factor_per_var=2, weight_mode="trainable", weight_value=1.0, weight_range=(0.5, 2.0), weight_choices=None):
    best_perm, best_loss, best_model = None, float('inf'), None
    X_train_base, Y_train = generate_data(num_inputs, repeat_factor_per_var=repeat_factor_per_var, repeat_factor=100)

    total_permutations = 0
    abandoned_permutations = 0

    for perm in itertools.permutations(range(num_inputs * repeat_factor_per_var)):
        total_permutations += 1
        X_train = X_train_base[:, list(perm)]
        temp_model = BinaryTreeLogicNet(num_inputs, repeat_factor_per_var, weight_mode, weight_value, weight_range, weight_choices)

        print(f"Training model with permutation: {perm} (Weight Mode: {weight_mode})")
        trained_model = train_model(temp_model, X_train, Y_train, epochs=5000)

        if trained_model is None:
            abandoned_permutations += 1
            continue

        with torch.no_grad():
            outputs = trained_model(X_train)
            loss = nn.BCELoss()(outputs, Y_train)

        if loss.item() < best_loss:
            best_loss, best_perm, best_model = loss.item(), perm, trained_model

    print(f"🔢 Total permutations tested: {total_permutations}")
    print(f"❌ Permutations abandoned: {abandoned_permutations}")

    return best_model, best_perm, best_loss

# 🔥 Inference Function
def run_inference(model, input_values, repeat_factor_per_var=2):
    """
    Runs inference on a trained model with a given input.
    - input_values: List of binary values (0s and 1s) corresponding to the original input variables.
    - repeat_factor_per_var: Number of times each input variable is repeated.
    """
    if model is None:
        print("❌ No trained model available for inference.")
        return

    # Expand input based on repeat factor
    expanded_input = input_values * repeat_factor_per_var
    input_tensor = torch.tensor([expanded_input], dtype=torch.float32)

    with torch.no_grad():
        prediction = model(input_tensor).item()

    print(f"🔮 Prediction for {input_values} → {prediction:.4f}")

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

# 🔥 Run Training & Inference
if __name__ == "__main__":
    num_inputs = 3  
    repeat_factor_per_var = 1  
    # model, best_perm, best_loss = train_and_select_best_permutation(
    #     num_inputs=num_inputs,
    #     repeat_factor_per_var=repeat_factor_per_var,
    #     weight_mode="trainable"
    # )

    model, best_perm, best_loss = train_and_select_best_permutation(
         num_inputs=num_inputs,
         repeat_factor_per_var=repeat_factor_per_var,
         weight_mode="fixed",
        weight_value=1.0,
    )

    if model is not None:
        print(f"✅ Best permutation found: {best_perm}, loss {best_loss}")
        print_estimated_expression(model, best_perm)
        # Test inference
        test_input = [0.2, 0.1, 0.5]  
        run_inference(model, test_input, repeat_factor_per_var)
    else:
        print("❌ No valid model found.")
