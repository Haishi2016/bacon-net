import torch
import torch.nn as nn
import torch.optim as optim
import itertools
import numpy as np

# 🔥 Generalized GCD Operator (Fixed NaN Issues)
def generalized_gcd(w1, w2, lambd):
    lambd = torch.sigmoid(lambd)  
    epsilon = 0 #1e-6  

    w1_safe = torch.abs(w1) + epsilon
    w2_safe = torch.abs(w2) + epsilon

    # return (w1_safe ** lambd) * (w2_safe ** (1 - lambd)) + (1 - lambd) * torch.max(w1_safe, w2_safe)
    return torch.min(w1_safe, w2_safe) + (1 - lambd) * torch.max(w1_safe, w2_safe)

# 🔥 Learnable Permutation Layer
class PermutationLayer(nn.Module):
    def __init__(self, num_inputs, temp=0.1):
        super(PermutationLayer, self).__init__()
        self.num_inputs = num_inputs
        self.temp = temp  
        self.perm_weights = nn.Parameter(torch.randn(num_inputs, num_inputs) * 0.1)

    def forward(self, x):
        perm_matrix = torch.softmax(self.perm_weights / self.temp, dim=-1)  
        x_permuted = torch.matmul(perm_matrix, x.T).T  
        return x_permuted, perm_matrix  

class HardPermutationLayer(nn.Module):
    def __init__(self, num_inputs):
        super(HardPermutationLayer, self).__init__()
        self.num_inputs = num_inputs
        self.perm_indices = nn.Parameter(torch.randperm(num_inputs).float(), requires_grad=True)

    def forward(self, x):
        """
        Selects a hard permutation by rounding the learned indices.
        """
        perm_indices = torch.argsort(self.perm_indices)  # Get discrete permutation order
        x_permuted = x[:, perm_indices]  # Apply permutation
        return x_permuted, perm_indices

class GumbelPermutationLayer(nn.Module):
    def __init__(self, num_inputs, temp=0.5):
        super(GumbelPermutationLayer, self).__init__()
        self.num_inputs = num_inputs
        self.temp = temp  
        self.perm_logits = nn.Parameter(torch.randn(num_inputs, num_inputs) * 0.1)

    def sample_gumbel(self, shape, eps=1e-10):
        U = torch.rand(shape)
        return -torch.log(-torch.log(U + eps) + eps)

    def forward(self, x):
        """
        Applies a learnable permutation using Gumbel-Softmax.
        """
        gumbel_noise = self.sample_gumbel(self.perm_logits.shape)
        perm_matrix = torch.softmax((self.perm_logits + gumbel_noise) / self.temp, dim=-1)  
        
        x_permuted = torch.matmul(perm_matrix, x.T).T  
        return x_permuted, perm_matrix

class HardGumbelPermutationLayer(nn.Module):
    def __init__(self, num_inputs, temp=1.0):
        super(HardGumbelPermutationLayer, self).__init__()
        self.num_inputs = num_inputs
        self.temp = temp  
        self.perm_logits = nn.Parameter(torch.randn(num_inputs, num_inputs) * 0.1)

    def sample_gumbel(self, shape, eps=1e-10):
        U = torch.rand(shape, device=self.perm_logits.device)
        return -torch.log(-torch.log(U + eps) + eps)

    def forward(self, x):
        """
        Selects a hard permutation using Straight-Through Gumbel-Softmax.
        """
        gumbel_noise = self.sample_gumbel(self.perm_logits.shape)
        soft_perm_matrix = torch.softmax((self.perm_logits + gumbel_noise) / self.temp, dim=-1)  

        # 🔹 Force a Hard Permutation
        hard_perm_matrix = torch.zeros_like(soft_perm_matrix)
        hard_perm_matrix.scatter_(1, torch.argmax(soft_perm_matrix, dim=-1, keepdim=True), 1.0)

        # 🔹 Straight-Through Estimation
        perm_matrix = soft_perm_matrix + (hard_perm_matrix - soft_perm_matrix).detach()
        
        x_permuted = torch.matmul(perm_matrix, x.T).T  
        return x_permuted, perm_matrix


class BinaryTreeLogicNet(nn.Module):
    def __init__(self, input_size, repeat_factor_per_var=2, weight_mode="trainable", weight_value=1.0, weight_range=(0.5, 2.0), weight_choices=None):
        super(BinaryTreeLogicNet, self).__init__()
        self.original_input_size = input_size
        self.expanded_input_size = input_size * repeat_factor_per_var  
        self.weight_mode = weight_mode
        self.weight_value = weight_value
        self.weight_range = weight_range
        self.weight_choices = torch.tensor(weight_choices, dtype=torch.float32) if weight_choices else None

        # 🔹 Hard One-Shot Permutation Layer
        self.permutation_layer = HardGumbelPermutationLayer(self.expanded_input_size)

        # Weights and Biases
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
        self.num_layers = self.expanded_input_size - 1  

        for _ in range(self.num_layers):
            if weight_mode == "fixed":
                self.weights.append(nn.Parameter(torch.tensor([weight_value, weight_value], dtype=torch.float32), requires_grad=False))
            elif weight_mode == "range":
                self.weights.append(nn.Parameter(torch.rand(2) * (weight_range[1] - weight_range[0]) + weight_range[0]))
            elif weight_mode == "discrete":
                self.weights.append(nn.Parameter(torch.choice(self.weight_choices, (2,)), requires_grad=True))
            else:  # "trainable"
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
        x_permuted, perm_indices = self.permutation_layer(x)
        node_outputs = list(x_permuted.T)  
        layer_outputs = []  

        for i in range(self.num_layers):
            w = self.weights[i]
            bias = self.biases[i]
            node_outputs.append(generalized_gcd(w[0] * node_outputs.pop(0), w[1] * node_outputs.pop(0), bias))
            layer_outputs.append(node_outputs[-1])

        final_output = torch.sigmoid(self.fc_out(layer_outputs[-1].unsqueeze(1)))

        return (final_output, layer_outputs, perm_indices) if return_all_layers else final_output


# 🔥 Generate Training Data
def generate_data(num_inputs, repeat_factor_per_var=2, repeat_factor=100):
    data = []
    labels = []
    base_cases = list(itertools.product([0, 1], repeat=num_inputs))  

    for _ in range(repeat_factor):
        for values in base_cases:
            A, B = values  
            y = A or B 
            expanded_values = list(values) * repeat_factor_per_var  
            data.append(expanded_values)
            labels.append([y])

    return torch.tensor(data, dtype=torch.float32), torch.tensor(labels, dtype=torch.float32)

# 🔥 Train Model with Integrated Permutation & Weight Modes
def train_model(model, X_train, Y_train, epochs=5000, early_stopping_threshold=0.0001, min_layers_required=2):
    # optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    criterion = nn.BCELoss()

    prev_layer_losses = [float("inf")] * model.num_layers
    best_layer = None  

    for epoch in range(epochs):
        optimizer.zero_grad()
        final_output, layer_outputs, perm_matrix = model(X_train, return_all_layers=True)

        layer_losses = [criterion(torch.sigmoid(out).unsqueeze(1), Y_train).item() for out in layer_outputs]

        for layer_idx, loss in enumerate(layer_losses):
            loss_reduction = prev_layer_losses[layer_idx] - loss
            if loss_reduction > early_stopping_threshold and best_layer is None:
                best_layer = layer_idx

        prev_layer_losses = layer_losses  
        
        final_loss = criterion(torch.sigmoid(final_output), Y_train)
        final_loss.backward()
        optimizer.step()

        if (epoch + 1) % 1000 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {final_loss.item():.4f}, Best Layer: {best_layer}")

        if best_layer is not None and best_layer >= min_layers_required:
            print(f"🛑 Stopping early at layer {best_layer}")
            return best_layer, perm_matrix  

    return model.num_layers - 1, perm_matrix  

# 🔥 Print Estimated Logical Expression
def print_estimated_expression(model, learned_permutation, best_layer):
    with torch.no_grad():
        perm_matrix = learned_permutation.detach().numpy()
        weights = [w.numpy() for w in model.weights[:max(2, best_layer+1)]]
        biases = [torch.tensor(b.numpy(), dtype=torch.float32) for b in model.biases[:max(2, best_layer+1)]]

    num_inputs = len(perm_matrix)
    vars = [f"X{i+1}" for i in range(num_inputs)]

    perm_indices = np.argmax(perm_matrix, axis=1)
    perm_indices = np.clip(perm_indices, 0, len(vars) - 1)
    perm_vars = [vars[i] for i in perm_indices]

    expression_stack = []

    for i, (w, b) in enumerate(zip(weights, biases)):
        gate_type = "AND" if torch.sigmoid(b).item() > 0.5 else "OR"

        if i == 0:
            expression_stack.append(perm_vars[i])  
        else:
            prev_expr = expression_stack.pop()
            new_expr = f"({prev_expr} {gate_type} {perm_vars[i]})"

            if new_expr not in expression_stack:
                expression_stack.append(new_expr)

    print(f"📢 Estimated Logical Expression: {expression_stack[0]}")

# 🔥 Run Training
if __name__ == "__main__":
    num_inputs = 2  
    repeat_factor_per_var = 1  
    X_train, Y_train = generate_data(num_inputs, repeat_factor_per_var=repeat_factor_per_var, repeat_factor=100)

    model = BinaryTreeLogicNet(num_inputs, repeat_factor_per_var, weight_mode="fixed", weight_value=1.0)
    best_layer, learned_permutation = train_model(model, X_train, Y_train, epochs=200000)

    if model is not None:
        print_estimated_expression(model, learned_permutation, best_layer)
