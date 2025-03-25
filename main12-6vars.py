import torch
import torch.nn as nn
import torch.optim as optim
import itertools
import numpy as np
from itertools import product
import copy
import random 
from heapq import nlargest
import itertools

NUM_INPUT_VARS = 12  # 👈 Change this value to control number of inputs
NOISE_DECREASE_RATIO = 0.95 # Decrease noise if loss decreases
NOISE_INCREASE_RATIO = 1.05 # Increase noise if loss increases or plateaus
NOISE_MIN = 0.0 # Minimum noise scale for Gumbel noise. Set to 0 for no noise.
NOISE_MAX = 2.0 # Maximum noise scale for Gumbel noise. Set to 0 for no noise.
PERMUTATION_MAX = 1000 # Maximum number of permutations to sample from soft alignment. Set to 0 for all. Not recommend for vars >= 12.
FREEZE_LOSS_THRESHOLD=0.01  # 0.05 works well for vars below 12. A higher value means the model is more eager to try out possible permutations to freeze.
FROZEN_SELECTION_THRESHOLD=0.001 # threshold for selecting the frozen model.

seen_permutations = set()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

def sample_gumbel(shape, device=None, eps=1e-20):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    U = torch.rand(shape, device=device)
    return -torch.log(-torch.log(U + eps) + eps)

def gumbel_sinkhorn(log_alpha, temperature=1.0, n_iters=20, noise_scale=1.0):
    noise = sample_gumbel(log_alpha.shape, device=log_alpha.device) * noise_scale
    perturbed = log_alpha + noise
    return sinkhorn(perturbed, temperature=temperature, n_iters=n_iters)


def evaluate_permutation(model_template, perm, X, Y, weight_mode, weight_value, weight_range, weight_choices):
    # Create a fresh model with the given permutation frozen
    model = BinaryTreeLogicNet(X.shape[1], weight_mode, weight_value, weight_range, weight_choices).to(device)
    X, Y = X.to(device), Y.to(device)
    model.input_to_leaf = FrozenInputToLeaf(torch.tensor(perm), X.shape[1])
    model.eval()
    with torch.no_grad():
        outputs = model(X)
        if torch.isnan(outputs).any() or (outputs < 0).any() or (outputs > 1).any():
            return float("inf")  # Disqualify
        loss = nn.BCELoss()(outputs, Y)
    return loss.item()

def sample_topk_permutations(num_inputs, k, model_template, X, Y, weight_mode, weight_value, weight_range, weight_choices):
    with torch.no_grad():
        if hasattr(model_template.input_to_leaf, "logits"):
            P = sinkhorn(model_template.input_to_leaf.logits, temperature=model_template.input_to_leaf.temperature)
        else:
            raise ValueError("Model does not have learnable logits for Sinkhorn.")

    topk_candidates = set()
    max_topk = k if k > 0 else 100  # Allow fallback for all if k=0

    def greedy_match(P, num_choices_per_leaf):
        num_choices_per_leaf = min(5, num_inputs)
        top_inputs = [torch.topk(P[i], k=min(num_choices_per_leaf, P.shape[1])).indices.tolist() for i in range(num_inputs)]
        for combo in itertools.product(*top_inputs):
            if len(set(combo)) == num_inputs:
                topk_candidates.add(tuple(combo))
            if len(topk_candidates) >= max_topk:
                break

    greedy_match(P, num_choices_per_leaf=5)

    if not topk_candidates:
        argmax_perm = tuple(P.argmax(dim=1).tolist())
        topk_candidates.add(argmax_perm)

    print(f"🧠 Evaluating {len(topk_candidates)} promising permutations from soft alignment...")

    losses = []
    for perm in topk_candidates:
        loss = evaluate_permutation(model_template, perm, X, Y, weight_mode, weight_value, weight_range, weight_choices)
        losses.append((perm, loss))

    topk_by_loss = sorted(losses, key=lambda x: x[1])
    return [torch.tensor(p[0], device=device) for p in topk_by_loss]



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
        self.register_buffer("P_hard", torch.zeros(len(hard_assignment), num_inputs).to(device))
        for leaf_idx, input_idx in enumerate(hard_assignment):
            self.P_hard[leaf_idx, input_idx] = 1.0

    def forward(self, x):
        return torch.matmul(x, self.P_hard.t())  # Ensure correct device

class InputToLeafSinkhorn(nn.Module):
    def __init__(self, num_inputs, num_leaves, temperature=1.0, sinkhorn_iters=20, use_gumbel=True):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_leaves = num_leaves
        self.temperature = temperature
        self.sinkhorn_iters = sinkhorn_iters
        self.use_gumbel = use_gumbel
        self.gumbel_noise_scale = 1.0  # You can anneal this

        self.logits = nn.Parameter(torch.randn(num_leaves, num_inputs))

    def forward(self, x):
        if self.use_gumbel:
            P = gumbel_sinkhorn(self.logits, temperature=self.temperature, n_iters=self.sinkhorn_iters, noise_scale=self.gumbel_noise_scale)
        else:
            P = sinkhorn(self.logits, temperature=self.temperature, n_iters=self.sinkhorn_iters)
        return torch.matmul(x, P.t())

    def decrease_temperature(self, factor=0.98, noise_decay=0.98):
        self.temperature *= factor
        self.gumbel_noise_scale *= noise_decay




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
        self.input_to_leaf = InputToLeafSinkhorn(input_size, self.num_leaves, use_gumbel=True)

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
            w = self.weights[i].detach().cpu().numpy()

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

        print("\n🌲 Binary Tree Structure:\n")
        root = f"Node{self.num_layers}"
        print(format_tree(root))


def generate_data(num_vars=5, repeat_factor=100):
    print("🧠 Generating data...")
    assert num_vars >= 2, "Need at least 2 variables for expression."
    data = []
    labels = []
    base_cases = list(itertools.product([0, 1], repeat=num_vars))

    # Step 1: generate stable ops per variable link
    ops = [random.choice(["and", "or"]) for _ in range(num_vars - 1)]

    # Step 2: generate variable names and build expression strings
    var_names = [chr(ord('A') + i) for i in range(num_vars)]
    symbolic_expr = var_names[0]
    eval_expr = "x[0]"
    for i in range(1, num_vars):
        op = ops[i - 1]
        symbolic_expr = f"({symbolic_expr} {op} {var_names[i]})"
        eval_expr = f"({eval_expr} {op} x[{i}])"

    # Step 3: evaluate the expression across the truth table
    for _ in range(repeat_factor):
        for x in base_cases:
            y = int(eval(eval_expr))
            data.append(list(x))
            labels.append([y])

    return (
        torch.tensor(data, dtype=torch.float32),
        torch.tensor(labels, dtype=torch.float32),
        {
            "expression_text": symbolic_expr,
            "eval_expr": eval_expr,
            "ops": ops,
            "num_vars": num_vars,
            "var_names": var_names
        }
    )

def train_and_select_best_model(weight_mode="trainable", weight_value=1.0,
                                 weight_range=(0.5, 2.0), weight_choices=None,
                                 freeze_loss_threshold=0.005, freeze_patience=100, max_retries=10):
    X_train, Y_train,  expr_info = generate_data(NUM_INPUT_VARS, repeat_factor=100)
    X_train, Y_train = X_train.to(device), Y_train.to(device)

    print("🧠 Expression used for training:", expr_info["expression_text"])

    def create_model():
        model =  BinaryTreeLogicNet(X_train.shape[1], weight_mode, weight_value, weight_range, weight_choices)
        return model.to(device)
    

    for attempt in range(max_retries):
        print(f"\n🔥 Attempt {attempt + 1}/{max_retries}")
        model = create_model()
        optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-5)
        criterion = nn.BCELoss()
        frozen = False
        frozen_abandoned = False
        patience_counter = 0
        best_frozen_loss = float('inf')
        loss_history = []
        noise_increase = NOISE_INCREASE_RATIO
        noise_decrease = NOISE_DECREASE_RATIO
        min_noise = NOISE_MIN
        max_noise = NOISE_MAX
        for epoch in range(12000):
            if hasattr(model.input_to_leaf, "temperature") and (epoch + 1) % 1000 == 0:
                model.input_to_leaf.temperature *= 0.8

            optimizer.zero_grad()
            outputs = model(X_train)
            
          
            if torch.isnan(outputs).any() or (outputs < 0).any() or (outputs > 1).any():
                print("⚠️ Instability detected. Restarting.")
                break

            loss = criterion(outputs, Y_train)

            loss.backward()
            optimizer.step()


            loss_history.append(loss.item())
            if len(loss_history) > 5:
                loss_history.pop(0)
                diffs = np.diff(loss_history)
                
                if all(d < 0 for d in diffs):  # strictly decreasing
                    model.input_to_leaf.gumbel_noise_scale = max(model.input_to_leaf.gumbel_noise_scale * noise_decrease, min_noise)
                    # print(f"🔻 Stable improvement. Noise scale: {model.input_to_leaf.gumbel_noise_scale:.4f}")
                elif all(abs(d) < 1e-4 for d in diffs):  # plateau
                    model.input_to_leaf.gumbel_noise_scale = min(model.input_to_leaf.gumbel_noise_scale * noise_increase, max_noise)
                    # print(f"🟰 Plateau. Noise scale: {model.input_to_leaf.gumbel_noise_scale:.4f}")
                elif any(d > 0 for d in diffs):  # getting worse
                    model.input_to_leaf.gumbel_noise_scale = min(model.input_to_leaf.gumbel_noise_scale * noise_increase, max_noise)
                    # print(f"🔺 Loss increased. Noise scale: {model.input_to_leaf.gumbel_noise_scale:.4f}")

            if not frozen and loss.item() < freeze_loss_threshold:
                print(f"🧊 Low loss at epoch {epoch}, sampling top-k permutations...")
                candidates = sample_topk_permutations(
                    model.original_input_size,
                    k=PERMUTATION_MAX,  # or 0 for all
                    model_template=model,
                    X=X_train,
                    Y=Y_train,
                    weight_mode=weight_mode,
                    weight_value=weight_value,
                    weight_range=weight_range,
                    weight_choices=weight_choices
                )


                best_loss = float('inf')
                best_model = None
                best_perm = None

                for perm in candidates:
                    perm_tensor = perm.clone().detach()
                    temp_model = copy.deepcopy(model).to(device)
                    temp_model.input_to_leaf = FrozenInputToLeaf(perm_tensor, temp_model.original_input_size)
                    temp_loss = criterion(temp_model(X_train), Y_train)
                    print(f"   🔍 Perm {perm} → Loss: {temp_loss.item():.4f}")

                    if temp_loss < best_loss:
                        best_loss = temp_loss
                        best_model = temp_model
                        best_perm = perm

                if best_model is not None and best_loss < freeze_loss_threshold:
                    print(f"✅ Freezing best permutation: {best_perm} with loss {best_loss:.4f}")
                    return best_model, best_loss, expr_info
                else:
                    print("🚫 No good permutation found in top-k. Restarting.")
                    break


            if frozen:
                if loss.item() < best_frozen_loss:
                    best_frozen_loss = loss.item()
                    patience_counter = 0
                else:
                    patience_counter += 1

                if best_frozen_loss < FROZEN_SELECTION_THRESHOLD:
                        print(f"✅ Accepted: frozen model converged with loss {best_frozen_loss:.4f}")
                        return model, best_frozen_loss, expr_info

                if patience_counter > freeze_patience:
                    print(f"🚨 Abandoning frozen model due to stagnation (loss: {loss.item():.4f})")
                    frozen_abandoned = True
                    break

            if epoch % 200 == 0:
                print(f"   Epoch {epoch} - Temp: {model.input_to_leaf.temperature:.4f} - Noise: {model.input_to_leaf.gumbel_noise_scale:.4f} - Loss: {loss.item():.4f}")
            
        if not frozen or frozen_abandoned:
            print("🚫 Training completed without freezing or freezing is abandoned. Abandoning this direction.")
            continue
        return model, loss.item(), expr_info

    print("❌ Exhausted all attempts without finding a stable model.")
    return None, None


    return model, loss.item()

def print_estimated_expression(model, expr_info):
    if not hasattr(model.input_to_leaf, "logits"):
        print("🔒 Permutation is frozen. Skipping Sinkhorn inspection.\n")
        return

    with torch.no_grad():
        perm_matrix = sinkhorn(model.input_to_leaf.logits, n_iters=30, temperature=model.input_to_leaf.temperature).numpy()

    num_vars = expr_info["num_vars"] if expr_info else perm_matrix.shape[1]
    vars = [chr(ord("A") + i) for i in range(num_vars)]

    print("🔍 Estimated Input-to-Leaf Assignment (Soft):\n")
    for i, row in enumerate(perm_matrix):
        top_idx = np.argmax(row)
        important_vars = [(vars[j], row[j]) for j in range(len(vars)) if row[j] >= 0.2]
        print(f"Leaf {i+1} receives input from:")
        for var, weight in important_vars:
            print(f"  - {var}: {weight:.3f}")
        print(f"  → Most likely input: {vars[top_idx]}\n")

def extract_hard_permutation(model):
    if not hasattr(model.input_to_leaf, "logits"):
        print("🔒 Permutation is already frozen. Returning current hard mapping.")
        # Reverse engineer hard mapping from P_hard buffer
        P = model.input_to_leaf.P_hard.cpu().numpy()
        return torch.tensor(np.argmax(P, axis=1))
    
    with torch.no_grad():
        perm_matrix = sinkhorn(model.input_to_leaf.logits, n_iters=50, temperature=model.input_to_leaf.temperature)
        return perm_matrix.argmax(dim=1).to("cpu")


# 🔥 Run Training & Visualization
if __name__ == "__main__":

    print(f"\n           🔢 Number of variables: {NUM_INPUT_VARS}")
    print(f"         📉 Noise decrease ration: {NOISE_DECREASE_RATIO}")
    print(f"         📈 Noise increase ration: {NOISE_INCREASE_RATIO}")
    print(f"                 🔇 Minimum noise: {NOISE_MIN}")
    print(f"                 🔊 Maximum noise: {NOISE_MAX}")
    print(f"🔢 Maximum permutation candidates: {PERMUTATION_MAX}")
    print(f"         🧊 Freeze loss threshold: {FREEZE_LOSS_THRESHOLD}")
    print(f"     🎯 Model selection threshold: {FROZEN_SELECTION_THRESHOLD}")

    model, loss, expr_info = train_and_select_best_model(weight_mode="fixed", weight_value=1.0, max_retries=100, freeze_loss_threshold=FREEZE_LOSS_THRESHOLD)

    if model is None:
        print("❌ No valid model found after retries.")
    else:
        model.print_tree_structure()
        print(f"\n🧠 Final loss: {loss}\n")
        print_estimated_expression(model, expr_info)

        hard_mapping = extract_hard_permutation(model)
        var_names = [chr(ord('A') + i) for i in range(expr_info["num_vars"])]

        print("\n🧠 Hard Input-to-Leaf Assignment (based on argmax):\n")
        for leaf_idx, input_idx in enumerate(hard_mapping):
            print(f"   🍃 Leaf {leaf_idx + 1} ← Input {input_idx + 1} ({var_names[input_idx]})")

        print("\n📌 Ground truth expression structure:")
        print(expr_info["expression_text"])

