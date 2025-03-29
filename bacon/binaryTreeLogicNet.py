import torch
import torch.nn as nn
from bacon.inputToLeafSinkhorn import inputToLeafSinkhorn
from bacon.frozonInputToLeaf import frozenInputToLeaf
import torch.optim as optim
import numpy as np
import itertools
import copy
import logging 
import logging 

class binaryTreeLogicNet(nn.Module):
    def __init__(self, 
                 input_size, 
                 weight_mode="fixed", 
                 weight_value=1.0, 
                 weight_range=(0.5, 2.0), 
                 weight_choices=None,
                 noise_increase=1.05,
                 noise_decrease=0.95,
                 min_noise=0.0,
                 max_noise=2.0,
                 freeze_loss_threshold=0.07,
                 permutation_max=100):
        super(binaryTreeLogicNet, self).__init__()
        self.original_input_size = input_size
        self.num_leaves = input_size  # 🔹 Each input gets its own leaf initially
        self.weight_mode = weight_mode
        self.weight_value = weight_value
        self.weight_range = weight_range
        self.weight_choices = torch.tensor(weight_choices, dtype=torch.float32) if weight_choices else None
        # 🔹 Fully Connected Input-to-Leaf Mapping
        self.input_to_leaf = inputToLeafSinkhorn(input_size, self.num_leaves, use_gumbel=True)
        self.noise_increase = noise_increase
        self.noise_decrease = noise_decrease
        self.min_noise = min_noise
        self.max_noise = max_noise
        self.is_frozen = False
        self.freeze_loss_threshold = freeze_loss_threshold
        self.permutation_max = permutation_max
        self.locked_perm = None  # For frozen models
        self.optimizer = optim.Adam(self.parameters(), lr=0.01, weight_decay=1e-5)
        # Weights and Biases
        self.num_layers = self.num_leaves - 1  # Leaf nodes feed into binary tree
        self.reinitialize(weight_mode, weight_value, weight_range, weight_choices)
    def reinitialize(self, weight_mode, weight_value, weight_range, weight_choices):
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()
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
    def save_model(self, file_name):
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'is_frozen': self.is_frozen,
            'locked_perm': self.locked_perm,
        }, file_name)
    def load_model(self, file_name):
        checkpoint = torch.load(file_name, weights_only=True)
        state_dict = checkpoint['model_state_dict']
        self.is_frozen = checkpoint.get('is_frozen', False)
        self.locked_perm = checkpoint.get('locked_perm', None)
        if self.is_frozen:
            # Frozen → use FrozenInputToLeaf
            P_hard = state_dict['input_to_leaf.P_hard']
            self.input_to_leaf = frozenInputToLeaf(
                hard_assignment=torch.argmax(P_hard, dim=1),
                num_inputs=self.original_input_size
            )
        # Now load weights
        self.load_state_dict(state_dict)

    def forward(self, x):
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
            node_outputs.append(self.generalized_gcd(w[0] * left, w[1] * right, bias))
            layer_outputs.append(node_outputs[-1])

        final_output = torch.sigmoid(self.fc_out(layer_outputs[-1].unsqueeze(1)))
        return final_output

    def generalized_gcd(self, a, b, r):
        epsilon = 1e-6  # To prevent division by zero
        r = torch.sigmoid(r) * 10 - 5  # Map sigmoid to range ~[-5, 5] for stability

        a = torch.clamp(a, min=epsilon)
        b = torch.clamp(b, min=epsilon)

        # Avoid r = 0 case with Taylor approximation
        is_near_zero = (r.abs() < 1e-2)
        safe_r = r + (~is_near_zero).float() * 1e-6

        gcd = ((a ** safe_r) + (b ** safe_r)) / 2
        gcd = gcd ** (1 / safe_r)

        # Approximate geometric mean when r ≈ 0
        geo_mean = torch.sqrt(a * b)
        return torch.where(is_near_zero, geo_mean, gcd)

    def train_model(self, x, y, epochs):
        self.reinitialize(self.weight_mode, self.weight_value, self.weight_range, self.weight_choices)
        loss_history = []
        self.is_frozen = False
        self.optimizer = optim.Adam(self.parameters(), lr=0.01, weight_decay=1e-5)
        criterion = nn.BCELoss()
        for epoch in range(epochs):
            if hasattr(self.input_to_leaf, "temperature") and (epoch + 1) % 1000 == 0:
                self.input_to_leaf.temperature *= 0.8
            self.optimizer.zero_grad()
            outputs = self(x)
            if torch.isnan(outputs).any() or (outputs < 0).any() or (outputs > 1).any():
                raise RuntimeError("Instability detected. Can't be trained further.")
            loss = criterion(outputs, y)
            loss.backward()
            self.optimizer.step()
            loss_history.append(loss.item())
            if not self.is_frozen:
                if len(loss_history) > 5:
                    loss_history.pop(0)
                    diffs = np.diff(loss_history)
                    
                    if all(d < 0 for d in diffs):  # strictly decreasing
                        self.input_to_leaf.gumbel_noise_scale = max(self.input_to_leaf.gumbel_noise_scale * self.noise_decrease, self.min_noise)
                        # print(f"🔻 Stable improvement. Noise scale: {model.input_to_leaf.gumbel_noise_scale:.4f}")
                    elif all(abs(d) < 1e-4 for d in diffs):  # plateau
                        self.input_to_leaf.gumbel_noise_scale = min(self.input_to_leaf.gumbel_noise_scale * self.noise_increase, self.max_noise)
                        # print(f"🟰 Plateau. Noise scale: {model.input_to_leaf.gumbel_noise_scale:.4f}")
                    elif any(d > 0 for d in diffs):  # getting worse
                        self.input_to_leaf.gumbel_noise_scale = min(self.input_to_leaf.gumbel_noise_scale * self.noise_increase, self.max_noise)
                        # print(f"🔺 Loss increased. Noise scale: {model.input_to_leaf.gumbel_noise_scale:.4f}")
            
            if not self.is_frozen and loss.item() < self.freeze_loss_threshold:
                print(f"🧊 Low loss at epoch {epoch}, sampling top-k permutations...")
                candidates = self.sample_topk_permutations(
                    self.original_input_size,
                    k=self.permutation_max,  # or 0 for all
                    model_template=self,
                    X=x,
                    Y=y,
                    weight_mode=self.weight_mode,
                    weight_value=self.weight_value,
                    weight_range=self.weight_range,
                    weight_choices=self.weight_choices
                )

                best_loss = float('inf')
                best_model = None
                best_perm = None

                for perm in candidates:
                    perm_tensor = perm.clone().detach()
                    temp_model = copy.deepcopy(self)
                    temp_model.input_to_leaf = frozenInputToLeaf(perm_tensor, temp_model.original_input_size)
                    temp_loss = criterion(temp_model(x), y)
                    print(f"   🔍 Perm {perm} → Loss: {temp_loss.item():.4f}")

                    if temp_loss < best_loss:
                        best_loss = temp_loss
                        best_model = temp_model
                        best_perm = perm

                if best_model is not None and best_loss < self.freeze_loss_threshold + 0.01:
                    print(f"✅ Freezing best permutation: {best_perm} with loss {best_loss:.4f}")
                    self.locked_perm = best_perm.clone().detach()
                     # Freeze the current model in-place
                    self.input_to_leaf = frozenInputToLeaf(best_perm, self.original_input_size)

                    # Re-initialize optimizer for frozen model
                    self.optimizer = optim.Adam(self.parameters(), lr=0.01, weight_decay=1e-5)

                    # Mark frozen mode and reset patience
                    self.is_frozen = True
                    patience_counter = 0
                    best_frozen_loss = float('inf')
                    continue  # ✅ Continue training the frozen model
                else:
                    print("🚫 No good permutation found in top-k. Restarting.")
                    self.is_frozen = False
                    self.input_to_leaf = inputToLeafSinkhorn(self.original_input_size, self.num_leaves, use_gumbel=True)
                    self.optimizer = optim.Adam(self.parameters(), lr=0.01, weight_decay=1e-5)
                    
            if epoch % 200 == 0:
                logging.info(f"   Epoch {epoch} - Loss: {loss.item():.4f}")
          

    def evaluate_permutation(self, model_template, perm, X, Y, weight_mode, weight_value, weight_range, weight_choices):
        # Create a fresh model with the given permutation frozen
        model = binaryTreeLogicNet(X.shape[1], weight_mode, weight_value, weight_range, weight_choices)
        X, Y = X, Y
        model.input_to_leaf = frozenInputToLeaf(torch.tensor(perm), X.shape[1])
        model.eval()
        with torch.no_grad():
            outputs = model(X)
            if torch.isnan(outputs).any() or (outputs < 0).any() or (outputs > 1).any():
                return float("inf")  # Disqualify
            loss = nn.BCELoss()(outputs, Y)
        return loss.item()

    def sinkhorn(self, log_alpha, n_iters=20, temperature=1.0):
        log_alpha = log_alpha / temperature
        A = torch.exp(log_alpha)

        for i in range(n_iters):
            A = A / A.sum(dim=1, keepdim=True)
            A = A / A.sum(dim=0, keepdim=True)

        return A

    def sample_topk_permutations(self, num_inputs, k, model_template, X, Y, weight_mode, weight_value, weight_range, weight_choices):
        with torch.no_grad():
            if hasattr(model_template.input_to_leaf, "logits"):
                P = self.sinkhorn(model_template.input_to_leaf.logits, temperature=model_template.input_to_leaf.temperature)
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
            loss = self.evaluate_permutation(model_template, perm, X, Y, weight_mode, weight_value, weight_range, weight_choices)
            losses.append((perm, loss))

        topk_by_loss = sorted(losses, key=lambda x: x[1])
        return [torch.tensor(p[0]) for p in topk_by_loss]


    def print_tree_structure(self, labels=None):
        """Prints the recursive left-heavy binary tree structure with optional custom labels for leaves."""
        if labels is not None and len(labels) != self.num_leaves:
            raise ValueError(f"Provided label count {len(labels)} does not match number of leaves {self.num_leaves}")

        leaf_names = [labels[i] if labels else f"Leaf {i+1}" for i in range(self.num_leaves)]
        node_dict = {}
        node_labels = {}
        weight_map = {}  # Maps (parent, child) → weight

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
            node_dict[parent] = (left, right)
            node_labels[parent] = f"{parent} (andness: {a_value:.3f})"
            weight_map[(parent, left)] = w[0]
            weight_map[(parent, right)] = w[1]

        def format_tree(node, parent=None, depth=0):
            indent = "  " * depth
            weight_str = ""
            if parent is not None and (parent, node) in weight_map:
                weight_str = f" [weight: {weight_map[(parent, node)]:.3f}]"

            label = node_labels.get(node, node)
            if node in node_dict:
                # Internal node
                left, right = node_dict[node]
                left_sub = format_tree(left, node, depth + 1)
                right_sub = format_tree(right, node, depth + 1)
                return f"{indent}{label}{weight_str}\n{left_sub}\n{right_sub}"
            else:
                # Leaf node
                return f"{indent}{node}{weight_str}"

        print("\n🌲 Binary Tree Structure:\n")
        root = f"Node{self.num_layers}"
        print(format_tree(root))



