import torch
import torch.nn as nn
from bacon.inputToLeafSinkhorn import inputToLeafSinkhorn
from bacon.frozonInputToLeaf import frozenInputToLeaf
import torch.optim as optim
import numpy as np
import copy
import logging 

import torch
from scipy.optimize import linear_sum_assignment

class binaryTreeLogicNet(nn.Module):
    def __init__(self, 
                 input_size, 
                 weight_mode="trainable", 
                 weight_value=1.0, 
                 weight_range=(0.0, 1.0), 
                 weight_choices=None,
                 noise_increase=1.05,
                 noise_decrease=0.95,
                 loss_amplifier=1000.0,
                 min_noise=0.0,
                 max_noise=2.0,
                 lock_loss_tolerance=0.04,
                 freeze_loss_threshold=0.07,
                 permutation_max=10000,
                 tree_layout="left",
                 device=None):
        super(binaryTreeLogicNet, self).__init__()
        self.original_input_size = input_size
        self.num_leaves = input_size  # 🔹 Each input gets its own leaf initially
        self.weight_mode = weight_mode
        self.weight_value = weight_value
        self.weight_range = weight_range
        self.loss_amplifier = loss_amplifier
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weight_choices = torch.tensor(weight_choices, dtype=torch.float32, device=self.device) if weight_choices else None
        self.input_to_leaf = inputToLeafSinkhorn(input_size, self.num_leaves, use_gumbel=True).to(self.device)  
        self.noise_increase = noise_increase
        self.noise_decrease = noise_decrease
        self.min_noise = min_noise
        self.max_noise = max_noise
        self.lock_loss_tolerance = lock_loss_tolerance
        self.is_frozen = False
        self.freeze_loss_threshold = freeze_loss_threshold
        self.permutation_max = permutation_max
        self.locked_perm = None  # For frozen models
        self.reset_optimizer()  # Initialize optimizer
        self.tree_layout = tree_layout  # Layout for visualization
        self.num_layers = self.num_leaves - 1  # Leaf nodes feed into binary tree
        self.reinitialize(weight_mode, weight_value, weight_range, weight_choices)
    def reset_optimizer(self, learning_rate=0.2):
        """Reset the optimizer for the model."""
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=0.0)
    def reinitialize(self, weight_mode, weight_value, weight_range, weight_choices):
        self.is_frozen = False
        self.locked_perm = None
        self.input_to_leaf = inputToLeafSinkhorn(
            self.original_input_size,
            self.num_leaves,
            use_gumbel=True
        )
        self.input_to_leaf.gumbel_noise_scale = 1.0
        self.input_to_leaf.temperature = 1.0
        self.weights = nn.ParameterList()
        self.biases = nn.ParameterList()

        for _ in range(self.num_layers):
            if weight_mode == "fixed":
                self.weights.append(nn.Parameter(torch.tensor([weight_value, weight_value], dtype=torch.float32), requires_grad=False))
            elif weight_mode == "range":
                self.weights.append(nn.Parameter(
                    torch.rand(2) * (weight_range[1] - weight_range[0]) + weight_range[0]
                ))
            elif weight_mode == "discrete":
                self.weights.append(nn.Parameter(
                    torch.choice(self.weight_choices, (2,)), requires_grad=True
                ))
            else:  # "trainable"
                self.weights.append(nn.Parameter(torch.rand(1))) 

            self.biases.append(nn.Parameter(torch.rand(1) * 3 - 1))
        self.apply(self.initialize_weights)
        self.to(self.device) 

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
        self.to(self.device)

    def build_balanced_tree(self, node_outputs, weights, biases):
        def combine(start, end, depth=0):
            if start == end:
                return node_outputs[start]
            mid = (start + end) // 2
            left = combine(start, mid, depth + 1)
            right = combine(mid + 1, end, depth + 1)

            idx = combine.node_index
            a_scaled = biases[idx]
            w = weights[idx]
            out = self.generalized_gcd(left, right, a_scaled, w, 1-w)

            combine.node_index += 1
            return out

        combine.node_index = 0
        return combine(0, len(node_outputs) - 1)

    def forward(self, x):
        # 🔹 Compute input-to-leaf values
        leaf_values = self.input_to_leaf(x)

        if torch.isnan(leaf_values).any():
            raise ValueError("[DEBUG] NaNs detected in leaf_values!")

        node_outputs = list(leaf_values.T)  
        if self.tree_layout == "balanced":
            final = self.build_balanced_tree(node_outputs, self.weights, self.biases)
        else:
            layer_outputs = []  

            for i in range(self.num_layers):
                w = self.weights[i]
                bias = self.biases[i]

                a = bias
                w = self.weights[i]

                if i == 0:
                    left = node_outputs[0]
                    right = node_outputs[1]
                else:
                    left = node_outputs[-1]  # previous node
                    right = node_outputs[i + 1]  # next input
                nres = self.generalized_gcd(left, right, a, w, 1-w)
                # TODO: this is dangerous if weights a nan. In general, we should figure out why nan is happening at the first place
                if torch.isnan(nres).any():
                    nres = torch.where(torch.isnan(nres), bias, nres)
                node_outputs.append(nres)
                # node_outputs.append(self.generalized_gcd(left, right, bias, w_soft[0], w_soft[1]))
                layer_outputs.append(node_outputs[-1])
                final = node_outputs[-1]

        return final.unsqueeze(1)  # Add batch dimension
    
    def F(self, x,y,a, w0, w1):
        try:
            epsilon = 1e-6  # To prevent division by zero

            x = torch.where(torch.isnan(x), torch.tensor(epsilon, device=x.device), x)
            y = torch.where(torch.isnan(y), torch.tensor(epsilon, device=y.device), y)

            x = torch.clamp(x, min=epsilon, max=1-epsilon)
            y = torch.clamp(y, min=epsilon, max=1-epsilon)

            if not isinstance(a, torch.Tensor):
                a = torch.tensor(a, dtype=torch.float32)

            # a = a.clamp(-1.0 + epsilon, 2.0 - epsilon)  # avoid exact ends
            # a = torch.nan_to_num(a, nan=-1.0, posinf=2.0-epsilon, neginf=-1.0+epsilon)
            # a = a.clamp(-1.0 + epsilon, 2.0 - epsilon)  # avoid exact ends
            # if a == 2, return 1 of x==y==1, otherwise 0
            if torch.any(torch.abs(a - 2) < epsilon):
                cond = torch.logical_and(torch.abs(x - 1) < epsilon, torch.abs(y - 1) < epsilon)
                result = torch.where(cond, torch.ones_like(x), torch.zeros_like(x))
                if torch.isnan(result).any():
                    print(f"[TRACE] Rule 0 result has NaN: {torch.isnan(result).any()}")
                return result

            # if 1.25 < a < 2, return (xy)^(sqrt(3/(2-a))-1)
            elif torch.logical_and(a >= 0.75, a < 2):
                result = (x ** (2*w0) * y ** (2*w1)) ** (torch.sqrt(3 / (2 - a)) - 1)
                if torch.isnan(result).any():
                    print(f"[TRACE] Rule 1 result has NaN: {torch.isnan(result).any()}")
                
                return result
            
            # 1/2 < a < 3/4 return (3-4a)(0.5x+0.5y) + (4a-2)(0.5x^R+0.5y^R)^1/R
            elif torch.logical_and(a > 0.5, a < 0.75):
                result = (3-4*a)*(w0*x+w1*y) + (4*a-2)*(x ** (2*w0) * y ** (2*w1)) ** (torch.sqrt(3 / (2 - a)) - 1)
                if torch.isnan(result).any():
                    result = torch.where(torch.isnan(result), torch.tensor(float('inf'), device=result.device), result)
                    print(f"[TRACE] Rule 6 result has NaN: {torch.isnan(result).any()} scalar_a={a} x={x} y={y}")
                return result
            
            # a == 0.5 return 0.5x+0.5y
            elif torch.any(torch.abs(a - 0.5) < epsilon):
                result = w0*x + w1*y
                if torch.isnan(result).any():
                    print(f"[TRACE] Rule 7 result has NaN: {torch.isnan(result).any()}")
                return result

            # -1 <= a < 0.5 return 1-F(1-x,1-y,1-a)
            elif torch.logical_and(a >= -1, a < 0.5):
                result = 1 - self.F(1-x, 1-y, (1-a).clamp(-1.0 + epsilon, 2.0 - epsilon), w0, w1)
                #result = 1 - self.F(1-x, 1-y, 1-a, w0, w1)
                if torch.isnan(result).any():
                    print(f"[TRACE] Rule 8 result has NaN: {torch.isnan(result).any()}")
                return result
            else:
                raise ValueError(f"Invalid value for a: {a}. Must be in [-1, 2].")
        except Exception as e:
            print(f"[ERROR] Exception in F: {e}")
            print(f"[DEBUG] x: {x}, y: {y}, a: {a}, w0: {w0}, w1: {w1}")
            raise e

    def generalized_gcd(self, a, b, r, w0, w1):
        if a is None or b is None:
            raise ValueError(f"[ERROR] One of the inputs to generalized_gcd is None! a={a}, b={b}")
        if torch.isnan(a).any() or torch.isnan(b).any():
            raise ValueError(f"[ERROR] NaN input to generalized_gcd: a={a}, b={b}")
        return self.F(a, b, r, w0, w1)
    
    def train_model(self, x, y, epochs):
        self.reinitialize(self.weight_mode, self.weight_value, self.weight_range, self.weight_choices)
        loss_history = []
        self.is_frozen = False        
        self.reset_optimizer() 
        criterion = nn.BCELoss()
        best_indexes = []
        epoch = 0
        while epoch < epochs:
            if hasattr(self.input_to_leaf, "temperature") and (epoch + 1) % 1000 == 0:
                self.input_to_leaf.temperature *= 0.8
           
            self.optimizer.zero_grad()
            outputs = self(x)
            if torch.isnan(outputs).any() or (outputs < 0).any() or (outputs > 1).any():
                raise RuntimeError("Instability detected. Can't be trained further.")
            loss = criterion(outputs, y) * self.loss_amplifier
            loss.backward()
            self.optimizer.step()

            with torch.no_grad():
                for w in self.weights:
                    w.clamp_(0.0, 1.0)
            with torch.no_grad():
                for b in self.biases:
                    b.clamp_(-1.0, 2.0)

                    
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
                        # self.input_to_leaf.temperature = min(self.input_to_leaf.temperature * 1.2, 5.0)
                        # print(f"🟰 Plateau. Noise scale: {model.input_to_leaf.gumbel_noise_scale:.4f}")
                    elif any(d > 0 for d in diffs):  # getting worse
                        self.input_to_leaf.gumbel_noise_scale = min(self.input_to_leaf.gumbel_noise_scale * self.noise_increase, self.max_noise)
                        # self.input_to_leaf.temperature = min(self.input_to_leaf.temperature * 1.2, 5.0)
                        # print(f"🔺 Loss increased. Noise scale: {model.input_to_leaf.gumbel_noise_scale:.4f}")
            
            if not self.is_frozen and loss.item() < self.freeze_loss_threshold:
                print(f"🧊 Low loss at epoch {epoch}, sampling top-k permutations...")
                best_model, best_perm, best_loss, best_index = self.sample_best_permutation(
                    model_template=self,
                    topk=self.permutation_max,
                    X=x,
                    Y=y,
                    noise_std=0.1)                                    
                if best_model is not None:
                    best_indexes.append(best_index)
                if best_model is not None and best_loss < self.freeze_loss_threshold + self.lock_loss_tolerance:
                    print(f"✅ Freezing best permutation: {best_perm} with loss {best_loss:.4f}")
                    self.locked_perm = torch.tensor(best_perm, dtype=torch.long).clone().detach()
                     # Freeze the current model in-place
                    self.input_to_leaf = frozenInputToLeaf(best_perm, self.original_input_size)
                    # Re-initialize optimizer for frozen model                    
                    self.reset_optimizer(learning_rate=0.02)  # Lower learning rate for frozen model
                    # Mark frozen mode and reset patience
                    self.is_frozen = True
                    patience_counter = 0
                    epoch = 0
                    best_frozen_loss = float('inf')
                    continue  # ✅ Continue training the frozen model
                else:
                    print("🚫 No good permutation found in top-k. Restarting.")
                    self.is_frozen = False
                    self.input_to_leaf = inputToLeafSinkhorn(self.original_input_size, self.num_leaves, use_gumbel=True).to(self.device)
                    self.reset_optimizer() 
                    
            if epoch % 200 == 0:
                logging.info(f"   Epoch {epoch} - Loss: {loss.item():.4f}")
            epoch += 1
        logging.info(f"🧾 Indexes of best models: {best_indexes}")

    def sinkhorn(self, log_alpha, n_iters=20, temperature=1.0):
        log_alpha = log_alpha / temperature
        A = torch.exp(log_alpha)

        for i in range(n_iters):
            A = A / A.sum(dim=1, keepdim=True)
            A = A / A.sum(dim=0, keepdim=True)

        return A

    def sample_best_permutation(self, model_template, topk, X, Y, noise_std=0.1):
        criterion = nn.BCELoss()
        with torch.no_grad():
            if hasattr(model_template.input_to_leaf, "logits"):
                P = self.sinkhorn(
                    model_template.input_to_leaf.logits,
                    temperature=model_template.input_to_leaf.temperature
                )
            else:
                raise ValueError("Model does not have learnable logits for Sinkhorn.")

            P_np = P.cpu().numpy()
            perms = set()
            best_loss = float("inf")
            best_perm = None
            best_model = None
            best_index = None
            for index in range(topk * 2):  # Allow duplicates, just filter later
                if len(perms) == 0:
                    noisy_P = P_np  # baseline, no noise
                else:
                    noise = np.random.normal(loc=0.0, scale=noise_std, size=P_np.shape)
                    noisy_P = P_np + noise
                noisy_P = np.nan_to_num(noisy_P, nan=0.0, posinf=1e6, neginf=-1e6)
                row_ind, col_ind = linear_sum_assignment(-noisy_P)  # maximize confidence
                perm = tuple(col_ind[row_ind.argsort()])  # sort by leaf index
                if perm in perms:
                    continue
                perms.add(perm)

                perm_tensor = torch.tensor(perm, dtype=torch.long).clone().detach()
                temp_model = copy.deepcopy(model_template)
                temp_model.input_to_leaf = frozenInputToLeaf(perm_tensor, temp_model.original_input_size)
                temp_loss = criterion(temp_model(X), Y)
                print(f"   🔍 Perm {perm} → Loss: {temp_loss.item():.4f}")

                if temp_loss > 0.9 and best_index is None:
                    print(f"   🚫 Perm {perm} group rejected due to high loss.")
                    return None, None, 1, -1
                if temp_loss < best_loss:
                    best_loss = temp_loss
                    best_model = temp_model
                    best_perm = perm       
                    best_index = index             
                    
                if len(perms) >= topk:
                    break

            if best_perm is None:
                return None, None, 1, -1

            print(f"✅ Best permutation selected: {best_perm} (Loss: {best_loss:.4f})")
            return best_model, best_perm, best_loss, best_index
        
    def prune_features(self, features):
        if not self.is_frozen:
            raise RuntimeError("Model is not frozen. Can't prune features.")
        if features >= self.num_leaves:
            raise ValueError(f"Cannot prune more features than leaves. {features} > {self.num_leaves}")

        pruned_weights = self.weights[features - 1:]  # start from the first remaining aggregator
        pruned_biases = self.biases[features - 1:]

        def pruned_forward(x):
            node_outputs = list(x.T)
            layer_outputs = []
            epsilon = 1e-6
            for i in range(len(pruned_weights)):
                w = pruned_weights[i]
                a = pruned_biases[i]

                # w_soft = torch.softmax(w, dim=0)
                if i == 0:
                    # Special case: apply remaining aggregator to (0, x0)
                    left = torch.zeros_like(node_outputs[0])
                    right = node_outputs[0]
                else:
                    # Normal aggregator behavior
                    left = node_outputs[-1]
                    right = node_outputs[i]
                z = self.generalized_gcd(left, right, a, w, 1-w)
                node_outputs.append(z)
                layer_outputs.append(z)
            # final = torch.sigmoid(self.fc_out(layer_outputs[-1].unsqueeze(1)))
            # return final
            return layer_outputs[-1].unsqueeze(1)  # Remove the extra dimension

        return pruned_forward

    def prune_by_disjunction(self, threshold=-1.0):
        epsilon = 1e-6
        if not self.is_frozen:
            raise RuntimeError("Model must be frozen before pruning.")

        kept_weights = []
        kept_biases = []
        removed_indexes = []

        for i, b in enumerate(self.biases):
            # a_scaled = b.clamp(-1.0 + epsilon, 2.0 - epsilon)
            if b > threshold:
                kept_weights.append(self.weights[i])
                kept_biases.append(b)
            else:
                removed_indexes.append(i)

        print(f"🔍 Pruned {len(removed_indexes)} disjunctive nodes (a > {threshold})")

        def pruned_forward(x):
            node_outputs = list(x.T)
            layer_outputs = []
            epsilon = 1e-6
            for i in range(len(kept_weights)):
                w = kept_weights[i]
                a = kept_biases[i]
              
                if i == 0:
                    left = torch.zeros_like(node_outputs[0])
                    right = node_outputs[0]
                else:
                    left = node_outputs[-1]
                    right = node_outputs[i]

                z = self.generalized_gcd(left, right, a, w, 1-w)
                node_outputs.append(z)
                layer_outputs.append(z)

            return layer_outputs[-1].unsqueeze(1)

        return pruned_forward