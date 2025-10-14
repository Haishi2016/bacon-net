import torch
import torch.nn as nn
from bacon.inputToLeafSinkhorn import inputToLeafSinkhorn
from bacon.frozonInputToLeaf import frozenInputToLeaf
import torch.optim as optim
import numpy as np
import copy
import logging 
import torch.nn.functional as F

import torch
from scipy.optimize import linear_sum_assignment

class binaryTreeLogicNet(nn.Module):
    """    Represents a binary tree logic network for interpretable decision-making using graded logic.

    Args:
        input_size (int): Number of input features.
        weight_mode (str, optional): Mode for weight configuration. Defaults to "trainable".
        weight_value (float, optional): Initial value for fixed weights. Defaults to 0.5.
        weight_range (tuple, optional): Range for random weights. Defaults to (0.0, 1.0).
        weight_choices (list, optional): Choices for discrete weights. Defaults to None.
        noise_increase (float, optional): Factor to increase noise. Defaults to 1.05.
        noise_decrease (float, optional): Factor to decrease noise. Defaults to 0.95.
        loss_amplifier (float, optional): Amplifier for the loss. Defaults to 1000.0.
        min_noise (float, optional): Minimum noise level. Defaults to 0.0.
        max_noise (float, optional): Maximum noise level. Defaults to 2.0.
        is_frozen (bool, optional): Whether to freeze the structure. Defaults to False.
        lock_loss_tolerance (float, optional): The maximum tolerated accuracy loss when locking the structure. Defaults to 0.04. Note that this is multiplied by `loss_amplifier`.
        freeze_loss_threshold (float, optional): Loss threshold at which to freeze structure learning. Defaults to 0.07. Note that this is multiplied by `loss_amplifier`.
        permutation_max (int, optional): Maximum permutations to explore. Defaults to 10000.
        tree_layout (str, optional): Layout of the tree. Defaults to "left".
        weight_penalty_strength (float, optional): Penalty strength on weights. Defaults to 1e-3. A strong penalty leads to more balaned weights (closer to 0.5).
        aggregator (callable, optional): Aggregator to be used. Defaults to "lsp.full_weight".
        device (torch.device, optional): Device to run the model on. Defaults to None (uses CUDA if available).
    """
    def __init__(self, 
                 input_size, 
                 weight_mode="trainable", 
                 weight_normalization="minmax",
                 weight_value=0.5, 
                 weight_range=(0.0, 1.0), 
                 weight_choices=None,
                 noise_increase=1.05,
                 noise_decrease=0.95,
                 loss_amplifier=1.0,
                 normalize_andness= True,
                 min_noise=0.0,
                 max_noise=2.0,
                 is_frozen = False,
                 lock_loss_tolerance=0.04,
                 freeze_loss_threshold=0.07,
                 permutation_max=10000,
                 tree_layout="left",
                 weight_penalty_strength=1e-3,
                 aggregator="lsp.full_weight",
                 early_stop_patience = 10,
                 early_stop_min_delta = 1e-4,
                 early_stop_threshold = 0.01,
                 device=None):
        super(binaryTreeLogicNet, self).__init__()
        self.original_input_size = input_size
        self.num_leaves = input_size  # 🔹 Each input gets its own leaf initially
        self.weight_mode = weight_mode
        self.weight_value = weight_value
        self.weight_range = weight_range
        self.loss_amplifier = loss_amplifier
        self.weight_normalization = weight_normalization
        self.aggregator = aggregator
        self.device = device if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.weight_choices = torch.tensor(weight_choices, dtype=torch.float32, device=self.device) if weight_choices else None
        self.noise_increase = noise_increase
        self.noise_decrease = noise_decrease
        self.min_noise = min_noise
        self.max_noise = max_noise
        self.normalize_andness = normalize_andness
        self.early_stop_patience = early_stop_patience
        self.early_stop_min_delta = early_stop_min_delta
        self.early_stop_threshold = early_stop_threshold
        self.lock_loss_tolerance = lock_loss_tolerance * self.loss_amplifier  # Adjust tolerance based on loss amplifier
        self.is_frozen = is_frozen
        self.freeze_loss_threshold = freeze_loss_threshold * self.loss_amplifier  # Adjust threshold based on loss amplifier
        self.permutation_max = permutation_max
        self.locked_perm = None  # For frozen models
        self.tree_layout = tree_layout  # Layout for visualization
        self.num_layers = self.num_leaves - 1  # Leaf nodes feed into binary 
        self.weight_penalty_strength = weight_penalty_strength
        self.layer_outputs = None  # For storing layer outputs during forward pass
        self._reinitialize(weight_mode, weight_value, weight_range, weight_choices, self.is_frozen)
        self.reset_optimizer()  # Initialize optimizer
    def reset_optimizer(self, learning_rate=0.2):
        """Reset the optimizer for the model.
        Args:
            learning_rate (float, optional): Learning rate for the optimizer. Defaults to 0.2.
        """
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, weight_decay=0.0)
    def _reinitialize(self, weight_mode, weight_value, weight_range, weight_choices, is_frozen=False):
        """Reinitialize the model with new parameters.

        Args:
            weight_mode (str): Mode for weight configuration.
            weight_value (float): Initial value for fixed weights.
            weight_range (tuple): Range for random weights.
            weight_choices (list): Choices for discrete weights.
            is_frozen (bool): Whether to freeze the structure.
        """
        self.is_frozen = is_frozen
        if not self.is_frozen:
            self.locked_perm = None
            self.input_to_leaf = inputToLeafSinkhorn(
                self.original_input_size,
                self.num_leaves,
                use_gumbel=True
            ).to(self.device)
        else:
            best_perm = torch.arange(self.num_leaves, dtype=torch.long)
            self.locked_perm = best_perm.clone().detach()
            self.input_to_leaf = frozenInputToLeaf(best_perm, self.original_input_size).to(self.device)
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
                self.weights.append(nn.Parameter(torch.rand(2))) 

            self.biases.append(nn.Parameter(torch.rand(1) * 3 - 1))
        self.apply(self._initialize_weights)
        self.to(self.device) 

    def _initialize_weights(self, m):
        """Initialize weights of the model.

        Args:
            m (nn.Module): Module to initialize.
        """
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.5)  
    def save_model(self, file_name):
        """Save the model state to a file.

        Args:
            file_name (str): Path to save the model.
        """
        torch.save({
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'is_frozen': self.is_frozen,
            'locked_perm': self.locked_perm,
        }, file_name)
    def load_model(self, file_name):
        """Load the model state from a file.

        Args:
            file_name (str): Path to load the model from.
        """
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
        """Build a balanced binary tree from node outputs.

        Args:
            node_outputs (list): List of node outputs.
            weights (list): List of weights for the nodes.  
            biases (list): List of biases for the nodes.
        Returns:
            torch.Tensor: Final output of the balanced tree.
        """
        def combine(start, end, depth=0):
            if start == end:
                return node_outputs[start]
            mid = (start + end) // 2
            left = combine(start, mid, depth + 1)
            right = combine(mid + 1, end, depth + 1)

            idx = combine.node_index
            a_scaled = biases[idx]
            w = weights[idx]
            out = self.aggregator.aggregate(left, right, a_scaled, w, 1-w)

            combine.node_index += 1
            return out

        combine.node_index = 0
        return combine(0, len(node_outputs) - 1)

    def forward(self, x):
        """Forward pass through the binary tree logic network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        try:
            # 🔹 Compute input-to-leaf values
            leaf_values = self.input_to_leaf(x)

            if torch.isnan(leaf_values).any():
                raise ValueError("[DEBUG] NaNs detected in leaf_values!")

            node_outputs = list(leaf_values.T)  
            if self.tree_layout == "balanced":
                final = self.build_balanced_tree(node_outputs, self.weights, self.biases)
            else:
                self.layer_outputs = []  

                for i in range(self.num_layers):
                    bias = self.biases[i]

                    if self.normalize_andness:
                        a = torch.sigmoid(bias) * 3-1
                    else:
                        a = bias
                    if self.weight_mode == "fixed":
                        w = torch.tensor([self.weight_value,self.weight_value], dtype=torch.float32, device=self.device)
                    else:
                        if self.weight_normalization == "softmax":
                            w = F.softmax(self.weights[i], dim=0)
                        elif self.weight_normalization == "minmax":
                            raw_w = self.weights[i]
                            w_min = raw_w.min()
                            w_max = raw_w.max()
                            denom = w_max - w_min

                            if denom.item() == 0:
                                # Both weights are equal — fallback to uniform weights
                                w = torch.tensor([0.5, 0.5], device=self.device)
                            else:
                                w = (raw_w - w_min) / denom
                                w_sum = w.sum()
                                if w_sum.item() == 0:
                                    w = torch.tensor([0.5, 0.5], device=self.device)
                                else:
                                    w = w / w_sum        
                        else:
                            w = self.weights[i]                

                    if i == 0:
                        left = node_outputs[0]
                        right = node_outputs[1]
                    else:
                        left = node_outputs[-1]  # previous node
                        right = node_outputs[i + 1]  # next input
                    nres = self.aggregator.aggregate(left, right, a, w[0], w[1])
                    # TODO: this is dangerous if weights a nan. In general, we should figure out why nan is happening at the first place
                    if torch.isnan(nres).any():
                        nres = torch.where(torch.isnan(nres), bias, nres)
                    node_outputs.append(nres)
                    # node_outputs.append(self.generalized_gcd(left, right, bias, w_soft[0], w_soft[1]))
                    self.layer_outputs.append(nres.detach().clone())
                    final = node_outputs[-1]

            return final.unsqueeze(1)  # Add batch dimension
        except Exception as e:
            logging.error(f"[ERROR] Exception in forward pass: {e}")
            logging.error(f"[DEBUG] Input x: {x}")
            raise e
        
    def train_model(self, x, y, epochs, is_frozen):
        """Train the binary tree logic network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
            y (torch.Tensor): Target tensor of shape (batch_size, 1).   
            epochs (int): Number of training epochs.
        Returns:
            list: History of loss values during training.
        """
        self._reinitialize(self.weight_mode, self.weight_value, self.weight_range, self.weight_choices, is_frozen)
        loss_history = []
        self.reset_optimizer() 
        criterion = nn.BCELoss()
        best_indexes = []
        epoch = 0
        patience_counter = 0
        best_loss = float('inf')
        while epoch < epochs:
            if hasattr(self.input_to_leaf, "temperature") and (epoch + 1) % 1000 == 0:
                self.input_to_leaf.temperature *= 0.8
           
            self.optimizer.zero_grad()
            outputs = self(x)
            if torch.isnan(outputs).any() or (outputs < 0).any() or (outputs > 1).any():
                raise RuntimeError("Instability detected. Can't be trained further.")
            

            loss = criterion(outputs, y)

            if self.weight_mode == "trainable":
                depth_weight_penalty = 0.0
                N = len(self.weights)
                for i, (w, b) in enumerate(zip(self.weights, self.biases)):
                    # Existing depth penalty (optional baseline)
                    penalty_strength = self.weight_penalty_strength # * (i + 1)
                    depth_weight_penalty += penalty_strength * ((torch.sigmoid(w) - 0.5) ** 2).mean()

                loss += depth_weight_penalty

            loss = loss * self.loss_amplifier  # Amplify loss

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

            if self.is_frozen:
                if loss.item() < best_loss - self.early_stop_min_delta:
                    best_loss = loss.item()
                    patience_counter = 0
                else:
                    patience_counter += 1

                if best_loss < self.early_stop_threshold:
                    logging.info(f"🎯 Early stopping triggered by reaching low loss: {best_loss:.6f} at epoch {epoch}")
                    break

                if patience_counter >= self.early_stop_patience:
                    logging.info(f"🛑 Early stopping triggered by plateau. Best loss: {best_loss:.6f}")
                    break
        
            if epoch % 200 == 0:
                logging.info(f"   🏋️ Epoch {epoch} - Loss: {loss.item():.4f}")
            epoch += 1
        logging.info(f"🧾 Indexes of best models: {best_indexes}")

    def _sinkhorn(self, log_alpha, n_iters=20, temperature=1.0):
        """Sinkhorn normalization to convert logits to a doubly stochastic matrix.

        Args:
            log_alpha (torch.Tensor): Logits of shape (n_leaves, n_leaves).
            n_iters (int, optional): Number of Sinkhorn iterations. Defaults to 20.
            temperature (float, optional): Temperature for scaling logits. Defaults to 1.0.
        Returns:
            torch.Tensor: Doubly stochastic matrix of shape (n_leaves, n_leaves).
        """
        log_alpha = log_alpha / temperature
        A = torch.exp(log_alpha)

        for i in range(n_iters):
            A = A / A.sum(dim=1, keepdim=True)
            A = A / A.sum(dim=0, keepdim=True)

        return A

    def sample_best_permutation(self, model_template, topk, X, Y, noise_std=0.1):
        """Sample the best permutation of leaves based on Sinkhorn normalization.

        Args:
            model_template (binaryTreeLogicNet): Template model to sample from.
            topk (int): Number of top permutations to sample.
            X (torch.Tensor): Input tensor of shape (batch_size, input_size).
            Y (torch.Tensor): Target tensor of shape (batch_size, 1).
            noise_std (float, optional): Standard deviation of noise to add. Defaults to 0.1.
        Returns:
            tuple: Best model, best permutation, best loss, and index of the best permutation.
        """
        criterion = nn.BCELoss()
        with torch.no_grad():
            if hasattr(model_template.input_to_leaf, "logits"):
                P = self._sinkhorn(
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
        """Prune the features of the binary tree logic network.

        Args:
            features (int): Number of features to prune.
        Returns:
            callable: Pruned forward function.
        """
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
                if self.weight_mode == "fixed":
                    w = torch.tensor([self.weight_value,self.weight_value], dtype=torch.float32, device=self.device)
                else:
                    if self.weight_normalization == "softmax":
                        w = F.softmax(pruned_weights[i], dim=0)
                    elif self.weight_normalization == "minmax":
                        raw_w = pruned_weights[i]
                        w_min = raw_w.min()
                        w_max = raw_w.max()
                        denom = w_max - w_min

                        if denom.item() == 0:
                            # Both weights are equal — fallback to uniform weights
                            w = torch.tensor([0.5, 0.5], device=self.device)
                        else:
                            w = (raw_w - w_min) / denom
                            w_sum = w.sum()
                            if w_sum.item() == 0:
                                w = torch.tensor([0.5, 0.5], device=self.device)
                            else:
                                w = w / w_sum
                    else:
                        w = pruned_weights[i]              
                
                if self.normalize_andness:
                    a = torch.sigmoid(pruned_biases[i])*3-1
                else:
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
                z = self.aggregator.aggregate(left, right, a, w[0], w[1])
                node_outputs.append(z)
                layer_outputs.append(z)
            # final = torch.sigmoid(self.fc_out(layer_outputs[-1].unsqueeze(1)))
            # return final
            return layer_outputs[-1].unsqueeze(1)  # Remove the extra dimension

        return pruned_forward