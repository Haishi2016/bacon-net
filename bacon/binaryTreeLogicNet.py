import torch
import torch.nn as nn
from bacon.inputToLeafSinkhorn import inputToLeafSinkhorn
from bacon.frozonInputToLeaf import frozenInputToLeaf
from bacon.transformationLayer import (
    TransformationLayer,
    IdentityTransformation,
    NegationTransformation,
    PeakTransformation,
    ValleyTransformation,
    StepUpTransformation,
    StepDownTransformation
)
from bacon.fullyConnectedTree import FullyConnectedTree
from bacon.alternatingTree import AlternatingTree
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
        tree_layout (str, optional): Layout of the tree. Defaults to "left". Options: "left", "balanced", "paired", "full", "alternating".
        weight_penalty_strength (float, optional): Penalty strength on weights. Defaults to 1e-3. A strong penalty leads to more balaned weights (closer to 0.5).
        aggregator (callable, optional): Aggregator to be used. Defaults to "lsp.full_weight".
        device (torch.device, optional): Device to run the model on. Defaults to None (uses CUDA if available).
        full_tree_depth (int, optional): Depth of the fully connected tree. Only used when tree_layout="full". Defaults to None (uses input_size - 1).
        full_tree_shape (str, optional): Shape of the fully connected tree. "triangle" (default) or "square".
        full_tree_temperature (float, optional): Initial temperature for sigmoid edge weights. Defaults to 3.0.
        full_tree_final_temperature (float, optional): Final temperature after annealing. Defaults to 0.1.
        full_tree_max_egress (int, optional): Each source concentrates on top-K destinations (via loss). Defaults to None (no constraint).
        use_permutation_layer (bool, optional): Whether to use the permutation layer. Defaults to True.
            Set to False for full tree layout to let the tree learn input routing directly.
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
                 tree_layout="left",
                 weight_penalty_strength=1e-3,
                 aggregator="lsp.full_weight",
                 early_stop_patience = 10,
                 early_stop_min_delta = 1e-4,
                 early_stop_threshold = 0.01,
                 use_transformation_layer: bool = False,
                 transformation_temperature: float = 1.0,
                 transformation_use_gumbel: bool = False,
                 transformations = None,
                 device=None,
                 sinkhorn_iters=100,
                 # Full tree parameters
                 full_tree_depth: int = None,
                 full_tree_shape: str = "triangle",
                 full_tree_temperature: float = 3.0,
                 full_tree_final_temperature: float = 0.1,
                 full_tree_max_egress: int = None,
                 full_tree_concentrate_ingress: bool = False,
                 full_tree_use_sinkhorn: bool = False,
                 # Alternating tree parameters
                 alternating_learn_first_routing: bool = True,
                 alternating_learn_subsequent_routing: bool = True,
                 alternating_max_egress: int = 1,
                 alternating_use_straight_through: bool = True,
                 alternating_balance_weight: float = 50.0,
                 alternating_egress_weight: float = 0.5,
                 use_permutation_layer: bool = True):
        super(binaryTreeLogicNet, self).__init__()
        
        # Store full tree parameters
        self.full_tree_depth = full_tree_depth
        self.full_tree_shape = full_tree_shape
        self.full_tree_temperature = full_tree_temperature
        self.full_tree_final_temperature = full_tree_final_temperature
        self.full_tree_max_egress = full_tree_max_egress
        self.full_tree_concentrate_ingress = full_tree_concentrate_ingress
        self.full_tree_use_sinkhorn = full_tree_use_sinkhorn
        self.use_permutation_layer = use_permutation_layer
        
        # Store alternating tree parameters
        self.alternating_learn_first_routing = alternating_learn_first_routing
        self.alternating_learn_subsequent_routing = alternating_learn_subsequent_routing
        self.alternating_max_egress = alternating_max_egress
        self.alternating_use_straight_through = alternating_use_straight_through
        self.alternating_balance_weight = alternating_balance_weight
        self.alternating_egress_weight = alternating_egress_weight
        
        self.original_input_size = input_size
        self.num_leaves = input_size  # 🔹 Each input gets its own leaf initially
        self.weight_mode = weight_mode
        self.weight_value = weight_value
        self.weight_range = weight_range
        self.loss_amplifier = loss_amplifier
        self.weight_normalization = weight_normalization
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
        self.is_frozen = is_frozen
        self.locked_perm = None  # For frozen models
        self.tree_layout = tree_layout  # Layout for visualization
        self.num_layers = self.num_leaves - 1  # Leaf nodes feed into binary 
        self.aggregator = aggregator       
        self.weight_penalty_strength = weight_penalty_strength
        self.sinkhorn_iters = sinkhorn_iters  # Sinkhorn iteration count for convergence
        self.layer_outputs = None  # For storing layer outputs during forward pass
        self.pruned_aggregators = set()  # Track which aggregators have been pruned
        self.evaluation_limit = None  # For growing analysis: stop evaluation after N aggregators (None = full tree)
        
        # Transformation layer (optional)
        self.use_transformation_layer = use_transformation_layer
        self._custom_transformations = transformations  # Store for deepcopy to work correctly
        
        if self.use_transformation_layer:
            # Create default transformations if none provided
            # Use self._custom_transformations so deepcopy preserves it
            if self._custom_transformations is None:
                self._custom_transformations = [
                    IdentityTransformation(input_size),      # f(x) = x
                    NegationTransformation(input_size),      # f(x) = 1 - x
                    PeakTransformation(input_size),          # f(x) = 1 - |x - t| (high at peak)
                    ValleyTransformation(input_size),        # f(x) = |x - t| (low at valley)
                    StepUpTransformation(input_size),        # f(x) ramps 0→1 at threshold
                    StepDownTransformation(input_size)       # f(x) ramps 1→0 at threshold
                ]
            
            self.transformation_layer = TransformationLayer(
                num_features=input_size,
                transformations=self._custom_transformations,
                temperature=transformation_temperature,
                use_gumbel=transformation_use_gumbel,
                device=self.device
            ).to(self.device)
            self.add_module("transformation_layer", self.transformation_layer)
        else:
            self.transformation_layer = None
        
        self._reinitialize(weight_mode, weight_value, weight_range, weight_choices, self.is_frozen)
        self.reset_optimizer()  # Initialize optimizer        
        if hasattr(self.aggregator, "attach_to_tree"):
            # For full tree layout, compute correct number of aggregator nodes
            if self.tree_layout == "full" and self.fully_connected_tree is not None:
                # Sum of all non-input layer widths = total aggregator nodes
                num_agg_nodes = sum(self.fully_connected_tree.layer_widths[1:])
            elif self.tree_layout == "alternating" and self.alternating_tree is not None:
                # Use alternating tree's node count
                num_agg_nodes = self.alternating_tree.num_agg_nodes
            else:
                num_agg_nodes = self.num_layers
            self.aggregator.attach_to_tree(num_agg_nodes)      
            self.add_module("aggregator", self.aggregator)
            # Ensure aggregator is on the correct device after attach_to_tree creates parameters
            self.aggregator.to(self.device)

    def __deepcopy__(self, memo):
        """Custom deepcopy to preserve transformation configuration.
        
        Without this, deepcopy would create a new instance with default transformations
        instead of preserving the custom transformations passed during initialization.
        """
        import logging
        
        # Log what we're copying
        orig_trans_count = len(self.transformation_layer.transformations) if self.transformation_layer else 0
        logging.info(f"🔄 Deep copying model with {orig_trans_count} transformations")
        
        # Use state_dict approach - this preserves the exact module structure
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        
        # Copy all non-module attributes first
        for k, v in self.__dict__.items():
            if not isinstance(v, (nn.Module, nn.Parameter, nn.ParameterList)):
                try:
                    setattr(result, k, copy.deepcopy(v, memo))
                except:
                    setattr(result, k, v)
        
        # Now copy modules - this preserves their exact structure
        for k, v in self.__dict__.items():
            if isinstance(v, (nn.Module, nn.Parameter, nn.ParameterList)):
                setattr(result, k, copy.deepcopy(v, memo))
        
        # Verify result
        result_trans_count = len(result.transformation_layer.transformations) if result.transformation_layer else 0
        logging.info(f"✅ Deepcopy complete: {result_trans_count} transformations")
        
        return result

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
        if not self.use_permutation_layer:
            # No permutation - direct passthrough (useful for full tree where edge selection handles routing)
            self.locked_perm = None
            self.input_to_leaf = nn.Identity()
        elif not self.is_frozen:
            self.locked_perm = None
            self.input_to_leaf = inputToLeafSinkhorn(
                self.original_input_size,
                self.num_leaves,
                use_gumbel=True,
                sinkhorn_iters=self.sinkhorn_iters
            ).to(self.device)           
        else:
            best_perm = torch.arange(self.num_leaves, dtype=torch.long)
            self.locked_perm = best_perm.clone().detach()
            self.input_to_leaf = frozenInputToLeaf(best_perm, self.original_input_size).to(self.device)
        
        # Set temperature/noise only if the layer supports it
        if hasattr(self.input_to_leaf, 'gumbel_noise_scale'):
            self.input_to_leaf.gumbel_noise_scale = 1.0
        if hasattr(self.input_to_leaf, 'temperature'):
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
        
        # Initialize FullyConnectedTree for "full" layout
        if self.tree_layout == "full":
            self.fully_connected_tree = FullyConnectedTree(
                num_inputs=self.original_input_size,
                depth=self.full_tree_depth,
                shape=self.full_tree_shape,
                temperature=self.full_tree_temperature,
                final_temperature=self.full_tree_final_temperature,
                use_gumbel=True,
                max_egress=self.full_tree_max_egress,
                concentrate_ingress=self.full_tree_concentrate_ingress,
                use_sinkhorn=self.full_tree_use_sinkhorn,
                device=self.device
            )
            self.add_module("fully_connected_tree", self.fully_connected_tree)
        else:
            self.fully_connected_tree = None
        
        # Initialize AlternatingTree for "alternating" layout
        if self.tree_layout == "alternating":
            self.alternating_tree = AlternatingTree(
                num_inputs=self.original_input_size,
                learn_first_routing=self.alternating_learn_first_routing,
                learn_subsequent_routing=self.alternating_learn_subsequent_routing,
                max_egress=self.alternating_max_egress,
                use_straight_through=self.alternating_use_straight_through,
                temperature=self.full_tree_temperature,  # Reuse temperature params
                final_temperature=self.full_tree_final_temperature,
                use_gumbel=True,
                gumbel_noise_scale=1.0,
                device=self.device
            )
            self.add_module("alternating_tree", self.alternating_tree)
        else:
            self.alternating_tree = None
        
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
        import logging
        num_trans = len(self.transformation_layer.transformations) if self.transformation_layer else 0
        
        checkpoint = {
            'model_state_dict': self.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'is_frozen': self.is_frozen,
            'locked_perm': self.locked_perm,
            'tree_layout': self.tree_layout,
            # Save transformation configuration
            'use_transformation_layer': self.use_transformation_layer,
            'num_transformations': num_trans,
        }
        
        logging.info(f"💾 Saving model to {file_name}")
        logging.info(f"   use_transformation_layer: {self.use_transformation_layer}")
        logging.info(f"   num_transformations: {num_trans}")
        if self.transformation_layer:
            logging.info(f"   transformation types: {[type(t).__name__ for t in self.transformation_layer.transformations]}")
        
        torch.save(checkpoint, file_name)
    def load_model(self, file_name):
        """Load the model state from a file.

        Args:
            file_name (str): Path to load the model from.
            
        Raises:
            ValueError: If there's an architecture mismatch that would cause random reinitialization.
        """
        import logging
        checkpoint = torch.load(file_name, weights_only=True)
        state_dict = checkpoint['model_state_dict']
        self.is_frozen = checkpoint.get('is_frozen', False)
        self.locked_perm = checkpoint.get('locked_perm', None)
        self.tree_layout = checkpoint.get('tree_layout', getattr(self, 'tree_layout', 'left'))
        
        # Check transformation configuration compatibility
        saved_use_trans = checkpoint.get('use_transformation_layer', True)
        saved_num_trans = checkpoint.get('num_transformations', 6)
        current_num_trans = len(self.transformation_layer.transformations) if self.transformation_layer else 0
        
        if saved_use_trans != self.use_transformation_layer:
            raise ValueError(f"Architecture mismatch: Checkpoint has use_transformation_layer={saved_use_trans}, current model has {self.use_transformation_layer}. Delete checkpoint and retrain.")
        
        if self.use_transformation_layer and saved_num_trans != current_num_trans:
            raise ValueError(f"Transformation count mismatch: Checkpoint has {saved_num_trans} transformations, current model has {current_num_trans}. Delete checkpoint and retrain.")
        
        if self.is_frozen:
            # Frozen → use FrozenInputToLeaf (only if P_hard exists in state dict)
            if 'input_to_leaf.P_hard' in state_dict:
                P_hard = state_dict['input_to_leaf.P_hard']
                self.input_to_leaf = frozenInputToLeaf(
                    hard_assignment=torch.argmax(P_hard, dim=1),
                    num_inputs=self.original_input_size
                )
            elif 'input_to_leaf.logits' in state_dict:
                # Model was marked frozen but never actually converted - use logits
                import logging
                logging.warning(f"   ⚠️ Model marked frozen but has logits, not P_hard. Keeping unfrozen input layer.")
                self.is_frozen = False
            # else: neither exists, let load_state_dict handle it
        else:
            # Not marked frozen, but check if state_dict has P_hard (flag may have been lost)
            if 'input_to_leaf.P_hard' in state_dict and 'input_to_leaf.logits' not in state_dict:
                import logging
                logging.info(f"   📋 State dict has P_hard (frozen structure). Creating frozen input layer.")
                P_hard = state_dict['input_to_leaf.P_hard']
                self.input_to_leaf = frozenInputToLeaf(
                    hard_assignment=torch.argmax(P_hard, dim=1),
                    num_inputs=self.original_input_size
                )
                self.is_frozen = True
        # Now load weights - check for mismatched keys
        current_keys = set(self.state_dict().keys())
        saved_keys = set(state_dict.keys())
        missing_in_model = saved_keys - current_keys
        missing_in_checkpoint = current_keys - saved_keys
        
        if missing_in_model:
            logging.warning(f"   ⚠️ Keys in checkpoint but not in model (won't be loaded): {list(missing_in_model)[:10]}")
        if missing_in_checkpoint:
            logging.warning(f"   ⚠️ Keys in model but not in checkpoint (will keep random init): {list(missing_in_checkpoint)[:10]}")
        
        # Debug: Log specific important keys
        alternating_keys = [k for k in saved_keys if 'alternating' in k]
        operator_keys = [k for k in saved_keys if 'op_logits' in k]
        current_operator_keys = [k for k in current_keys if 'op_logits' in k]
        if alternating_keys:
            logging.info(f"   📋 Alternating tree keys in checkpoint: {len(alternating_keys)}")
        if operator_keys:
            logging.info(f"   📋 Operator logits keys in checkpoint: {operator_keys}")
            logging.info(f"   📋 Operator logits keys in model: {current_operator_keys}")
        
        # Load with strict=False for backward compatibility
        self.load_state_dict(state_dict, strict=False)
        self.to(self.device)
        
        # Verify critical values loaded correctly (wrapped in try/except to not break loading)
        try:
            if hasattr(self, 'alternating_tree') and self.alternating_tree is not None:
                if hasattr(self.alternating_tree, 'coeff_layers') and len(self.alternating_tree.coeff_layers) > 0:
                    first_coeff = self.alternating_tree.coeff_layers[0].get_coefficients()
                    logging.info(f"   📋 Loaded alternating tree first coeffs: {first_coeff.data[:3].tolist()}")
        except Exception as e:
            logging.warning(f"   ⚠️ Error verifying loaded values: {e}")
        
        # Clamp bias parameters to valid range when normalize_andness is False
        # This fixes models saved with out-of-bounds bias values
        if not self.normalize_andness:
            with torch.no_grad():
                for bias in self.biases:
                    bias.clamp_(-0.99, 1.99)  # Leave small margin from exact bounds

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
            bias = biases[idx]
            
            # Apply andness normalization if enabled
            if self.normalize_andness:
                a_scaled = torch.sigmoid(bias) * 3 - 1
            else:
                a_scaled = bias
            
            w = weights[idx]
            
            # Handle weight indexing for different weight modes
            if self.weight_mode == 'fixed':
                # Fixed weights are stored as single values, index them properly
                w_val = w[0] if hasattr(w, '__getitem__') and len(w.shape) > 0 else w
                out = self.aggregator.aggregate([left, right], a_scaled, [w_val, 1 - w_val])
            else:
                # Trainable weights
                out = self.aggregator.aggregate([left, right], a_scaled, [w, 1 - w])

            combine.node_index += 1
            return out

        combine.node_index = 0
        return combine(0, len(node_outputs) - 1)

    def build_paired_tree(self, node_outputs, weights, biases):
        """Build a two-stage 'paired' tree: first pair inputs (0,1), (2,3), ... then fold pairs.

        Args:
            node_outputs (list): Leaf outputs.
            weights (list): Aggregator weights per node.
            biases (list): Aggregator biases per node.
        Returns:
            torch.Tensor: Final aggregated output.
        """
        def get_a_w(i):
            bias = biases[i]
            if self.normalize_andness:
                a = torch.sigmoid(bias) * 3 - 1
            else:
                a = bias
            if self.weight_mode == "fixed":
                w = torch.tensor([self.weight_value, self.weight_value], dtype=torch.float32, device=self.device)
            else:
                if self.weight_normalization == "softmax":
                    w = F.softmax(weights[i], dim=0)
                elif self.weight_normalization == "minmax":
                    raw_w = weights[i]
                    w_min = raw_w.min()
                    w_max = raw_w.max()
                    denom = w_max - w_min
                    if denom.item() == 0:
                        w = torch.tensor([0.5, 0.5], device=self.device)
                    else:
                        w = (raw_w - w_min) / denom
                        w_sum = w.sum()
                        if w_sum.item() == 0:
                            w = torch.tensor([0.5, 0.5], device=self.device)
                        else:
                            w = w / w_sum
                else:
                    w = weights[i]
            return a, w

        self.layer_outputs = []
        idx = 0
        pair_outputs = []
        j = 0
        while j < len(node_outputs):
            if j + 1 < len(node_outputs):
                a, w = get_a_w(idx)
                out = self.aggregator.aggregate([node_outputs[j], node_outputs[j + 1]], a, [w[0], w[1]])
                pair_outputs.append(out)
                self.layer_outputs.append(out.detach().clone())
                idx += 1
            else:
                # Odd leftover passes through to next stage
                pair_outputs.append(node_outputs[j])
            j += 2

        # Fold all pair outputs to a single output
        current = pair_outputs[0]
        for k in range(1, len(pair_outputs)):
            a, w = get_a_w(idx)
            current = self.aggregator.aggregate([current, pair_outputs[k]], a, [w[0], w[1]])
            self.layer_outputs.append(current.detach().clone())
            idx += 1
        return current

    def forward(self, x, targets=None):
        """Forward pass through the binary tree logic network.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size).
            targets (torch.Tensor|None): Optional target tensor [batch,1] in {0,1};
                if provided and auto_refine is enabled during training, the model may
                run a light-weight permutation search and freeze the best permutation.
        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1).
        """
        try:
            # 🔹 Compute input-to-leaf values
            leaf_values = self.input_to_leaf(x)

            if torch.isnan(leaf_values).any():
                raise ValueError("[DEBUG] NaNs detected in leaf_values!")
            
            # 🔹 Apply transformation layer if enabled
            if self.transformation_layer is not None:
                leaf_values = self.transformation_layer(leaf_values)
                
                if torch.isnan(leaf_values).any():
                    raise ValueError("[DEBUG] NaNs detected after transformation layer!")
            
            if hasattr(self.aggregator, "start_forward"):
                self.aggregator.start_forward()

            node_outputs = list(leaf_values.T)
            if self.tree_layout == "balanced":
                final = self.build_balanced_tree(node_outputs, self.weights, self.biases)
            elif self.tree_layout == "paired":
                final = self.build_paired_tree(node_outputs, self.weights, self.biases)
            elif self.tree_layout == "full":
                # Use FullyConnectedTree for forward pass
                out = self.fully_connected_tree(leaf_values, aggregator=self.aggregator, 
                                                 normalize_andness=self.normalize_andness)
                # FullyConnectedTree already returns (batch, 1) shape
                # Only clamp for classification (LSP-style), not for arithmetic regression
                from bacon.aggregators.math import ArithmeticOperatorSet
                if not isinstance(self.aggregator, ArithmeticOperatorSet):
                    epsilon = 1e-7
                    out = torch.clamp(out, min=epsilon, max=1.0-epsilon)
                return out
            elif self.tree_layout == "alternating":
                # Use AlternatingTree for forward pass (separates coefficients from routing)
                out = self.alternating_tree(leaf_values, aggregator=self.aggregator)
                # AlternatingTree returns (batch, 1) shape
                from bacon.aggregators.math import ArithmeticOperatorSet
                if not isinstance(self.aggregator, ArithmeticOperatorSet):
                    epsilon = 1e-7
                    out = torch.clamp(out, min=epsilon, max=1.0-epsilon)
                return out
            else:
                self.layer_outputs = []  

                for i in range(self.num_layers):
                    bias = self.biases[i]

                    if self.normalize_andness:
                        a = torch.sigmoid(bias) * 3-1
                    else:
                        a = bias
                    
                    # Check for pruning first, before fixed mode
                    if i in self.pruned_aggregators:
                        w = self.weights[i]  # Use pruned weights directly
                    elif self.weight_mode == "fixed":
                        w = torch.tensor([self.weight_value,self.weight_value], dtype=torch.float32, device=self.device)
                    else:
                        raw_w = self.weights[i]
                        if self.weight_normalization == "softmax":
                            w = F.softmax(raw_w, dim=0)
                        elif self.weight_normalization == "minmax":
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
                            w = raw_w                

                    if i == 0:
                        left = node_outputs[0]
                        right = node_outputs[1]
                    else:
                        left = node_outputs[-1]  # previous node
                        right = node_outputs[i + 1]  # next input
                                        
                    nres = self.aggregator.aggregate([left, right], a, [w[0], w[1]])
                    
                    # TODO: this is dangerous if weights a nan. In general, we should figure out why nan is happening at the first place
                    if torch.isnan(nres).any():
                        nres = torch.where(torch.isnan(nres), bias, nres)
                    node_outputs.append(nres)
                    # node_outputs.append(self.generalized_gcd(left, right, bias, w_soft[0], w_soft[1]))
                    self.layer_outputs.append(nres.detach().clone())
                    
                    # Check if we should stop early for growing analysis
                    if self.evaluation_limit is not None and i == self.evaluation_limit - 1:
                        final = node_outputs[-1]
                        break
                    
                    final = node_outputs[-1]

            out = final.unsqueeze(1)
            
            # Clamp output to (epsilon, 1-epsilon) to ensure BCELoss compatibility and prevent NaN
            # BCE loss has log(p) and log(1-p) terms, so p=0 or p=1 causes log(0)=-inf → NaN
            # Aggregators should produce values in [0, 1], but we need strict bounds for numerical stability
            epsilon = 1e-7
            out = torch.clamp(out, min=epsilon, max=1.0-epsilon)

            return out
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
    
    def prune_features(self, feature_index):
        """Prune a single feature by adjusting its corresponding aggregator weight.
        
        In a left-associative tree:
        - Feature 0 (left input of agg 0)
        - Feature 1 (right input of agg 0)
        - Feature 2 (right input of agg 1) - uses aggregator at index 1
        - Feature k (k>=2) uses aggregator at index k-1
        
        This is designed to be called incrementally from outside to build up cumulative pruning.
        Does NOT clear existing pruning state - adds to it.

        Args:
            feature_index (int): Index of the feature to prune (0 to num_leaves-1).
        """
        if not self.is_frozen:
            raise RuntimeError("Model is not frozen. Can't prune features.")
        if feature_index >= self.num_leaves:
            raise ValueError(f"Feature index {feature_index} out of range (num_leaves={self.num_leaves})")
        
        with torch.no_grad():
            if feature_index == 0:
                # Prune feature 0: bypass left input in agg 0
                self.weights[0].data = torch.tensor([0.0, 1.0], dtype=torch.float32, device=self.device)
                self.pruned_aggregators.add(0)
            elif feature_index == 1:
                # Prune feature 1: bypass right input in agg 0
                self.weights[0].data = torch.tensor([1.0, 0.0], dtype=torch.float32, device=self.device)
                self.pruned_aggregators.add(0)
            else:
                # Prune feature k (k>=2): bypass right input (new feature) in agg k-1
                agg_idx = feature_index - 1
                self.weights[agg_idx].data = torch.tensor([1.0, 0.0], dtype=torch.float32, device=self.device)
                self.pruned_aggregators.add(agg_idx)
    
    # =========================================================================
    # Full Tree Methods (for tree_layout="full")
    # =========================================================================
    
    def get_full_tree_egress_loss(self) -> torch.Tensor:
        """Get egress constraint loss for fully connected tree.
        
        Encourages each source node to concentrate outgoing edges to top-K destinations,
        where K is controlled by full_tree_max_egress parameter.
        Returns 0 if not using "full" layout or max_egress is None.
        """
        if self.tree_layout != "full" or self.fully_connected_tree is None:
            return torch.tensor(0.0, device=self.device)
        return self.fully_connected_tree.get_egress_constraint_loss()
    
    def get_full_tree_ingress_loss(self) -> torch.Tensor:
        """Get ingress constraint loss for fully connected tree.
        
        Discourages each destination node from receiving more than 2 inputs.
        Binary operators (add, mul) work best with exactly 2 inputs.
        Returns 0 if not using "full" layout.
        """
        if self.tree_layout != "full" or self.fully_connected_tree is None:
            return torch.tensor(0.0, device=self.device)
        return self.fully_connected_tree.get_ingress_constraint_loss()
    
    def get_full_tree_ingress_balance_loss(self) -> torch.Tensor:
        """Get ingress balance loss for fully connected tree.
        
        Encourages balanced distribution of inputs across destinations.
        Prevents all sources from routing to the same destination.
        Returns 0 if not using "full" layout.
        """
        if self.tree_layout != "full" or self.fully_connected_tree is None:
            return torch.tensor(0.0, device=self.device)
        return self.fully_connected_tree.get_ingress_balance_loss()
    
    def get_full_tree_scale_regularization_loss(self) -> torch.Tensor:
        """Get scale regularization loss for fully connected tree.
        
        Penalizes extreme scale coefficients to prevent coefficient hacking.
        Returns 0 if not using "full" layout.
        """
        if self.tree_layout != "full" or self.fully_connected_tree is None:
            return torch.tensor(0.0, device=self.device)
        return self.fully_connected_tree.get_scale_regularization_loss()

    def anneal_full_tree_temperature(self, progress: float) -> None:
        """Anneal the temperature of the fully connected tree.
        
        Args:
            progress: Training progress from 0.0 to 1.0
        """
        if self.tree_layout == "full" and self.fully_connected_tree is not None:
            self.fully_connected_tree.anneal_temperature(progress)
    
    def anneal_full_tree_gumbel(self, progress: float, initial: float = 1.0, final: float = 0.0) -> None:
        """Anneal the Gumbel noise scale of the fully connected tree.
        
        Args:
            progress: Training progress from 0.0 to 1.0
            initial: Initial noise scale
            final: Final noise scale
        """
        if self.tree_layout == "full" and self.fully_connected_tree is not None:
            self.fully_connected_tree.anneal_gumbel_noise(progress, initial, final)
    
    def harden_full_tree(self, mode: str = "argmax") -> None:
        """Harden the fully connected tree to discrete edge selections.
        
        Args:
            mode: Hardening mode.
                - "argmax": destination-wise argmax by default; if max_egress==1, uses row-wise argmax.
                - "auto": row-wise argmax when max_egress==1, otherwise "smart".
                - Other modes are forwarded to FullyConnectedTree.harden(...).
        """
        if self.tree_layout == "full" and self.fully_connected_tree is not None:
            effective_mode = mode
            if mode == "auto":
                effective_mode = "argmax_row" if self.full_tree_max_egress == 1 else "smart"
            elif mode == "argmax" and self.full_tree_max_egress == 1:
                # Respect explicit egress=1 intent: one outgoing edge per source.
                effective_mode = "argmax_row"

            self.fully_connected_tree.harden(effective_mode)
            self.is_frozen = True
            logging.info(f"🔒 Full tree hardened with mode='{effective_mode}' (requested='{mode}')")
    
    def unharden_full_tree(self) -> None:
        """Revert full tree hardening to allow continued training."""
        if self.tree_layout == "full" and self.fully_connected_tree is not None:
            self.fully_connected_tree.unharden()
            self.is_frozen = False
    
    def get_full_tree_confidence(self) -> float:
        """Get confidence score for fully connected tree edge selections.
        
        Higher values indicate more peaked edge weight distributions.
        Returns 0 if not using "full" layout.
        """
        if self.tree_layout != "full" or self.fully_connected_tree is None:
            return 0.0
        return self.fully_connected_tree.get_confidence()
    
    def get_full_tree_structure(self) -> dict:
        """Get the learned structure of the fully connected tree.
        
        Returns dictionary with layer widths, significant edges, and biases.
        Returns empty dict if not using "full" layout.
        """
        if self.tree_layout != "full" or self.fully_connected_tree is None:
            return {}
        return self.fully_connected_tree.get_tree_structure()        
    
    # =========================================================================
    # Alternating Tree Methods (for tree_layout="alternating")
    # =========================================================================
    
    def get_alternating_tree_balance_loss(self) -> torch.Tensor:
        """Get balance loss for alternating tree routing.
        
        Encourages balanced distribution of inputs across destinations.
        Returns 0 if not using "alternating" layout.
        """
        if self.tree_layout != "alternating" or self.alternating_tree is None:
            return torch.tensor(0.0, device=self.device)
        return self.alternating_tree.get_balance_loss()
    
    def get_alternating_tree_egress_loss(self) -> torch.Tensor:
        """Get egress loss for alternating tree routing.
        
        Encourages peaked row distributions (clear winner per source).
        Returns 0 if not using "alternating" layout.
        """
        if self.tree_layout != "alternating" or self.alternating_tree is None:
            return torch.tensor(0.0, device=self.device)
        return self.alternating_tree.get_egress_loss()
    
    def anneal_alternating_tree_temperature(self, progress: float) -> None:
        """Anneal the temperature of the alternating tree.
        
        Args:
            progress: Training progress from 0.0 to 1.0
        """
        if self.tree_layout == "alternating" and self.alternating_tree is not None:
            self.alternating_tree.anneal_temperature(progress)
    
    def anneal_alternating_tree_gumbel(self, progress: float, initial: float = 1.0, final: float = 0.0) -> None:
        """Anneal the Gumbel noise scale of the alternating tree.
        
        Args:
            progress: Training progress from 0.0 to 1.0
            initial: Initial noise scale
            final: Final noise scale
        """
        if self.tree_layout == "alternating" and self.alternating_tree is not None:
            self.alternating_tree.anneal_gumbel(progress, initial, final)
    
    def harden_alternating_tree(self) -> None:
        """Harden the alternating tree to discrete edge selections."""
        if self.tree_layout == "alternating" and self.alternating_tree is not None:
            self.alternating_tree.harden()
            self.is_frozen = True
            logging.info("🔒 Alternating tree hardened")
    
    def unharden_alternating_tree(self) -> None:
        """Revert alternating tree hardening to allow continued training."""
        if self.tree_layout == "alternating" and self.alternating_tree is not None:
            self.alternating_tree.is_frozen = False
            for agg_layer in self.alternating_tree.agg_layers:
                if hasattr(agg_layer, 'is_hardened'):
                    agg_layer.is_hardened = False
                    agg_layer.hard_edges = None
            self.is_frozen = False
    
    def get_alternating_tree_structure_description(self) -> str:
        """Get human-readable structure description for alternating tree.
        
        Returns empty string if not using "alternating" layout.
        """
        if self.tree_layout != "alternating" or self.alternating_tree is None:
            return ""
        return self.alternating_tree.get_structure_description()
    
    def get_alternating_tree_num_nodes(self) -> int:
        """Get number of aggregation nodes in alternating tree.
        
        Returns 0 if not using "alternating" layout.
        """
        if self.tree_layout != "alternating" or self.alternating_tree is None:
            return 0
        return self.alternating_tree.num_agg_nodes