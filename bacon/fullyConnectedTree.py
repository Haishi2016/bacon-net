"""
Fully Connected Tree Module

This module implements a fully connected network structure with parameterized depth
that can be trained to converge to a sparse tree structure.

Key features:
- Fully connected between adjacent layers
- Edge weights use sigmoid activation (learnable, independent weights)
- Optional max_egress regulation: encourage each source to concentrate on top-K destinations
- Explicit hardening process to enforce discrete edges after training
- Compatible with existing tree_layout options ("left", "balanced", "full")

Architecture:
- Layer 0: Input features (n_inputs nodes)
- Layer 1 to depth-1: Hidden layers (configurable width, typically n_inputs)
- Layer depth: Single output node

Each node in layer L can receive inputs from all nodes in layer L-1.
Edge weights are sigmoid(logits/temperature) - simple and differentiable.
Concentration is controlled via max_egress loss, not edge normalization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from typing import Optional, Tuple, List


class FullyConnectedTree(nn.Module):
    """
    A fully connected network that can be trained to become a sparse tree.
    
    Args:
        num_inputs (int): Number of input features (layer 0 width)
        depth (int): Number of aggregation layers. depth=n_inputs-1 matches left/balanced tree depth.
        layer_widths (list[int], optional): Width of each hidden layer. 
            If None, uses constant width = num_inputs for all hidden layers, then 1 for output.
        temperature (float): Initial temperature for sigmoid (higher = softer edges)
        final_temperature (float): Final temperature after annealing (lower = sharper edges)
        use_gumbel (bool): Whether to add Gumbel noise for stochastic exploration
        gumbel_noise_scale (float): Scale of Gumbel noise (annealed during training)
        max_egress (int, optional): Each source concentrates on top-K destinations (via loss)
        device (torch.device): Device to run on
    
    The network learns:
    - edge_logits: Per-layer edge logit matrices [depth x (layer_in x layer_out)]  
    - aggregator_biases: Andness parameter per destination node
    """
    
    def __init__(
        self,
        num_inputs: int,
        depth: Optional[int] = None,
        layer_widths: Optional[List[int]] = None,
        shape: str = "triangle",  # "triangle" or "square"
        temperature: float = 3.0,
        final_temperature: float = 0.1,
        use_gumbel: bool = True,
        gumbel_noise_scale: float = 1.0,
        max_egress: Optional[int] = None,
        concentrate_ingress: bool = False,
        use_sinkhorn: bool = False,
        device: Optional[torch.device] = None
    ):
        super().__init__()
        
        self.num_inputs = num_inputs
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.shape = shape
        self.use_sinkhorn = use_sinkhorn
        
        # Default depth matches left/balanced tree (n-1 aggregators for n inputs)
        if depth is None:
            depth = num_inputs - 1
        self.depth = depth
        
        # Compute layer widths based on shape
        if layer_widths is None:
            if use_sinkhorn:
                # Sinkhorn mode: square layers for proper doubly-stochastic matrices
                # We simulate triangular reduction using phantom nodes (zeroed outputs)
                layer_widths = [num_inputs] * (depth + 1)
            elif shape == "triangle":
                # Triangular: [n, n-1, n-2, ..., 1] - natural reduction like binary tree
                layer_widths = list(range(num_inputs, 0, -1))[:depth + 1]
                # Ensure we have enough layers and end with 1
                while len(layer_widths) < depth + 1:
                    layer_widths.append(1)
            else:  # "square"
                # Square: [n, n, n, ..., 1] - full connectivity at each layer
                layer_widths = [num_inputs] * depth + [1]
        
        # For non-sinkhorn modes, ensure output is 1
        if not use_sinkhorn and layer_widths[-1] != 1:
            layer_widths[-1] = 1
            
        self.layer_widths = layer_widths
        
        # Create active node masks for sinkhorn mode (simulate triangular tree)
        # active_masks[l] is a tensor of shape (layer_widths[l],) with 1.0 for active, 0.0 for phantom
        # Registered as buffers so they move with the model to the correct device
        self._n_active_masks = 0
        if use_sinkhorn:
            for l in range(len(layer_widths)):
                # Number of active nodes decreases: n, n-1, n-2, ..., 1
                n_active = max(1, num_inputs - l)
                mask = torch.zeros(layer_widths[l])
                mask[:n_active] = 1.0
                self.register_buffer(f"active_mask_{l}", mask)
                self._n_active_masks += 1
            logging.info(f"Sinkhorn mode: square layers {layer_widths}, active nodes per layer: {[max(1, num_inputs - l) for l in range(len(layer_widths))]}")
        
        self.temperature = temperature
        self.final_temperature = final_temperature
        self.use_gumbel = use_gumbel
        self.gumbel_noise_scale = gumbel_noise_scale
        self.max_egress = max_egress  # Row-wise softmax: each source concentrates on few destinations
        self.concentrate_ingress = concentrate_ingress  # Column-wise softmax: each dest concentrates on few sources
        self.max_ingress = 2  # For soft penalty loss calculation (not hard constraint)
        
        # Edge logits: one matrix per layer transition
        # edge_logits[l] is shape (layer_widths[l], layer_widths[l+1])
        # Represents selection weights from layer l to layer l+1
        self.edge_logits = nn.ParameterList()
        # Edge scales: separate learnable scale/coefficient per edge (unbounded)
        # Allows edges to have coefficients > 1 or < 0
        self.edge_scales = nn.ParameterList()
        for l in range(self.depth):
            in_width = self.layer_widths[l]
            out_width = self.layer_widths[l + 1]
            # Initialize logits to 3.0 for near-unity selection via sigmoid
            # sigmoid(3.0) ≈ 0.95, so edges start nearly fully selected
            # Learning then "prunes" less useful edges by driving logits negative
            logits = torch.full((in_width, out_width), 3.0)
            self.edge_logits.append(nn.Parameter(logits))
            # Initialize scales to 1.0 (no scaling initially)
            scales = torch.ones(in_width, out_width)
            self.edge_scales.append(nn.Parameter(scales))
        
        # Aggregator biases (andness parameter) per destination node per layer
        # biases[l] has shape (layer_widths[l+1],) - one bias per node in layer l+1
        self.biases = nn.ParameterList()
        for l in range(self.depth):
            out_width = self.layer_widths[l + 1]
            bias = torch.randn(out_width)
            self.biases.append(nn.Parameter(bias))
        
        # For frozen/hardened models
        self.is_frozen = False
        self.hardened_edges = None  # Stores discrete edge selections after hardening
        
        self.to(self.device)
    
    def _sample_gumbel(self, shape: tuple, eps: float = 1e-20) -> torch.Tensor:
        """Sample from Gumbel(0, 1) distribution."""
        U = torch.rand(shape, device=self.device)
        return -torch.log(-torch.log(U + eps) + eps)
    
    def _get_active_mask(self, layer_idx: int) -> Optional[torch.Tensor]:
        """Get the active mask for a layer (Sinkhorn mode only)."""
        if self._n_active_masks > 0:
            return getattr(self, f"active_mask_{layer_idx}")
        return None

    def _compute_edge_weights(self, layer_idx: int) -> torch.Tensor:
        """
        Compute edge selection weights for a layer transition.
        
        Concentration modes:
        - max_egress (row-wise softmax): each source concentrates on few destinations  
        - concentrate_ingress (column-wise softmax): each destination concentrates on few sources
        - use_sinkhorn: alternating row/column normalization for balanced assignment
        
        Returns:
            Edge selection weight matrix of shape (in_width, out_width)
        """
        if self.is_frozen and self.hardened_edges is not None:
            return self.hardened_edges[layer_idx]
        
        logits = self.edge_logits[layer_idx]  # (in_width, out_width)
        
        # Add Gumbel noise during training for exploration
        if self.training and self.use_gumbel and self.gumbel_noise_scale > 0:
            noise = self._sample_gumbel(logits.shape) * self.gumbel_noise_scale
            logits = logits + noise
        
        # Choose weight computation based on concentration mode
        if self.use_sinkhorn:
            # Sinkhorn for square matrices with phantom node masking
            weights = self._sinkhorn_with_mask(logits / self.temperature, layer_idx)
        elif self.concentrate_ingress:
            # Column-wise softmax: each destination concentrates on few sources
            weights = F.softmax(logits / self.temperature, dim=0)
        elif self.max_egress is not None:
            # Row-wise softmax: each source concentrates on few destinations
            weights = F.softmax(logits / self.temperature, dim=1)
        else:
            # Independent sigmoid: each edge weight is independent
            weights = torch.sigmoid(logits / self.temperature)
        
        return weights
    
    def _sinkhorn_with_mask(self, logits: torch.Tensor, layer_idx: int, iters: int = 10) -> torch.Tensor:
        """
        Log-domain Sinkhorn for square matrices with phantom node masking.
        
        For layer transition l: sources are layer l, destinations are layer l+1
        - Active sources: determined by active_masks[l]
        - Active destinations: determined by active_masks[l+1]
        
        Phantom destinations (inactive) get -inf logits so no edges route to them.
        Phantom sources (inactive) similarly have their rows zeroed.
        
        This creates a doubly-stochastic matrix over the active submatrix.
        """
        # Get active masks for source and destination layers
        src_mask = self._get_active_mask(layer_idx)  # (in_width,)
        dst_mask = self._get_active_mask(layer_idx + 1)  # (out_width,)
        
        # Mask logits: set phantom destinations to -inf (they receive no edges)
        # Phantom sources will have their row contributions ignored
        log_weights = logits.clone()
        log_weights = log_weights.clamp(-20, 20)  # Prevent overflow
        
        # Mask out phantom destinations (columns)
        dst_mask_expanded = dst_mask.unsqueeze(0)  # (1, out_width)
        log_weights = log_weights + torch.where(
            dst_mask_expanded > 0.5,
            torch.zeros_like(log_weights),
            torch.full_like(log_weights, -1e6)  # Large negative instead of -inf for stability
        )
        
        # Mask out phantom sources (rows)
        src_mask_expanded = src_mask.unsqueeze(1)  # (in_width, 1)
        log_weights = log_weights + torch.where(
            src_mask_expanded > 0.5,
            torch.zeros_like(log_weights),
            torch.full_like(log_weights, -1e6)
        )
        
        # Log-domain Sinkhorn iterations
        for _ in range(iters):
            # Row normalization: subtract logsumexp of each row
            log_weights = log_weights - torch.logsumexp(log_weights, dim=1, keepdim=True)
            # Column normalization: subtract logsumexp of each column  
            log_weights = log_weights - torch.logsumexp(log_weights, dim=0, keepdim=True)
        
        # Convert to probability domain
        weights = torch.exp(log_weights.clamp(-20, 20))
        
        # Zero out phantom rows/columns explicitly for clean gradients
        weights = weights * src_mask_expanded * dst_mask_expanded
        
        return weights
    
    def _get_edge_scales(self, layer_idx: int) -> torch.Tensor:
        """
        Get the learnable scale/coefficient for each edge.
        
        These are unbounded and allow edges to have coefficients > 1 or negative.
        The final edge contribution is: selection_weight * scale * input
        
        Returns:
            Edge scale matrix of shape (in_width, out_width)
        """
        return self.edge_scales[layer_idx]
    
    def forward(self, x: torch.Tensor, aggregator=None, normalize_andness: bool = True) -> torch.Tensor:
        """
        Forward pass through the fully connected tree.
        
        Args:
            x: Input tensor of shape (batch_size, num_inputs)
            aggregator: Aggregator module with .aggregate() method
            normalize_andness: Whether to apply sigmoid normalization to biases
            
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        if aggregator is None:
            raise ValueError("aggregator must be provided for FullyConnectedTree forward pass")
        
        # Current layer values: (batch_size, layer_width)
        current = x  # Shape: (batch_size, num_inputs)
        
        if hasattr(aggregator, "start_forward"):
            aggregator.start_forward()
        
        # Process each layer
        for l in range(self.depth):
            edge_weights = self._compute_edge_weights(l)  # (in_width, out_width) - selection
            edge_scales = self._get_edge_scales(l)        # (in_width, out_width) - coefficients
            bias = self.biases[l]  # (out_width,)
            
            out_width = self.layer_widths[l + 1]
            next_values = []
            
            # For each destination node in the next layer
            for j in range(out_width):
                # Get selection weights and scales for edges going to node j
                w_col = edge_weights[:, j]  # (in_width,)
                s_col = edge_scales[:, j]   # (in_width,) - coefficients
                
                # Compute andness from bias
                if normalize_andness:
                    a = torch.sigmoid(bias[j]) * 3 - 1  # Range [-1, 2]
                else:
                    a = bias[j]
                
                # Aggregation with selection weights and scales
                node_output = self._aggregate_weighted(
                    current, w_col, s_col, a, aggregator
                )
                next_values.append(node_output)
            
            # Stack to get next layer: (batch_size, out_width)
            current = torch.stack(next_values, dim=1)
            
            # In Sinkhorn mode, zero out phantom nodes (simulate triangular tree)
            if self.use_sinkhorn and self._n_active_masks > 0:
                mask = self._get_active_mask(l + 1).unsqueeze(0)  # (1, out_width)
                current = current * mask
        
        # Final output: (batch_size, 1) or (batch_size, num_inputs) in Sinkhorn mode
        if self.use_sinkhorn:
            # In Sinkhorn mode, only the first node of the last layer is the output
            return current[:, :1]
        return current
    
    def _aggregate_weighted(
        self, 
        inputs: torch.Tensor,   # (batch, in_width)
        weights: torch.Tensor,  # (in_width,) - selection weights (softmax or sigmoid)
        scales: torch.Tensor,   # (in_width,) - coefficients (unbounded)
        andness: torch.Tensor,  # scalar
        aggregator
    ) -> torch.Tensor:
        """
        Aggregate multiple weighted inputs into a single output.
        
        Each destination node makes ONE aggregator call with all its inputs.
        The aggregator's operator (add, sub, mul, div) is applied with combined weights.
        
        Combined weight per input = selection_weight * scale
        
        This ensures one operator selection per destination node.
        """
        _, in_width = inputs.shape

        if getattr(aggregator, "uses_edge_scales", True):
            combined_weights = weights * scales
        else:
            combined_weights = weights
        
        # Collect all inputs as a list for the aggregator
        input_list = [inputs[:, i] for i in range(in_width)]
        weight_list = [combined_weights[i] for i in range(in_width)]
        
        # Single aggregate call - one operator choice per node
        result = aggregator.aggregate(input_list, andness, weight_list)
        
        return result
    
    def get_edge_weights(self) -> List[torch.Tensor]:
        """Get all edge weight matrices (for visualization/debugging)."""
        return [self._compute_edge_weights(l) for l in range(self.depth)]
    
    def get_egress_constraint_loss(self) -> torch.Tensor:
        """
        Compute loss to encourage peaked edge distributions (one dominant edge per row).
        
        With row-wise softmax, each row sums to 1. This loss encourages the distribution
        to be peaked rather than uniform, so each source concentrates on one destination.
        
        Uses negative entropy: peaked distributions have low entropy, uniform has high.
        """
        if self.max_egress is None:
            return torch.tensor(0.0, device=self.device)
        
        loss = torch.tensor(0.0, device=self.device)
        eps = 1e-8
        
        for l in range(self.depth):
            weights = self._compute_edge_weights(l)  # (in_width, out_width)
            in_width, out_width = weights.shape
            
            # Skip if only one output (trivially peaked)
            if out_width <= 1:
                continue
            
            # Entropy per row: -sum(p * log(p))
            # Lower entropy = more peaked distribution
            row_entropy = -(weights * torch.log(weights + eps)).sum(dim=1)
            
            # We want low entropy, so add it as a penalty
            # Normalize by max possible entropy (uniform distribution)
            max_entropy = torch.log(torch.tensor(float(out_width), device=self.device))
            normalized_entropy = row_entropy / max_entropy  # In [0, 1]
            
            loss = loss + normalized_entropy.mean()
        
        return loss / self.depth
    
    def get_ingress_constraint_loss(self) -> torch.Tensor:
        """
        Compute loss to discourage nodes from receiving more than max_ingress inputs.
        
        Binary operators (add, mul) naturally take 2 inputs. Having 3+ inputs
        forces the aggregator to compute (a op b op c) which may not match
        the intended expression structure.
        
        Uses a soft penalty: sum of edge weights per destination that exceed max_ingress.
        """
        loss = torch.tensor(0.0, device=self.device)
        
        for l in range(self.depth):
            weights = self._compute_edge_weights(l)  # (in_width, out_width)
            in_width, out_width = weights.shape
            
            # Skip if inputs <= max_ingress (can't exceed)
            if in_width <= self.max_ingress:
                continue
            
            # Column-wise: sum of incoming edge weights per destination
            # This is the "effective number of inputs" per destination
            ingress_sum = weights.sum(dim=0)  # (out_width,)
            
            # Penalty: how much ingress exceeds max_ingress (soft ReLU)
            # If ingress_sum = 2.0 and max_ingress = 2, penalty = 0
            # If ingress_sum = 2.5 and max_ingress = 2, penalty = 0.5
            excess = torch.relu(ingress_sum - self.max_ingress)
            loss = loss + excess.mean()
        
        return loss / self.depth if self.depth > 0 else loss
    
    def get_ingress_balance_loss(self) -> torch.Tensor:
        """
        Compute loss to encourage balanced distribution of inputs across destinations.
        
        With row-softmax, each source picks 1 destination. Without balance constraint,
        all sources might pick the same destination. This loss penalizes when column
        sums are imbalanced (one destination gets disproportionately many inputs).
        
        For 3 sources → 2 destinations, fair share is 1.5 per destination.
        If one destination gets 2.5 and the other 0.5, that's imbalanced.
        
        Uses logits directly to ensure gradients flow even when weights are sharpened.
        """
        loss = torch.tensor(0.0, device=self.device)
        
        for l in range(self.depth):
            logits = self.edge_logits[l]  # (in_width, out_width)
            in_width, out_width = logits.shape
            
            # Skip if only one destination (no balance needed)
            if out_width <= 1:
                continue
            
            # Compute soft edge weights using current temperature
            # (same as _compute_edge_weights but without Gumbel noise for stability)
            if self.max_egress is not None:
                weights = F.softmax(logits / self.temperature, dim=1)
            else:
                weights = torch.sigmoid(logits / self.temperature)
            
            # Column sums: effective number of inputs per destination
            col_sums = weights.sum(dim=0)  # (out_width,)
            
            # Fair share: total weight / num destinations
            fair_share = in_width / out_width
            
            # Penalty 1: Variance of column sums (encourage equal distribution)
            variance = ((col_sums - fair_share) ** 2).mean()
            
            # Penalty 2: Penalize any column receiving less than 1 input heavily
            # This prevents any destination from being "starved"
            min_col = col_sums.min()
            starvation_penalty = F.relu(1.0 - min_col) ** 2  # Quadratic penalty if min < 1
            
            loss = loss + variance + starvation_penalty
        
        return loss / self.depth if self.depth > 0 else loss
    
    def get_scale_regularization_loss(self) -> torch.Tensor:
        """
        Compute loss to discourage extreme scale coefficients.
        
        Penalizes scales far from 1.0 to prevent "coefficient hacking" where
        the model uses extreme scales to approximate wrong expressions.
        
        Uses log-scale penalty: (log|scale| - 0)^2 = log^2|scale|
        This penalizes both very large (>1) and very small (<1) scales equally.
        """
        loss = torch.tensor(0.0, device=self.device)
        
        for l in range(self.depth):
            scales = self.edge_scales[l]  # (in_width, out_width)
            weights = self._compute_edge_weights(l)  # selection weights
            
            # Only penalize scales on selected edges (weight > 0.1)
            # Unselected edges don't matter
            selected_mask = (weights > 0.1).float()
            
            # Log-scale penalty: penalizes scales far from 1.0 in log space
            # log(1.0) = 0, log(0.1) = -2.3, log(10) = 2.3
            log_scales = torch.log(torch.abs(scales) + 1e-8)
            scale_penalty = (log_scales ** 2) * selected_mask
            
            loss = loss + scale_penalty.mean()
        
        return loss / self.depth if self.depth > 0 else loss

    def harden(self, mode: str = "argmax") -> None:
        """
        Harden the network by converting soft edge weights to discrete selections.
        
        Args:
            mode: Hardening mode
                - "argmax": Each destination takes argmax of incoming edges
                - "argmax_row": Each source takes argmax of outgoing edges  
                - "threshold": Edges above threshold become 1, others 0
                - "hungarian": Optimal assignment via Hungarian algorithm (for square matrices)
                - "smart": Use Hungarian for square, column argmax for bottleneck, row argmax for expansion
        """
        new_hardened_edges = []

        # Derive hardened structure from the current soft logits/temperatures, not from any
        # previously hardened state. This also avoids reading from a partially built list.
        prev_hardened_edges = self.hardened_edges
        prev_is_frozen = self.is_frozen
        self.hardened_edges = None
        self.is_frozen = False

        try:
            with torch.no_grad():
                for l in range(self.depth):
                    soft_weights = self._compute_edge_weights(l)
                    in_width, out_width = soft_weights.shape
                    
                    # Determine effective mode for this layer
                    effective_mode = mode
                    if mode == "smart":
                        if in_width == out_width:
                            effective_mode = "hungarian"
                        elif in_width > out_width:
                            # Bottleneck: multiple sources combine into fewer destinations
                            # All sources should contribute - use "all" mode
                            effective_mode = "all"
                        else:
                            # Expansion: fewer sources go to more destinations
                            effective_mode = "argmax_row"
                    
                    if effective_mode == "argmax":
                        # Each column (destination) picks one source (argmax)
                        hard_weights = torch.zeros_like(soft_weights)
                        max_indices = soft_weights.argmax(dim=0)
                        for j in range(out_width):
                            hard_weights[max_indices[j], j] = 1.0
                    
                    elif effective_mode == "argmax_row":
                        # Each row (source) picks one destination (argmax)
                        hard_weights = torch.zeros_like(soft_weights)
                        max_indices = soft_weights.argmax(dim=1)
                        for i in range(in_width):
                            hard_weights[i, max_indices[i]] = 1.0
                            
                    elif effective_mode == "threshold":
                        # Edges above 0.5 become 1
                        hard_weights = (soft_weights > 0.5).float()
                    
                    elif effective_mode == "all":
                        # All edges become 1 (for bottleneck layers where all must contribute)
                        hard_weights = torch.ones_like(soft_weights)
                        
                    elif effective_mode == "hungarian":
                        # Use Hungarian algorithm for optimal assignment
                        # Only works well for square matrices
                        from scipy.optimize import linear_sum_assignment
                        cost_matrix = -soft_weights.cpu().numpy()
                        # Pad matrix if not square
                        max_dim = max(in_width, out_width)
                        padded_cost = np.zeros((max_dim, max_dim))
                        padded_cost[:in_width, :out_width] = cost_matrix
                        row_ind, col_ind = linear_sum_assignment(padded_cost)
                        hard_weights = torch.zeros_like(soft_weights)
                        for r, c in zip(row_ind, col_ind):
                            if r < in_width and c < out_width:
                                hard_weights[r, c] = 1.0
                    else:
                        raise ValueError(f"Unknown hardening mode: {mode}")
                    
                    new_hardened_edges.append(hard_weights)
        except Exception:
            self.hardened_edges = prev_hardened_edges
            self.is_frozen = prev_is_frozen
            raise

        self.hardened_edges = new_hardened_edges
        self.is_frozen = True
        logging.info(f"🔒 FullyConnectedTree hardened with mode='{mode}'")
    
    def unharden(self) -> None:
        """Revert hardening to allow continued training."""
        self.hardened_edges = None
        self.is_frozen = False
        logging.info("🔓 FullyConnectedTree unhard (soft weights restored)")
    
    def anneal_temperature(self, progress: float) -> None:
        """
        Anneal temperature from initial to final value.
        
        Args:
            progress: Training progress from 0.0 to 1.0
        """
        # Exponential annealing
        log_temp = (1 - progress) * torch.log(torch.tensor(self.temperature)) + \
                   progress * torch.log(torch.tensor(self.final_temperature))
        self.temperature = torch.exp(log_temp).item()
    
    def anneal_gumbel_noise(self, progress: float, initial_scale: float = 1.0, final_scale: float = 0.0) -> None:
        """Anneal Gumbel noise scale from initial to final value."""
        self.gumbel_noise_scale = (1 - progress) * initial_scale + progress * final_scale
    
    def get_tree_structure(self) -> dict:
        """
        Extract the learned tree structure for visualization/distillation.
        
        Returns:
            Dictionary with tree structure including:
            - layer_widths: Width at each layer
            - edges: List of edge info with selection weight and scale
            - biases: Andness values per node
        """
        structure = {
            'layer_widths': self.layer_widths,
            'depth': self.depth,
            'edges': [],
            'biases': []
        }
        
        with torch.no_grad():
            for l in range(self.depth):
                weights = self._compute_edge_weights(l)
                scales = self._get_edge_scales(l)
                bias = self.biases[l]
                
                in_width, out_width = weights.shape
                
                for i in range(in_width):
                    for j in range(out_width):
                        w = weights[i, j].item()
                        s = scales[i, j].item()
                        if w > 0.01:  # Only significant edges
                            structure['edges'].append({
                                'layer': l,
                                'src': i,
                                'dst': j,
                                'weight': w,
                                'scale': s
                            })
                
                for j in range(out_width):
                    b = bias[j].item()
                    a = torch.sigmoid(torch.tensor(b)).item() * 3 - 1
                    structure['biases'].append({
                        'layer': l + 1,
                        'node': j,
                        'andness': a
                    })
        
        return structure
    
    def get_confidence(self) -> float:
        """
        Compute confidence score based on how peaked the edge distributions are.
        
        Higher confidence = edges are closer to 0 or 1.
        """
        total_confidence = 0.0
        num_edges = 0
        
        with torch.no_grad():
            for l in range(self.depth):
                weights = self._compute_edge_weights(l)
                # Confidence: how far from 0.5 each weight is
                confidence = (2 * torch.abs(weights - 0.5)).mean().item()
                total_confidence += confidence * weights.numel()
                num_edges += weights.numel()
        
        return total_confidence / num_edges if num_edges > 0 else 0.0
    
    def __repr__(self) -> str:
        return (f"FullyConnectedTree(num_inputs={self.num_inputs}, depth={self.depth}, "
                f"layer_widths={self.layer_widths}, edge_constraint='{self.edge_constraint}', "
                f"temperature={self.temperature:.3f}, frozen={self.is_frozen})")
