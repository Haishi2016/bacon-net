"""
Alternating Coefficient-Aggregation Tree

Architecture that separates coefficient learning from structure learning:

1. Coefficient layers: Learn scalar multipliers (w*x), fixed 1:1 connections
2. Aggregation layers: Learn operators, only first layer has optional learned routing

Structure for n inputs:
  Input (n) → Coeff (n) → Agg (n-1) → Coeff (n-1) → Agg (n-2) → ... → Agg (1)

Key properties:
- Coefficients are learned in dedicated layers, not on edges
- First aggregation layer can learn routing (n→n-1) OR use external permutation layer
- Subsequent aggregation layers use fixed 2-ary pattern (node i takes inputs i, i+1)
- All edges become exactly 0 or 1 after hardening
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, List, Tuple


class CoefficientLayer(nn.Module):
    """Layer that applies learnable scalar coefficients to inputs (1:1 mapping)."""
    
    def __init__(
        self,
        width: int,
        device: torch.device,
        trainable: bool = True,
        learn_exponents: bool = False,
        min_exponent: float = 1.0,
        max_exponent: float = 2.0,
    ):
        super().__init__()
        self.width = width
        self.device = device
        self.trainable = trainable
        self.learn_exponents = learn_exponents
        self.min_exponent = float(min_exponent)
        self.max_exponent = float(max_exponent)
        if self.max_exponent < self.min_exponent:
            raise ValueError("max_exponent must be greater than or equal to min_exponent")
        # Learnable coefficients in LOG space - initialized to 0 (exp(0) = 1)
        # Using log-space prevents coefficients from going negative
        self.log_coefficients = nn.Parameter(torch.zeros(width), requires_grad=trainable)
        if self.learn_exponents:
            initial_exponent_logit = torch.full((width,), -3.0)
            self.exponent_logits = nn.Parameter(initial_exponent_logit, requires_grad=trainable)
        else:
            self.exponent_logits = None

    def _get_exponents(self) -> torch.Tensor:
        if not self.learn_exponents or self.exponent_logits is None:
            return torch.ones(self.width, device=self.device)
        exponent_span = self.max_exponent - self.min_exponent
        if exponent_span <= 0:
            return torch.full((self.width,), self.min_exponent, device=self.device)
        return self.min_exponent + exponent_span * torch.sigmoid(self.exponent_logits)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply coefficients: output[i] = exp(log_coeff[i]) * x[i]^b[i]."""
        # x shape: (batch, width)
        log_coeff_clamped = torch.clamp(self.log_coefficients, -10, 10)
        coefficients = torch.exp(log_coeff_clamped)
        if self.learn_exponents and self.exponent_logits is not None:
            exponents = self._get_exponents().unsqueeze(0)
            powered = torch.sign(x) * torch.pow(torch.clamp(x.abs(), min=1e-6), exponents)
            result = powered * coefficients.unsqueeze(0)
        else:
            result = x * coefficients.unsqueeze(0)
        return torch.clamp(result, -1e4, 1e4)
    
    def get_coefficients(self) -> torch.Tensor:
        log_coeff_clamped = torch.clamp(self.log_coefficients, -10, 10)
        return torch.exp(log_coeff_clamped).detach()

    def get_exponents(self) -> torch.Tensor:
        return self._get_exponents().detach()

    def get_exponent_regularization_loss(self) -> torch.Tensor:
        if not self.learn_exponents or self.exponent_logits is None:
            return torch.tensor(0.0, device=self.device)
        exponent_span = self.max_exponent - self.min_exponent
        if exponent_span <= 0:
            return torch.tensor(0.0, device=self.device)
        normalized = (self._get_exponents() - self.min_exponent) / exponent_span
        return (normalized * (1.0 - normalized)).mean()


class FirstAggregationLayer(nn.Module):
    """
    Aggregation layer with optional learned routing.
    
    Routes n inputs to n-1 outputs. If learn_routing=False, uses fixed pattern
    (assumes external permutation layer handles input ordering).
    
    max_egress controls routing concentration:
    - max_egress=1 + use_straight_through=True: HARD routing (exactly 1 destination, no splits)
    - max_egress=1 + use_straight_through=False: SOFT routing (softmax, may split)
    - max_egress=None: independent sigmoid per edge (most flexible)
    
    Note: Unlike operator selection, routing often doesn't converge with soft weights
    because multiple routing configurations can be mathematically equivalent.
    Recommend use_straight_through=True for discrete tree structures.
    """
    
    def __init__(
        self, 
        in_width: int, 
        out_width: int,
        learn_routing: bool = True,
        max_egress: int = 1,
        use_straight_through: bool = True,  # NEW: Force hard routing for max_egress=1
        temperature: float = 3.0,
        use_gumbel: bool = True,
        gumbel_noise_scale: float = 1.0,
        device: torch.device = None
    ):
        super().__init__()
        self.in_width = in_width
        self.out_width = out_width
        self.learn_routing = learn_routing
        self.max_egress = max_egress
        self.use_straight_through = use_straight_through
        self.temperature = temperature
        self.use_gumbel = use_gumbel
        self.gumbel_noise_scale = gumbel_noise_scale
        self.device = device or torch.device("cpu")
        
        if learn_routing:
            # Edge logits: (in_width, out_width)
            self.edge_logits = nn.Parameter(torch.zeros(in_width, out_width))
        else:
            # Fixed pattern: first input goes to node 0, remaining pair up
            # e.g., for 3→2: input 0→node0, inputs 1,2→node1
            self.register_buffer('fixed_edges', self._create_fixed_pattern())
        
        self.is_hardened = False
        self.hard_edges = None
        self.to(self.device)
    
    def _create_fixed_pattern(self) -> torch.Tensor:
        """Create fixed routing pattern when not learning."""
        # Simple pattern: node i takes inputs i and i+1 (wrapping for first)
        # For 3→2: node0 gets [0,1], node1 gets [2]
        # For 4→3: node0 gets [0,1], node1 gets [2], node2 gets [3]
        edges = torch.zeros(self.in_width, self.out_width)
        for i in range(self.in_width):
            dest = min(i, self.out_width - 1)  # Last inputs pile into last node
            edges[i, dest] = 1.0
        return edges
    
    def _sample_gumbel(self, shape: tuple, eps: float = 1e-20) -> torch.Tensor:
        U = torch.rand(shape, device=self.device)
        return -torch.log(-torch.log(U + eps) + eps)
    
    def get_edge_weights(self) -> torch.Tensor:
        if not self.learn_routing:
            return self.fixed_edges
        
        if self.is_hardened and self.hard_edges is not None:
            return self.hard_edges
        
        logits = self.edge_logits
        if self.training and self.use_gumbel and self.gumbel_noise_scale > 0:
            noise = self._sample_gumbel(logits.shape) * self.gumbel_noise_scale
            logits = logits + noise
        
        # Compute soft weights first
        if self.max_egress is not None:
            # Row-wise softmax: distribution over destinations per source
            soft_weights = F.softmax(logits / self.temperature, dim=1)
        else:
            # Independent sigmoid per edge
            soft_weights = torch.sigmoid(logits / self.temperature)
        
        # Apply straight-through if enabled (forces discrete routing during forward pass)
        if self.use_straight_through and self.max_egress is not None and self.max_egress == 1:
            # Hard one-hot selection (no gradient)
            hard_indices = soft_weights.argmax(dim=1)
            hard_weights = torch.zeros_like(soft_weights)
            hard_weights.scatter_(1, hard_indices.unsqueeze(1), 1.0)
            # Straight-through: forward uses hard, backward uses soft gradient
            weights = hard_weights - soft_weights.detach() + soft_weights
        else:
            # Use soft weights (allows splits, like operator selection)
            weights = soft_weights
        
        return weights
    
    def forward(self, x: torch.Tensor, aggregator) -> torch.Tensor:
        batch_size = x.shape[0]
        edge_weights = self.get_edge_weights()
        
        outputs = []
        for j in range(self.out_width):
            w_col = edge_weights[:, j]
            input_list = [x[:, i] for i in range(self.in_width)]
            weight_list = [w_col[i] for i in range(self.in_width)]
            
            andness = torch.tensor(0.5, device=self.device)
            node_output = aggregator.aggregate(input_list, andness, weight_list)
            outputs.append(torch.clamp(node_output, -1e4, 1e4))
        
        return torch.stack(outputs, dim=1)
    
    def harden(self) -> None:
        if not self.learn_routing:
            return
        with torch.no_grad():
            soft_weights = F.softmax(self.edge_logits / self.temperature, dim=1)
            hard = torch.zeros_like(soft_weights)
            argmax_indices = soft_weights.argmax(dim=1)
            for i in range(self.in_width):
                hard[i, argmax_indices[i]] = 1.0
            self.hard_edges = hard
            self.is_hardened = True
    
    def get_balance_loss(self) -> torch.Tensor:
        if not self.learn_routing:
            return torch.tensor(0.0, device=self.device)
        weights = self.get_edge_weights()
        col_sums = weights.sum(dim=0)
        fair_share = self.in_width / self.out_width
        variance = ((col_sums - fair_share) ** 2).mean()
        min_col = col_sums.min()
        starvation = F.relu(1.0 - min_col) ** 2
        return variance + starvation
    
    def get_egress_loss(self) -> torch.Tensor:
        """Egress loss encourages peaked routing when max_egress is set."""
        if not self.learn_routing:
            return torch.tensor(0.0, device=self.device)
        if self.max_egress is None:
            # No egress constraint when max_egress is not set
            return torch.tensor(0.0, device=self.device)
        weights = self.get_edge_weights()
        eps = 1e-8
        safe_weights = weights.clamp(min=eps)
        row_entropy = -(safe_weights * torch.log(safe_weights)).sum(dim=1)
        max_entropy = torch.log(torch.tensor(float(self.out_width), device=self.device))
        return (row_entropy / (max_entropy + eps)).mean()


class BinaryAggregationLayer(nn.Module):
    """
    Fixed binary aggregation layer (no routing to learn).
    
    Takes k inputs and produces k-1 outputs using fixed 2-ary pattern:
    - Node 0 takes inputs 0 and 1
    - Node 1 takes input 2 (identity-like, but still uses the operator)
    - ...
    - Or: Node i takes inputs i and i+1
    """
    
    def __init__(self, in_width: int, out_width: int, device: torch.device = None):
        super().__init__()
        self.in_width = in_width
        self.out_width = out_width
        self.device = device or torch.device("cpu")
        
        # Fixed binary pattern: node i takes inputs i and i+1
        # Last node gets any remaining inputs
        edges = torch.zeros(in_width, out_width)
        for i in range(in_width):
            # Each input goes to node floor(i/2) when reducing by half
            # Or: input i goes to node min(i, out_width-1)
            dest = min(i, out_width - 1)
            edges[i, dest] = 1.0
        self.register_buffer('edges', edges)
        self.to(self.device)
    
    def forward(self, x: torch.Tensor, aggregator) -> torch.Tensor:
        outputs = []
        for j in range(self.out_width):
            w_col = self.edges[:, j]
            input_list = [x[:, i] for i in range(self.in_width)]
            weight_list = [w_col[i] for i in range(self.in_width)]
            
            andness = torch.tensor(0.5, device=self.device)
            node_output = aggregator.aggregate(input_list, andness, weight_list)
            outputs.append(torch.clamp(node_output, -1e4, 1e4))
        
        return torch.stack(outputs, dim=1)


class AlternatingTree(nn.Module):
    """
    Alternating Coefficient-Aggregation Tree with learnable routing.
    
    Structure: Input(n) → Coeff(n) → Agg(n-1) → Coeff(n-1) → Agg(n-2) → ... → Agg(1)
    
    Routing (Coeff → Agg) can be learned or fixed.
    Agg → Coeff connections are always straight (1:1 mapping).
    
    Args:
        learn_first_routing: If True, learn routing in first layer (Input → Agg0).
                            If False, use external permutation layer for input ordering.
        learn_subsequent_routing: If True, learn routing in all layers after the first.
                                 If False, use fixed binary patterns.
        use_straight_through: If True and max_egress=1, use straight-through estimator
                             for hard discrete routing (recommended for tree structures).
                             If False, use soft weights like operator selection.
    """
    
    def __init__(
        self,
        num_inputs: int,
        learn_coefficients: bool = True,
        learn_first_routing: bool = True,
        learn_subsequent_routing: bool = True,
        learn_exponents: bool = False,
        min_exponent: float = 1.0,
        max_exponent: float = 2.0,
        max_egress: int = 1,
        use_straight_through: bool = True,  # NEW: control hard vs soft routing
        temperature: float = 3.0,
        final_temperature: float = 0.1,
        use_gumbel: bool = True,
        gumbel_noise_scale: float = 1.0,
        device: torch.device = None
    ):
        super().__init__()
        
        self.num_inputs = num_inputs
        self.learn_coefficients = learn_coefficients
        self.learn_first_routing = learn_first_routing
        self.learn_subsequent_routing = learn_subsequent_routing
        self.learn_exponents = learn_exponents
        self.min_exponent = min_exponent
        self.max_exponent = max_exponent
        self.max_egress = max_egress
        self.use_straight_through = use_straight_through
        self.temperature = temperature
        self.final_temperature = final_temperature
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.coeff_layers = nn.ModuleList()
        self.agg_layers = nn.ModuleList()
        
        # Build alternating layers
        current_width = num_inputs
        layer_idx = 0
        while current_width > 1:
            # Coefficient layer
            self.coeff_layers.append(
                CoefficientLayer(
                    current_width,
                    self.device,
                    trainable=learn_coefficients,
                    learn_exponents=learn_exponents,
                    min_exponent=min_exponent,
                    max_exponent=max_exponent,
                )
            )
            
            # Determine if this layer should learn routing
            if layer_idx == 0:
                should_learn = learn_first_routing
            else:
                should_learn = learn_subsequent_routing
            
            # Aggregation layer
            next_width = current_width - 1
            agg_layer = FirstAggregationLayer(
                current_width, next_width,
                learn_routing=should_learn,
                max_egress=max_egress,
                use_straight_through=use_straight_through,
                temperature=temperature,
                use_gumbel=use_gumbel,
                gumbel_noise_scale=gumbel_noise_scale,
                device=self.device
            )
            self.agg_layers.append(agg_layer)
            current_width = next_width
            layer_idx += 1

        # Single-input trees still need one coefficient stage so regression tasks
        # can fit scalar rescaling instead of degenerating to a parameter-free pass-through.
        if num_inputs == 1:
            self.coeff_layers.append(
                CoefficientLayer(
                    1,
                    self.device,
                    trainable=learn_coefficients,
                    learn_exponents=learn_exponents,
                    min_exponent=min_exponent,
                    max_exponent=max_exponent,
                )
            )
        
        self.num_agg_nodes = sum(layer.out_width for layer in self.agg_layers)
        self.is_frozen = False
        self.to(self.device)
        
        # Build routing description
        first_str = "learned" if learn_first_routing else "fixed"
        subseq_str = "learned" if learn_subsequent_routing else "fixed"
        coeff_str = "learned" if learn_coefficients else "fixed"
        exponent_str = f", exponent range: [{min_exponent:.2f}, {max_exponent:.2f}]" if learn_exponents else ""
        if learn_first_routing == learn_subsequent_routing:
            routing_str = f"all {first_str}"
        else:
            routing_str = f"first={first_str}, rest={subseq_str}"
        logging.info(f"AlternatingTree: {num_inputs} inputs, coeffs: {coeff_str}{exponent_str}, routing: {routing_str}, {len(self.agg_layers)} layers, {self.num_agg_nodes} total nodes")
    
    def forward(self, x: torch.Tensor, aggregator=None) -> torch.Tensor:
        if aggregator is None:
            raise ValueError("aggregator must be provided")
        
        if hasattr(aggregator, 'start_forward'):
            aggregator.start_forward()
        
        current = x
        coeff_idx = 0
        
        for agg_layer in self.agg_layers:
            # Apply coefficient layer
            if coeff_idx < len(self.coeff_layers):
                current = self.coeff_layers[coeff_idx](current)
                coeff_idx += 1
            # Apply aggregation
            current = agg_layer(current, aggregator)

        if coeff_idx < len(self.coeff_layers):
            current = self.coeff_layers[coeff_idx](current)
        
        return current
    
    def anneal_temperature(self, progress: float) -> None:
        """Anneal temperature for ALL aggregation layers."""
        new_temp = self.temperature - progress * (self.temperature - self.final_temperature)
        for agg_layer in self.agg_layers:
            if isinstance(agg_layer, FirstAggregationLayer):
                agg_layer.temperature = new_temp
    
    def anneal_gumbel(self, progress: float, initial: float = 1.0, final: float = 0.0) -> None:
        """Anneal Gumbel noise scale for ALL aggregation layers."""
        new_scale = initial - progress * (initial - final)
        for agg_layer in self.agg_layers:
            if isinstance(agg_layer, FirstAggregationLayer):
                agg_layer.gumbel_noise_scale = new_scale
    
    def get_balance_loss(self) -> torch.Tensor:
        """Get total balance loss across ALL aggregation layers."""
        total_loss = torch.tensor(0.0, device=self.device)
        for agg_layer in self.agg_layers:
            if isinstance(agg_layer, FirstAggregationLayer):
                total_loss = total_loss + agg_layer.get_balance_loss()
        return total_loss
    
    def get_egress_loss(self) -> torch.Tensor:
        """Get total egress loss across ALL aggregation layers."""
        total_loss = torch.tensor(0.0, device=self.device)
        for agg_layer in self.agg_layers:
            if isinstance(agg_layer, FirstAggregationLayer):
                total_loss = total_loss + agg_layer.get_egress_loss()
        return total_loss

    def get_exponent_regularization_loss(self) -> torch.Tensor:
        exponent_losses = []
        for coeff_layer in self.coeff_layers:
            if hasattr(coeff_layer, 'get_exponent_regularization_loss'):
                exponent_losses.append(coeff_layer.get_exponent_regularization_loss())
        if not exponent_losses:
            return torch.tensor(0.0, device=self.device)
        return torch.stack(exponent_losses).mean()
    
    def harden(self) -> None:
        for agg_layer in self.agg_layers:
            if hasattr(agg_layer, 'harden'):
                agg_layer.harden()
        self.is_frozen = True
    
    def get_structure_description(self) -> str:
        lines = ["\n🧠 Alternating Coefficient-Aggregation Tree:\n"]
        
        for i, coeff in enumerate(self.coeff_layers):
            coeffs = coeff.get_coefficients()
            if getattr(coeff, 'learn_exponents', False):
                exponents = coeff.get_exponents()
                coeff_str = ", ".join([f"a={c:.3f}, b={e:.3f}" for c, e in zip(coeffs, exponents)])
            else:
                coeff_str = ", ".join([f"{c:.3f}" for c in coeffs])
            lines.append(f"  Coeff Layer {i}: [{coeff_str}]")
        
        for i, agg_layer in enumerate(self.agg_layers):
            layer_type = "learned" if isinstance(agg_layer, FirstAggregationLayer) and agg_layer.learn_routing else "fixed"
            lines.append(f"  Agg Layer {i} ({layer_type}): {agg_layer.in_width} → {agg_layer.out_width}")
            edges = agg_layer.get_edge_weights() if hasattr(agg_layer, 'get_edge_weights') else agg_layer.edges
            for src in range(agg_layer.in_width):
                for dst in range(agg_layer.out_width):
                    w = edges[src, dst].item()
                    if w > 0.1:
                        lines.append(f"    {src} → {dst}: {w:.3f}")
        
        return "\n".join(lines)
