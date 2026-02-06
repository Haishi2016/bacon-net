"""
LSP Softmax Aggregator

A differentiable aggregator that combines 5 canonical LSP operators using
softmax-weighted mixing based on the tree's andness parameter 'a':
    A0(x,y) = x*y           (product / pure t-norm)
    A1(x,y) = min(x,y)      (minimum / weak AND)
    A2(x,y) = (x+y)/2       (arithmetic mean)
    A3(x,y) = max(x,y)      (maximum / weak OR)
    A4(x,y) = x+y-x*y       (probabilistic sum / pure t-conorm)

The weights are computed via softmax based on distance from the tree's 'a' parameter
to each operator's canonical andness value (configurable, default evenly-spaced).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from bacon.aggregators.base import AggregatorBase
from typing import Sequence, Any


class LspSoftmaxAggregator(AggregatorBase, nn.Module):
    """
    LSP Softmax Aggregator using 5 canonical operators.
    
    Computes F(x,y) = sum_i w_i * A_i(x,y) where weights are determined
    by the tree's andness parameter 'a':
        w = softmax(-|a - center_i|^2 / tau)
    
    Each operator has a canonical "center" andness value (default evenly-spaced):
        A0 (product):   a = 1.5   (pure t-norm)
        A1 (min):       a = 1.0   (weak AND)
        A2 (avg):       a = 0.5   (neutral)
        A3 (max):       a = 0.0   (weak OR)
        A4 (prob_sum):  a = -0.5  (pure t-conorm)
    
    Concentration can be encouraged via:
        1. Tau annealing: decrease tau over training (use anneal_tau() or set_tau())
        2. Entropy regularization: add entropy_loss(a) to training loss to push
           tree's 'a' toward canonical centers
    
    Args:
        tau: Temperature for softmax (default: 0.5). Lower = sharper selection.
        eps: Small constant for numerical stability (default: 1e-6).
        centers: Custom operator centers (default: [1.5, 1.0, 0.5, 0.0, -0.5]).
    """
    
    OPERATOR_NAMES = ["xy", "min", "avg", "max", "x+y-xy"]
    # Default: evenly-spaced centers with 0.5 gap
    DEFAULT_CENTERS = [1.5, 1.0, 0.5, 0.0, -0.5]
    NUM_OPS = 5
    
    def __init__(self, tau: float = 0.5, eps: float = 1e-6, centers: list = None):
        # Initialize both parent classes
        AggregatorBase.__init__(self)
        nn.Module.__init__(self)
        
        self.eps = eps
        self._tau = max(tau, 1e-4)  # Clamp to avoid division by zero
        self._initial_tau = self._tau  # Store for annealing
        
        # Use provided centers or default
        if centers is None:
            centers = self.DEFAULT_CENTERS
        self._centers_list = list(centers)  # Store for describe()
        
        # Register centers as buffer (not a parameter, but moves with device)
        self.register_buffer('centers', torch.tensor(centers, dtype=torch.float32))
    
    @property
    def tau(self) -> float:
        """Current temperature."""
        return self._tau
    
    def set_tau(self, new_tau: float):
        """Update temperature (clamped to minimum 1e-4)."""
        self._tau = max(new_tau, 1e-4)
    
    def anneal_tau(self, epoch: int, max_epochs: int, 
                   final_tau: float = 0.01, schedule: str = "exponential"):
        """
        Anneal temperature based on training progress.
        
        Args:
            epoch: Current epoch (0-indexed)
            max_epochs: Total number of epochs
            final_tau: Target tau at end of training (default: 0.01)
            schedule: "exponential" or "linear" (default: "exponential")
        
        Example:
            for epoch in range(max_epochs):
                agg.anneal_tau(epoch, max_epochs, final_tau=0.01)
                train_one_epoch(...)
        """
        progress = min(epoch / max(max_epochs - 1, 1), 1.0)
        
        if schedule == "exponential":
            # tau = initial * (final/initial)^progress
            ratio = final_tau / self._initial_tau
            new_tau = self._initial_tau * (ratio ** progress)
        elif schedule == "linear":
            # tau = initial + (final - initial) * progress
            new_tau = self._initial_tau + (final_tau - self._initial_tau) * progress
        else:
            raise ValueError(f"Unknown schedule: {schedule}")
        
        self.set_tau(new_tau)
        return self._tau
    
    def get_weights_for_andness(self, a: float) -> np.ndarray:
        """Return operator weights for a given andness value."""
        with torch.no_grad():
            a_tensor = torch.tensor(a, dtype=torch.float32)
            distances = -((a_tensor - self.centers) ** 2)
            w = F.softmax(distances / self._tau, dim=0)
            return w.cpu().numpy()
    
    def _compute_weights_from_andness(self, a: torch.Tensor) -> torch.Tensor:
        """
        Compute operator weights based on andness parameter.
        
        Uses squared distance to canonical centers with softmax:
            w_i = softmax(-|a - center_i|^2 / tau)
        
        Args:
            a: Andness parameter, typically in [-1, 2]
            
        Returns:
            Weights tensor of shape [5] or [5, batch_size]
        """
        # Ensure centers on same device
        if self.centers.device != a.device:
            self.centers = self.centers.to(a.device)
        
        # Handle scalar vs batched andness
        if a.dim() == 0:
            # Scalar: compute [5] weights
            distances = -((a - self.centers) ** 2)
            return F.softmax(distances / self._tau, dim=0)
        else:
            # Batched: a is [B], compute [5, B] weights
            # centers: [5], a: [B] -> distances: [5, B]
            distances = -((a.unsqueeze(0) - self.centers.unsqueeze(1)) ** 2)
            return F.softmax(distances / self._tau, dim=0)
    
    def entropy(self, a: float = 0.5) -> float:
        """
        Compute entropy of weight distribution for a given andness.
        Higher entropy = more uniform, lower entropy = more polarized.
        """
        w = self.get_weights_for_andness(a)
        h = -np.sum(w * np.log(w + self.eps))
        return float(h)
    
    def entropy_loss(self, a: torch.Tensor) -> torch.Tensor:
        """
        Compute differentiable entropy loss for use in training.
        
        Minimizing this loss encourages the tree to learn 'a' values
        that are close to canonical centers, producing concentrated
        (low-entropy) weight distributions.
        
        Args:
            a: Andness parameter tensor from tree (requires_grad=True)
            
        Returns:
            Scalar entropy loss (add to training loss with weight lambda)
            
        Example:
            loss = task_loss + 0.1 * agg.entropy_loss(a)
        """
        w = self._compute_weights_from_andness(a)  # [5] or [5, B]
        # Entropy: -sum(w * log(w))
        log_w = torch.log(w + self.eps)
        if w.dim() == 1:
            return -(w * log_w).sum()
        else:
            # Mean over batch
            return -(w * log_w).sum(dim=0).mean()
    
    def describe(self, a: float = 0.5) -> dict:
        """Return interpretability info as dict for a given andness."""
        w = self.get_weights_for_andness(a)
        return {
            "tau": self._tau,
            "andness": a,
            "weights": w.tolist(),
            "operators": self.OPERATOR_NAMES,
            "centers": self._centers_list,
            "entropy": self.entropy(a),
            "dominant_op": self.OPERATOR_NAMES[np.argmax(w)],
            "dominant_weight": float(np.max(w))
        }
    
    def __repr__(self) -> str:
        return f"LspSoftmaxAggregator(tau={self._tau:.4f}, centers={self._centers_list})"
    
    # ------------------------------------------------------------------
    # Core operator implementations
    # ------------------------------------------------------------------
    def _compute_operators(self, x: torch.Tensor, y: torch.Tensor, 
                           w0: torch.Tensor = None, w1: torch.Tensor = None) -> torch.Tensor:
        """
        Compute all 5 operator outputs for inputs x, y.
        
        When w0/w1 are provided (for pruning support), the operators are modified:
        - If w0=1, w1=0: output should be x (ignore y)
        - If w0=0, w1=1: output should be y (ignore x)
        - Otherwise: normal operation
        
        Returns tensor of shape [5, ...] where ... matches x.shape.
        """
        # Check if pruning weights indicate bypass
        if w0 is not None and w1 is not None:
            # Detect pruning: w0=1,w1=0 means use x only; w0=0,w1=1 means use y only
            # We use a soft approach: blend between normal ops and bypass
            # When w0 >> w1, all operators should return closer to x
            # When w1 >> w0, all operators should return closer to y
            
            # Normalize weights
            w_sum = w0 + w1 + 1e-8
            w0_norm = w0 / w_sum
            w1_norm = w1 / w_sum
            
            # Check for extreme pruning (one weight is ~0)
            is_x_only = (w1 < 0.01)  # Prune y, keep x
            is_y_only = (w0 < 0.01)  # Prune x, keep y
            
            if isinstance(is_x_only, torch.Tensor):
                is_x_only = is_x_only.any().item()
            if isinstance(is_y_only, torch.Tensor):
                is_y_only = is_y_only.any().item()
            
            if is_x_only:
                # All operators should return x (pruned feature y)
                return torch.stack([x, x, x, x, x], dim=0)
            elif is_y_only:
                # All operators should return y (pruned feature x)
                return torch.stack([y, y, y, y, y], dim=0)
        
        # Normal operation: compute all 5 operators
        # A0: product (strong AND)
        a0 = x * y
        
        # A1: minimum (weak AND)
        a1 = torch.min(x, y)
        
        # A2: arithmetic mean
        a2 = (x + y) / 2.0
        
        # A3: maximum (weak OR)
        a3 = torch.max(x, y)
        
        # A4: probabilistic sum (strong OR)
        a4 = x + y - x * y
        
        # Stack along new dimension 0
        return torch.stack([a0, a1, a2, a3, a4], dim=0)
    
    def _mix_operators(self, ops: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Mix operator outputs with weights computed from andness parameter.
        
        Args:
            ops: Tensor of shape [5, ...] with operator outputs
            a: Andness parameter from tree
            
        Returns:
            Mixed output with shape [...]
        """
        # Compute weights from andness
        w = self._compute_weights_from_andness(a)  # [5] or [5, B]
        
        # Reshape weights for broadcasting to match ops dims
        if w.dim() == 1:
            # w is [5], ops is [5, ...]
            for _ in range(ops.dim() - 1):
                w = w.unsqueeze(-1)
        # else w is already [5, B] matching ops [5, B, ...]
        
        # Weighted sum
        out = (w * ops).sum(dim=0)
        
        # Clamp to [0, 1] for safety (all ops preserve range, but floating point...)
        return torch.clamp(out, 0.0, 1.0)
    
    # ------------------------------------------------------------------
    # AggregatorBase interface (used by BACON tree)
    # ------------------------------------------------------------------
    def aggregate_float(self, values: Sequence[float], a: float, weights: Sequence[float]) -> float:
        xs = torch.tensor(values, dtype=torch.float32, device=self.centers.device)
        ws = torch.tensor(weights, dtype=torch.float32, device=self.centers.device)
        a_t = torch.tensor(a, dtype=torch.float32, device=self.centers.device)
        out = self.aggregate_tensor([xi for xi in xs], a_t, ws)
        return float(out.item())

    def aggregate_tensor(self, values: Sequence[Any], a: torch.Tensor, weights: Sequence[Any] | torch.Tensor) -> torch.Tensor:
        if len(values) == 0:
            raise ValueError("aggregate_tensor: values must be non-empty")
        # Ensure centers device
        dev = values[0].device
        if self.centers.device != dev:
            self.centers = self.centers.to(dev)
        X = torch.stack(values, dim=0)  # [N, ...]
        N = X.size(0)
        eps = 1e-6
        # Optional pruning bypass: if one weight dominates, return that input
        w = None
        if weights is not None:
            if isinstance(weights, torch.Tensor):
                w = weights.to(X.device, dtype=X.dtype)
            else:
                w = torch.stack([wi if isinstance(wi, torch.Tensor) else torch.as_tensor(wi, dtype=X.dtype, device=X.device) for wi in weights])
            # Normalize for stability
            s = w.sum() + eps
            w = w / s
            max_idx = torch.argmax(w).item()
            if (w[max_idx] > 1.0 - 0.01) and torch.all(w < 0.01 + (torch.arange(N, device=w.device) == max_idx).float() * 1.0):
                return X[max_idx]
        # Compute N-ary operator outputs
        Xc = torch.clamp(X, 0.0, 1.0)
        a0 = Xc.prod(dim=0)                  # product
        a1 = torch.min(Xc, dim=0).values     # min
        a2 = Xc.mean(dim=0)                  # avg
        a3 = torch.max(Xc, dim=0).values     # max
        a4 = 1.0 - (1.0 - Xc).prod(dim=0)    # probabilistic sum
        ops = torch.stack([a0, a1, a2, a3, a4], dim=0)             # [5, ...]
        return self._mix_operators(ops, a)
    
    # -------- N-ary aggregation (new) --------
    def aggregate_many_float(self, values: Sequence[float], a: float, weights: Sequence[float]) -> float:
        xs = torch.tensor(values, dtype=torch.float32, device=self.centers.device)
        ws = torch.tensor(weights, dtype=torch.float32, device=self.centers.device)
        a_t = torch.tensor(a, dtype=torch.float32, device=self.centers.device)
        out = self.aggregate_many_tensor([xi for xi in xs], a_t, ws)
        return float(out.item())

    def aggregate_many_tensor(self, values: Sequence[Any], a: torch.Tensor, weights: Sequence[Any] | torch.Tensor) -> torch.Tensor:
        if len(values) == 0:
            raise ValueError("aggregate_many_tensor: values must be non-empty")
        # Ensure centers device
        dev = values[0].device
        if self.centers.device != dev:
            self.centers = self.centers.to(dev)
        X = torch.stack(values, dim=0)  # [N, ...]
        N = X.size(0)
        eps = 1e-6
        # Optional pruning bypass: if one weight dominates, return that input
        w = None
        if weights is not None:
            if isinstance(weights, torch.Tensor):
                w = weights.to(X.device, dtype=X.dtype)
            else:
                w = torch.stack([wi if isinstance(wi, torch.Tensor) else torch.as_tensor(wi, dtype=X.dtype, device=X.device) for wi in weights])
            # Normalize for stability
            s = w.sum() + eps
            w = w / s
            max_idx = torch.argmax(w).item()
            if (w[max_idx] > 1.0 - 0.01) and torch.all(w < 0.01 + (torch.arange(N, device=w.device) == max_idx).float() * 1.0):
                return X[max_idx]
        # Compute N-ary operator outputs
        a0 = torch.clamp(X, 0.0, 1.0).prod(dim=0)                  # product
        a1 = torch.min(torch.clamp(X, 0.0, 1.0), dim=0).values     # min
        a2 = torch.clamp(X, 0.0, 1.0).mean(dim=0)                  # avg
        a3 = torch.max(torch.clamp(X, 0.0, 1.0), dim=0).values     # max
        a4 = 1.0 - (1.0 - torch.clamp(X, 0.0, 1.0)).prod(dim=0)    # probabilistic sum
        ops = torch.stack([a0, a1, a2, a3, a4], dim=0)             # [5, ...]
        return self._mix_operators(ops, a)
    
    # ------------------------------------------------------------------
    # nn.Module forward (can be used directly)
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, y: torch.Tensor, a: torch.Tensor,
                w0: torch.Tensor = None, w1: torch.Tensor = None) -> torch.Tensor:
        """
        Compute mixed aggregation F(x,y) = sum_i w_i * A_i(x,y).
        
        Args:
            x: First input tensor, values in [0, 1]
            y: Second input tensor, values in [0, 1]
            a: Andness parameter from tree (determines operator mixing)
            w0: Weight for x (used for pruning, optional)
            w1: Weight for y (used for pruning, optional)
            
        Returns:
            Aggregated output tensor in [0, 1]
        """
        # Ensure centers on same device as inputs
        if self.centers.device != x.device:
            self.centers = self.centers.to(x.device)
        
        ops = self._compute_operators(x, y, w0, w1)
        return self._mix_operators(ops, a)


class PerNodeLspSoftmaxAggregator(nn.Module):
    """
    Per-node LSP Softmax Aggregator for use with BACON trees.
    
    Creates a separate set of operator logits for each internal node,
    similar to OperatorSetAggregator but using LSP operator basis.
    
    BACON integration:
        - Call attach_to_tree(num_layers) after tree is built
        - Call start_forward() before each forward pass
        - Call aggregate(left, right, a, w0, w1) for each node
    """
    
    OPERATOR_NAMES = ["xy", "min", "avg", "max", "x+y-xy"]
    NUM_OPS = 5
    
    def __init__(self, tau: float = 1.0, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self._tau = max(tau, 1e-4)
        
        # Per-node logits: created in attach_to_tree()
        self.op_logits_per_node: nn.ParameterList | None = None
        self.num_layers: int | None = None
        
        # Internal pointer for forward pass
        self._node_ptr: int = 0
    
    @property
    def tau(self) -> float:
        return self._tau
    
    def set_tau(self, new_tau: float):
        self._tau = max(new_tau, 1e-4)
    
    def attach_to_tree(self, num_layers: int):
        """
        Called once BACON knows how many internal nodes exist.
        Creates operator logits for each node.
        """
        self.num_layers = num_layers
        self.op_logits_per_node = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.NUM_OPS)) for _ in range(num_layers)]
        )
    
    def start_forward(self):
        """Called at start of each forward pass to reset node pointer."""
        self._node_ptr = 0
    
    def aggregate(self, left: torch.Tensor, right: torch.Tensor, 
                  a: torch.Tensor, w0: torch.Tensor, w1: torch.Tensor) -> torch.Tensor:
        """
        Standard BACON aggregator signature.
        w0, w1 are used for pruning support (bypass inputs when one weight is ~0).
        """
        if self.op_logits_per_node is None or self.num_layers is None:
            raise RuntimeError("PerNodeLspSoftmaxAggregator: attach_to_tree() must be called first")
        
        node_index = min(self._node_ptr, self.num_layers - 1)
        logits = self.op_logits_per_node[node_index]
        self._node_ptr += 1
        
        # Ensure logits on same device
        if logits.device != left.device:
            self.op_logits_per_node[node_index] = nn.Parameter(logits.to(left.device))
            logits = self.op_logits_per_node[node_index]
        
        # Compute weights
        w = F.softmax(logits / self._tau, dim=0)  # [5]
        
        # Compute all operators (with pruning support)
        ops = self._compute_operators(left, right, w0, w1)  # [5, ...]
        
        # Mix with weights
        for _ in range(ops.dim() - 1):
            w = w.unsqueeze(-1)
        out = (w * ops).sum(dim=0)
        
        return torch.clamp(out, 0.0, 1.0)
    
    def _compute_operators(self, x: torch.Tensor, y: torch.Tensor,
                           w0: torch.Tensor = None, w1: torch.Tensor = None) -> torch.Tensor:
        """Compute all 5 operator outputs with pruning support."""
        # Check if pruning weights indicate bypass
        if w0 is not None and w1 is not None:
            # Detect pruning: w0=1,w1=0 means use x only; w0=0,w1=1 means use y only
            is_x_only = (w1 < 0.01)  # Prune y, keep x
            is_y_only = (w0 < 0.01)  # Prune x, keep y
            
            if isinstance(is_x_only, torch.Tensor):
                is_x_only = is_x_only.any().item()
            if isinstance(is_y_only, torch.Tensor):
                is_y_only = is_y_only.any().item()
            
            if is_x_only:
                return torch.stack([x, x, x, x, x], dim=0)
            elif is_y_only:
                return torch.stack([y, y, y, y, y], dim=0)
        
        # Normal operation
        a0 = x * y                  # product
        a1 = torch.min(x, y)        # min
        a2 = (x + y) / 2.0          # avg
        a3 = torch.max(x, y)        # max
        a4 = x + y - x * y          # probabilistic sum
        return torch.stack([a0, a1, a2, a3, a4], dim=0)
    
    def get_weights(self, node_index: int = None) -> np.ndarray:
        """Return weights for a specific node (or all nodes if None)."""
        if self.op_logits_per_node is None:
            raise RuntimeError("attach_to_tree() must be called first")
        
        with torch.no_grad():
            if node_index is not None:
                logits = self.op_logits_per_node[node_index]
                w = F.softmax(logits / self._tau, dim=0)
                return w.cpu().numpy()
            else:
                return [
                    F.softmax(logits / self._tau, dim=0).cpu().numpy()
                    for logits in self.op_logits_per_node
                ]
    
    def get_alpha(self, node_index: int = None) -> np.ndarray:
        """Return raw logits for a specific node (or all nodes if None)."""
        if self.op_logits_per_node is None:
            raise RuntimeError("attach_to_tree() must be called first")
        
        with torch.no_grad():
            if node_index is not None:
                return self.op_logits_per_node[node_index].cpu().numpy()
            else:
                return [logits.cpu().numpy() for logits in self.op_logits_per_node]
    
    def entropy(self, node_index: int = None) -> float | list:
        """Compute entropy for a specific node (or all nodes)."""
        if self.op_logits_per_node is None:
            raise RuntimeError("attach_to_tree() must be called first")
        
        def _entropy(logits):
            w = F.softmax(logits / self._tau, dim=0)
            return -torch.sum(w * torch.log(w + self.eps)).item()
        
        with torch.no_grad():
            if node_index is not None:
                return _entropy(self.op_logits_per_node[node_index])
            else:
                return [_entropy(logits) for logits in self.op_logits_per_node]
    
    def describe(self, node_index: int = None) -> dict | list:
        """Return interpretability info."""
        if node_index is not None:
            w = self.get_weights(node_index)
            return {
                "node": node_index,
                "tau": self._tau,
                "weights": w.tolist(),
                "operators": self.OPERATOR_NAMES,
                "alpha": self.get_alpha(node_index).tolist(),
                "entropy": self.entropy(node_index),
                "dominant_op": self.OPERATOR_NAMES[np.argmax(w)],
                "dominant_weight": float(np.max(w))
            }
        else:
            return [self.describe(i) for i in range(self.num_layers)]
