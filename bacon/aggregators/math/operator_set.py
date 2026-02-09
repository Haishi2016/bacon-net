import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import abstractmethod
from typing import Sequence

from bacon.aggregators.base import AggregatorBase


class OperatorSetAggregator(nn.Module, AggregatorBase):
    """
    Base class for per-node operator selection aggregators.

    For each internal node in BACON, there is a separate learnable logits 
    vector over operators. During forward pass, softmax/Gumbel-softmax is 
    applied to select operators differentiably.

    Subclasses must implement:
        - get_default_op_names(): Return default operator names for this type
        - _apply_op(name, values, a, weights): Apply the named operator

    BACON integration:
      * binaryTreeLogicNet.__init__ calls:
            if hasattr(self.aggregator, "attach_to_tree"):
                self.aggregator.attach_to_tree(self.num_layers)

      * binaryTreeLogicNet.forward calls:
            if hasattr(self.aggregator, "start_forward"):
                self.aggregator.start_forward()

      * Tree building keeps calling:
            self.aggregator.aggregate(values, a, weights)
        where values is a sequence of tensors and weights is a sequence of weights.
    """

    def __init__(
        self,
        op_names=None,
        use_gumbel: bool = True,
        tau: float = 1.0,
        eps: float = 1e-6,
        auto_harden_threshold: float | None = None,
    ):
        nn.Module.__init__(self)

        self.eps = eps
        self.use_gumbel = use_gumbel
        self.tau = tau
        self.auto_harden_threshold = auto_harden_threshold
        
        # Phase 1 default operator: if set, use this operator instead of learned blend
        # This prevents gradient explosion from div during structure discovery phase
        self.phase1_default_op: str | None = None

        if op_names is None:
            op_names = self.get_default_op_names()

        self.op_names = list(op_names)
        self.num_ops = len(self.op_names)

        # Per-node operator logits: created later in attach_to_tree().
        self.op_logits_per_node: nn.ParameterList | None = None
        self.num_layers: int | None = None

        # Internal pointer used during a single forward pass
        self._node_ptr: int = 0

    # ------------------------------------------------------------------
    # Abstract methods for subclasses
    # ------------------------------------------------------------------
    @abstractmethod
    def get_default_op_names(self) -> list:
        """Return the default list of operator names for this aggregator type."""
        pass

    @abstractmethod
    def _apply_op(self, name: str, values: Sequence[torch.Tensor], a: torch.Tensor, weights: Sequence[torch.Tensor]) -> torch.Tensor:
        """
        Apply the named operator to the input values.
        
        Args:
            name: Operator name (e.g., "add", "and")
            values: Sequence of input tensors
            a: Andness parameter
            weights: Sequence of weight tensors
            
        Returns:
            Result tensor
        """
        pass

    # ------------------------------------------------------------------
    # Wiring from BACON
    # ------------------------------------------------------------------
    def attach_to_tree(self, num_layers: int, initial_op_bias: str = None):
        """
        Called once BACON knows how many internal nodes (layers) exist.
        Creates a set of operator logits, one per internal node.
        
        If already attached with correct num_layers, does nothing (preserves existing logits).
        
        Args:
            num_layers: Number of internal nodes in the tree
            initial_op_bias: If set, initialize logits to favor this operator (e.g., "mul")
        """
        # If already attached with same size, preserve existing logits
        if self.num_layers == num_layers and self.op_logits_per_node is not None:
            return
            
        self.num_layers = num_layers
        
        if initial_op_bias and initial_op_bias in self.op_names:
            # Initialize with bias toward specified operator
            bias_idx = self.op_names.index(initial_op_bias)
            init_logits = torch.zeros(self.num_ops)
            init_logits[bias_idx] = 2.0  # Bias toward this operator
            self.op_logits_per_node = nn.ParameterList(
                [nn.Parameter(init_logits.clone()) for _ in range(num_layers)]
            )
        else:
            # Default: uniform initialization
            self.op_logits_per_node = nn.ParameterList(
                [nn.Parameter(torch.zeros(self.num_ops)) for _ in range(num_layers)]
            )

    def start_forward(self):
        """
        Called at the start of each forward() of the tree to reset node pointer.
        """
        self._node_ptr = 0

    # ------------------------------------------------------------------
    # AggregatorBase interface implementation
    # ------------------------------------------------------------------
    def aggregate_float(self, values: Sequence[float], a: float, weights: Sequence[float]) -> float:
        """Float path: convert to tensors and use tensor implementation."""
        xs = torch.tensor(values, dtype=torch.float32)
        ws = torch.tensor(weights, dtype=torch.float32)
        a_t = torch.tensor(a, dtype=torch.float32)
        out = self.aggregate_tensor([xi for xi in xs], a_t, [wi for wi in ws])
        return float(out.item())

    def aggregate_tensor(self, values: Sequence[torch.Tensor], a: torch.Tensor, weights: Sequence[torch.Tensor] | torch.Tensor) -> torch.Tensor:
        """Tensor path: main aggregation logic."""
        return self._aggregate_impl(values, a, weights)

    # ------------------------------------------------------------------
    # Main aggregation API used by BACON
    # ------------------------------------------------------------------
    def aggregate(self, values, a, weights):
        """
        Standard BACON aggregator signature.
        Dispatches to float or tensor implementation based on input type.
        
        Args:
            values: Sequence of input values (floats or tensors)
            a: Andness parameter (used for logic ops)
            weights: Sequence of weights
        """
        if not isinstance(values, (list, tuple)):
            raise TypeError("values must be a list/tuple of scalars or tensors")
        if len(values) == 0:
            raise ValueError("values must be non-empty")
        
        first = values[0]
        if isinstance(first, (float, int)):
            return self.aggregate_float(values, a, weights)
        elif isinstance(first, torch.Tensor):
            return self.aggregate_tensor(values, a, weights)
        else:
            raise TypeError("Unsupported input type for values")

    def _aggregate_impl(self, values, a, weights):
        """
        Core aggregation implementation for tensors.
        Uses the current node pointer to select a logits vector, applies
        softmax / Gumbel-softmax, and mixes candidate operators.
        """
        if self.op_logits_per_node is None or self.num_layers is None:
            raise RuntimeError(f"{self.__class__.__name__}: attach_to_tree(num_layers) must be called before use.")

        # Convert weights to list of tensors
        device = values[0].device
        dtype = values[0].dtype
        if isinstance(weights, torch.Tensor):
            w_tensors = [weights[i] for i in range(weights.shape[0])]
        else:
            w_tensors = []
            for w in weights:
                if isinstance(w, torch.Tensor):
                    w_tensors.append(w.to(device=device, dtype=dtype))
                else:
                    w_tensors.append(torch.tensor(w, device=device, dtype=dtype))

        if self._node_ptr >= self.num_layers:
            # Safety check; shouldn't normally happen if tree wiring is consistent.
            node_index = self.num_layers - 1
        else:
            node_index = self._node_ptr

        logits = self.op_logits_per_node[node_index]

        # Move pointer for the next node
        self._node_ptr += 1

        # Phase 1 default: use a single safe operator instead of learned blend
        # This prevents gradient explosion from div during structure discovery phase
        if self.phase1_default_op is not None:
            return self._apply_op(self.phase1_default_op, values, a, w_tensors)

        # Determine if we should use hard selection
        # auto_harden_threshold: if max prob exceeds threshold, use hard=True
        use_hard = False
        if self.auto_harden_threshold is not None:
            with torch.no_grad():
                # Use temperature-adjusted softmax to match gumbel_softmax behavior
                if self.use_gumbel:
                    probs = F.softmax(logits / self.tau, dim=0)
                else:
                    probs = F.softmax(logits, dim=0)
                max_prob = probs.max().item()
                use_hard = max_prob >= self.auto_harden_threshold

        if self.use_gumbel:
            op_w = F.gumbel_softmax(logits, tau=self.tau, hard=use_hard, dim=0)  # [K]
        else:
            if use_hard:
                # Hard selection via straight-through estimator
                probs = F.softmax(logits, dim=0)
                hard_sel = F.one_hot(probs.argmax(dim=0), self.num_ops).float()
                op_w = hard_sel - probs.detach() + probs  # STE
            else:
                op_w = F.softmax(logits, dim=0)  # [K]

        # Build all candidate operator outputs
        cand = []
        for name in self.op_names:
            out = self._apply_op(name, values, a, w_tensors)
            cand.append(out)

        cand = torch.stack(cand, dim=0)  # [K, B] if inputs are [B]

        # Mix operators according to learned weights
        out = (op_w.view(self.num_ops, 1) * cand).sum(dim=0)
        return out


# =============================================================================
# Boolean Operator Set
# =============================================================================

class BoolOperatorSet(OperatorSetAggregator):
    """
    Boolean/logical operator set aggregator.
    
    Default operators: ["and", "or"]
    
    Operations:
        - and: minimum of all values (strong AND)
        - or: maximum of all values (strong OR)
    """

    def __init__(
        self,
        op_names=None,
        use_gumbel: bool = True,
        tau: float = 1.0,
        eps: float = 1e-6,
        auto_harden_threshold: float | None = None,
    ):
        super().__init__(op_names=op_names, use_gumbel=use_gumbel, tau=tau, eps=eps, 
                         auto_harden_threshold=auto_harden_threshold)

    def get_default_op_names(self) -> list:
        return ["and", "or"]

    def _apply_op(self, name: str, values: Sequence[torch.Tensor], a: torch.Tensor, weights: Sequence[torch.Tensor]) -> torch.Tensor:
        """Apply boolean operation."""
        name = name.lower()
        
        if name == "and":
            # AND: minimum of all values
            result = values[0]
            for v in values[1:]:
                result = torch.min(result, v)
            return result
            
        elif name == "or":
            # OR: maximum of all values
            result = values[0]
            for v in values[1:]:
                result = torch.max(result, v)
            return result
            
        else:
            raise ValueError(f"Unknown boolean op: {name}")


# =============================================================================
# Arithmetic Operator Set
# =============================================================================

class ArithmeticOperatorSet(OperatorSetAggregator):
    """
    Arithmetic operator set aggregator with proper weighted operations.
    
    Default operators: ["add", "sub", "mul", "div"]
    
    Operations use proper weights (no normalization to [0,1]):
        - add: weighted sum = sum(w_i * x_i)
        - sub: weighted subtraction in order = w_0*x_0 - w_1*x_1 - ...
        - mul: product of weighted inputs = prod(w_i * x_i)
        - div: weighted division = (w_0*x_0) / (w_1*x_1 + eps), clamped for stability
    """

    def __init__(
        self,
        op_names=None,
        use_gumbel: bool = True,
        tau: float = 1.0,
        eps: float = 1e-6,
        auto_harden_threshold: float | None = None,
        output_clamp: float = 1e4,  # Clamp div/mul outputs to prevent explosion
    ):
        super().__init__(op_names=op_names, use_gumbel=use_gumbel, tau=tau, eps=eps, 
                         auto_harden_threshold=auto_harden_threshold)
        self.output_clamp = output_clamp

    def get_default_op_names(self) -> list:
        # return ["add", "sub", "mul", "div", "identity", "zero"]
        return ["add", "sub", "mul", "div", "identity"]

    def _apply_op(self, name: str, values: Sequence[torch.Tensor], a: torch.Tensor, weights: Sequence[torch.Tensor]) -> torch.Tensor:
        """
        Apply arithmetic operation with proper weights.
        No normalization - outputs can be any real number.
        """
        name = name.lower()
        
        if name == "add":
            # Weighted sum: sum(w_i * x_i)
            result = torch.zeros_like(values[0])
            for v, w in zip(values, weights):
                result = result + w * v
            return result
            
        elif name == "sub":
            # Weighted subtraction in order: w_0*x_0 - w_1*x_1 - w_2*x_2 - ...
            result = weights[0] * values[0]
            for v, w in zip(values[1:], weights[1:]):
                result = result - w * v
            return result
            
        elif name == "mul":
            # Product of weighted inputs: prod(w_i * x_i)
            result = weights[0] * values[0]
            for v, w in zip(values[1:], weights[1:]):
                result = result * (w * v)
            # Clamp to prevent explosion during soft operator selection
            if self.output_clamp is not None:
                result = torch.clamp(result, -self.output_clamp, self.output_clamp)
            return result
            
        elif name == "div":
            # Weighted division: (w_0*x_0) / (w_1*x_1 + eps)
            numerator = weights[0] * values[0]
            denominator = weights[1] * values[1] if len(values) > 1 else torch.ones_like(values[0])
            # Add eps to avoid division by zero
            denominator = denominator + self.eps * torch.sign(denominator + self.eps)
            result = numerator / denominator
            # Clamp to prevent explosion during soft operator selection
            if self.output_clamp is not None:
                result = torch.clamp(result, -self.output_clamp, self.output_clamp)
            return result
        
        elif name == "identity":
            # Pass through the highest-weighted input, ignore others
            # Allows nodes to "select" one input from many in deeper trees
            # Stack weights to find the max
            weight_stack = torch.stack([w.abs() if isinstance(w, torch.Tensor) else torch.tensor(abs(w)) for w in weights])
            max_idx = weight_stack.argmax().item()
            return weights[max_idx] * values[max_idx]
        
        elif name == "zero":
            # Always return zero (effectively prunes/disables this node)
            return torch.zeros_like(values[0])
            
        else:
            raise ValueError(f"Unknown arithmetic op: {name}")
