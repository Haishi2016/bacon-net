import torch
import torch.nn as nn
import torch.nn.functional as F


class OperatorSetAggregator(nn.Module):
    """
    Per-node operator finder aggregator.

    - Configurable operator set:
        kind="logic",  op_names like ["and", "or"]
        kind="arith",  op_names like ["add", "sub", "mul", "div"]
    - For each internal node in BACON there is a separate learnable
      logits vector over operators.

    BACON integration:
      * binaryTreeLogicNet.__init__ (or _reinitialize) calls:
            if hasattr(self.aggregator, "attach_to_tree"):
                self.aggregator.attach_to_tree(self.num_layers)

      * binaryTreeLogicNet.forward calls:
            if hasattr(self.aggregator, "start_forward"):
                self.aggregator.start_forward()

      * Tree building keeps calling:
            self.aggregator.aggregate(left, right, a, w[0], w[1])
        as before. The aggregator internally steps a node counter.
    """

    def __init__(
        self,
        kind: str = "logic",
        op_names=None,
        use_gumbel: bool = True,
        auto_lock: bool = False,
        lock_threshold: float = 0.98,
        tau: float = 1.0,
        eps: float = 1e-6,
    ):
        super().__init__()

        assert kind in ("logic", "arith"), f"Unknown kind: {kind}"
        self.kind = kind
        self.eps = eps
        self.use_gumbel = use_gumbel
        self.tau = tau

        self.auto_lock = auto_lock
        self.lock_threshold = lock_threshold

        if op_names is None:
            if kind == "logic":
                op_names = ["and", "or"]
            else:
                op_names = ["add", "sub", "mul", "div"]

        self.op_names = list(op_names)
        self.num_ops = len(self.op_names)

        # Per-node operator logits: created later in attach_to_tree().
        self.op_logits_per_node: nn.ParameterList | None = None
        self.num_layers: int | None = None

        # Internal pointer used during a single forward pass
        self._node_ptr: int = 0

         # Lock state: one entry since we have one node
        self.register_buffer("locked", torch.tensor([False]))
        self.register_buffer("locked_idx", torch.tensor([-1], dtype=torch.long))

    # ------------------------------------------------------------------
    # Wiring from BACON
    # ------------------------------------------------------------------
    def attach_to_tree(self, num_layers: int):
        """
        Called once BACON knows how many internal nodes (layers) exist.
        Creates a set of operator logits, one per internal node.
        """
        self.num_layers = num_layers
        self.op_logits_per_node = nn.ParameterList(
            [nn.Parameter(torch.zeros(self.num_ops)) for _ in range(num_layers)]
        )

    def start_forward(self):
        """
        Called at the start of each forward() of the tree to reset node pointer.
        """
        self._node_ptr = 0

    # ------------------------------------------------------------------
    # Main aggregation API used by BACON
    # ------------------------------------------------------------------
    def aggregate(self, left, right, a, w0, w1):
        """
        Standard BACON aggregator signature.
        Uses the current node pointer to select a logits vector, applies
        softmax / Gumbel-softmax, and mixes candidate operators.
        """
        if self.op_logits_per_node is None or self.num_layers is None:
            raise RuntimeError("OperatorSetAggregator: attach_to_tree(num_layers) must be called before use.")

        if self._node_ptr >= self.num_layers:
            # Safety check; shouldn't normally happen if tree wiring is consistent.
            node_index = self.num_layers - 1
        else:
            node_index = self._node_ptr

        logits = self.op_logits_per_node[node_index]

        # Auto-lock logic
        if self.auto_lock and (not bool(self.locked[node_index])):
            max_p, max_j = logits.max(dim=0)
            if max_p.item() >= self.lock_threshold:
                # Lock this node to a single operator (no more gradient to logits)
                self.locked[node_index] = True
                self.locked_idx[node_index] = max_j
                # Optional: print once
                print(
                    f"[OperatorSetAggregator] Locking node {node_index} to "
                    f"{self.operator_names[max_j]} (p={max_p.item():.3f})"
                )

        if bool(self.locked[node_index]):
            # Use a hard one-hot, detached from logits
            j = int(self.locked_idx[node_index].item())
            hard_probs = torch.zeros_like(logits)
            hard_probs[j] = 1.0
            logits = hard_probs  # no grad to logits        

        # Move pointer for the next node
        self._node_ptr += 1



        if self.use_gumbel:
            op_w = F.gumbel_softmax(logits, tau=self.tau, hard=False, dim=0)  # [K]
        else:
            op_w = F.softmax(logits, dim=0)  # [K]

        # Build all candidate operator outputs
        cand = []
        for name in self.op_names:
            if self.kind == "logic":
                out = self._apply_logic_op(name, left, right, a, w0, w1)
            else:
                out = self._apply_arith_op(name, left, right)
            cand.append(out)

        cand = torch.stack(cand, dim=0)  # [K,B] if inputs are [B]

        out = (op_w.view(self.num_ops, 1) * cand).sum(dim=0)
        out = torch.clamp(out, 0.0, 1.0)
        return out

    # ------------------------------------------------------------------
    # Logical ops (simple min/max flavor)
    # ------------------------------------------------------------------
    def _apply_logic_op(self, name, left, right, a, w0, w1):
        """
        Very simple logical ops using min/max; you can replace by your
        full LSP versions if desired.
        """
        name = name.lower()
        if name == "and":
            return torch.min(left, right)          # strong AND
        elif name == "or":
            return torch.max(left, right)          # strong OR
        else:
            raise ValueError(f"Unknown logic op: {name}")

    # ------------------------------------------------------------------
    # Arithmetic ops
    # ------------------------------------------------------------------
    def _apply_arith_op(self, name, left, right):
        """
        Arithmetic ops; assume inputs roughly in [0,1].
        Output is squashed to [0,1].
        """
        name = name.lower()
        if name == "add":
            z = left + right             # [0,2]
            z = z / 2.0
        elif name == "sub":
            z = left - right             # [-1,1]
            z = (z + 1.0) / 2.0
        elif name == "mul":
            z = left * right             # [0,1]
        elif name == "div":
            denom = torch.abs(right) + self.eps
            raw = left / denom
            z = torch.tanh(raw)          # [-1,1]
            z = (z + 1.0) / 2.0
        else:
            raise ValueError(f"Unknown arithmetic op: {name}")
        return z
