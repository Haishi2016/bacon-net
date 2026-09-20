r"""Rectangular graded-logic DAG head with *learned* (regularized) convergence.

An alternative to :class:`VectorFullTreeHead` whose triangular
funnel + egress-hardening *forces* a fixed convergent shape. Here the structure
is a plain **rectangular** stack -- ``depth`` layers each of a constant width
``W`` -- and convergence into a single-root tree is *encouraged by penalties*
rather than baked into the widths:

  1. **Single active root** -- the top layer should end up with exactly ONE
     active node (``root_lam`` penalizes ``(sum activity - 1)^2``). The scalar
     output is the activity-weighted pool of the top layer, so as the penalty
     bites the output collapses onto that one node.
  2. **Bounded fan-out** -- every node may feed at most ``max_parents`` parents
     (``N``, default 1). Edges are independent sigmoid gates ``g_ij``; the soft
     parent count ``sum_j g_ij * activity_j`` is penalized above ``N``
     (``parent_lam``).
  3. **Non-compactness penalty** -- each hidden layer is pushed to use FEW active
     nodes (``compact_lam`` penalizes the number of active nodes per layer), so
     the effective width narrows layer by layer -> a tree emerges.

Each node aggregates its (activity- and edge-gated) children with the
``full_weight`` continuous-andness power mean (:func:`lsp_power_mean`), a
per-head/per-node andness, and the same optional per-concept identity/negation
gate as the other heads. Temperatures sharpen the edge/activity gates over
training (:meth:`anneal`); :meth:`harden` commits the soft gates to a discrete,
auditable DAG.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from bacon.aggregators.lsp.full_weight import lsp_power_mean

# Hard-concrete (L0) gate constants (Louizos et al. 2018, "Learning Sparse Neural
# Networks through L0 Regularization"). A stochastic gate stretched to [gamma, zeta]
# then clamped to [0,1] so it can hit EXACTLY 0 (true pruning), with a
# differentiable expected-count penalty P(z>0).
_HC_BETA = 2.0 / 3.0
_HC_GAMMA = -0.1
_HC_ZETA = 1.1


class VectorRectTreeHead(nn.Module):
    """Rectangular graded-logic DAG head, convergence via soft regularizers.

    Parameters
    ----------
    input_size : int
        Number of input concepts (leaves), ``K``.
    num_heads : int
        Number of independent trees (e.g. classes).
    width : int
        Constant hidden-layer width ``W`` (rectangular). Defaults to ``K``.
    depth : int
        Number of width-``W`` aggregation layers stacked above the leaves.
    max_parents : int
        Fan-out budget ``N`` per node (default 1 -> tree).
    root_lam, parent_lam, compact_lam : float
        Weights of the single-root, fan-out, and non-compactness penalties.
    binarize_lam : float
        Optional entropy penalty on edge/activity gates to polarize them to
        ``{0,1}`` (helps hardening). Off by default.
    normalize_andness : bool
        Map bias -> andness via ``sigmoid(bias)*3 - 1`` in ``(-1, 2)``.
    use_negation : bool
        Add a per-head/per-concept identity-or-NOT leaf gate.
    use_coefficients : bool
        Add a smooth per-source relevance multiplier inside the child weights.
    temperature, final_temperature : float
        Gate-sharpness schedule (annealed via :meth:`anneal`).
    eps : float
        Output clamp for BCE stability.
    """

    def __init__(
        self,
        input_size: int,
        num_heads: int,
        *,
        width: Optional[int] = None,
        depth: int = 4,
        layer_widths: Optional[Sequence[int]] = None,
        max_parents: int = 1,
        root_lam: float = 1.0,
        parent_lam: float = 0.1,
        compact_lam: float = 0.05,
        binarize_lam: float = 0.0,
        edge_l1_lam: float = 0.0,
        edge_l0_lam: float = 0.0,
        straight_through: bool = False,
        leaf_shortcut: bool = False,
        normalize_andness: bool = True,
        use_negation: bool = False,
        use_coefficients: bool = False,
        use_checkpoint: bool = False,
        temperature: float = 2.0,
        final_temperature: float = 0.1,
        transform_final_temperature: float = 0.1,
        eps: float = 1e-7,
    ):
        super().__init__()
        if input_size < 1:
            raise ValueError(f"input_size must be >= 1, got {input_size}")
        if num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {num_heads}")
        if depth < 1:
            raise ValueError(f"depth must be >= 1, got {depth}")

        self.input_size = int(input_size)
        self.num_heads = int(num_heads)
        self.width = int(width) if width is not None else int(input_size)
        self.depth = int(depth)
        self.max_parents = int(max_parents)
        self.root_lam = float(root_lam)
        self.parent_lam = float(parent_lam)
        self.compact_lam = float(compact_lam)
        self.binarize_lam = float(binarize_lam)
        self.edge_l1_lam = float(edge_l1_lam)
        self.edge_l0_lam = float(edge_l0_lam)
        self.use_l0 = self.edge_l0_lam > 0.0
        self.straight_through = bool(straight_through)
        self.leaf_shortcut = bool(leaf_shortcut)
        self.normalize_andness = bool(normalize_andness)
        self.use_negation = bool(use_negation)
        self.use_coefficients = bool(use_coefficients)
        self.use_checkpoint = bool(use_checkpoint)
        self.eps = float(eps)

        # Width schedule per layer. Default is RECTANGULAR ([K, W, W, ..., W]);
        # a tapered `layer_widths` funnels toward the root -- e.g. [39, 5, 1]
        # reproduces fulltree-style even aggregation (branching ~8 over K=312).
        if layer_widths is not None:
            lw = [int(w) for w in layer_widths]
            if len(lw) < 1 or any(w < 1 for w in lw):
                raise ValueError(f"layer_widths must be >=1 ints, got {layer_widths}")
            self.depth = len(lw)
            self.width = lw[-1]
            self.widths = [self.input_size] + lw
        else:
            self.widths = [self.input_size] + [self.width] * self.depth
        # Top-layer width: the root pointer selects one of THESE nodes (== 1 for
        # a full funnel, making the pointer trivial -> a genuine tree root).
        self.wtop = self.widths[-1]
        # Per-step SOURCE dim: with leaf_shortcut, layers above the first also
        # draw children directly from the K concept leaves (skip/dense edges), so
        # a high-impact concept can reach an upper node without being diluted
        # through every aggregation. Which concepts shortcut is learned by the
        # edge gates (impact-driven) and kept sparse by the fan-out/compact loss.
        self.src_dims = [self.widths[l] + (self.input_size if (self.leaf_shortcut and l >= 1) else 0)
                         for l in range(self.depth)]

        self._temp0 = float(temperature)
        self._temp_final = float(final_temperature)
        self._transform_temp_final = float(transform_final_temperature)
        self.register_buffer("temperature", torch.tensor(float(temperature)))
        self.register_buffer("transform_temperature", torch.tensor(1.0))
        self.register_buffer("frozen", torch.tensor(False))

        # Independent per-edge presence logits (the ONLY structural parameters)
        # and a per-node andness bias. Node *presence* is DERIVED from the edges
        # (soft-OR of incoming edges), not a separate gate -- so the compactness
        # penalty genuinely prunes nodes and the hardened DAG matches training.
        self.edge_logits = nn.ParameterList()
        self.andness_bias = nn.ParameterList()
        for l in range(self.depth):
            w_in, w_out = self.src_dims[l], self.widths[l + 1]
            self.edge_logits.append(nn.Parameter(torch.randn(num_heads, w_in, w_out) * 0.1))
            self.andness_bias.append(nn.Parameter(torch.rand(num_heads, w_out) * 3 - 1))
            self.register_buffer(f"frozen_edge_{l}", torch.zeros(num_heads, w_in, w_out))
        if self.use_l0:
            # per-edge hard-concrete L0 gate location; init OPEN (log_alpha=2 ->
            # P(open)~0.97) so the tree starts fully connected and the L0 penalty
            # PRUNES edges to exactly 0 -> learned fan-out that survives harden().
            self.gate_log_alpha = nn.ParameterList(
                nn.Parameter(torch.full((num_heads, self.src_dims[l], self.widths[l + 1]), 2.0)
                             + 0.01 * torch.randn(num_heads, self.src_dims[l], self.widths[l + 1]))
                for l in range(self.depth))
        # Root pointer over the top layer's W nodes (softmax -> argmax at harden).
        self.root_select_logits = nn.Parameter(torch.randn(num_heads, self.wtop) * 0.1)
        self.register_buffer("frozen_root", torch.zeros(num_heads, self.wtop))

        if self.use_coefficients:
            self.coeff_logits = nn.ParameterList(
                nn.Parameter(torch.zeros(num_heads, self.src_dims[l])) for l in range(self.depth))

        if self.use_negation:
            t = torch.zeros(num_heads, self.input_size, 2)
            t[..., 0] = 2.0
            t = t + 0.01 * torch.randn_like(t)
            self.transform_logits = nn.Parameter(t)
            self.register_buffer("transform_frozen", torch.tensor(False))
            self.register_buffer("frozen_transform", torch.zeros(num_heads, self.input_size))

        # Regularizer terms from the most recent forward (populated in forward).
        self._reg = {}

    # ---------------------------------------------------------------- schedule
    def anneal(self, p: float) -> None:
        """Sharpen gate temperatures as training progresses (``p`` in [0, 1])."""
        p = float(max(0.0, min(1.0, p)))
        self.temperature.fill_(self._temp0 + (self._temp_final - self._temp0) * p)
        self.transform_temperature.fill_(1.0 + (self._transform_temp_final - 1.0) * p)

    # ---------------------------------------------------------------- gates
    def _edges(self, l: int) -> torch.Tensor:
        """Per-edge presence probabilities ``g in [0,1]``, shape ``[H, w_in, w_out]``."""
        if bool(self.frozen):
            return getattr(self, f"frozen_edge_{l}")
        return torch.sigmoid(self.edge_logits[l] / self.temperature.clamp_min(1e-4))

    def _activity(self, l: int) -> torch.Tensor:
        """Deprecated alias kept for compatibility: node presence of hidden layer
        ``l+1`` derived from its incoming edges, ``[H, w_out]``."""
        return self._presence(self._edges(l))

    @staticmethod
    def _presence(g: torch.Tensor) -> torch.Tensor:
        """Soft-OR of a node's incoming edges: ``1 - prod_i(1 - g_ij)``, ``[H, w_out]``.
        Exact for binary edges (present iff >=1 edge); smooth for soft gates."""
        return 1.0 - torch.prod((1.0 - g.clamp(0.0, 1.0 - 1e-6)), dim=1)

    def _l0_gate(self, l: int) -> torch.Tensor:
        """Hard-concrete gate ``z in [0,1]`` per edge: stochastic in train (sampled
        from the stretched concrete), deterministic in eval. Multiplies the edge so
        the L0 penalty can drive it to EXACTLY 0 (true prune)."""
        la = self.gate_log_alpha[l]
        if self.training:
            u = torch.rand_like(la).clamp_(1e-6, 1.0 - 1e-6)
            s = torch.sigmoid((torch.log(u) - torch.log1p(-u) + la) / _HC_BETA)
        else:
            s = torch.sigmoid(la)
        s = s * (_HC_ZETA - _HC_GAMMA) + _HC_GAMMA
        return s.clamp(0.0, 1.0)

    def _l0_open_prob(self, l: int) -> torch.Tensor:
        """Differentiable ``P(z > 0)`` per edge = the expected-open-edge L0 count."""
        return torch.sigmoid(self.gate_log_alpha[l] - _HC_BETA * math.log(-_HC_GAMMA / _HC_ZETA))

    def _l0_z_test(self, l: int) -> torch.Tensor:
        """Deterministic test-time gate value (for hardening / eval commit)."""
        s = torch.sigmoid(self.gate_log_alpha[l]) * (_HC_ZETA - _HC_GAMMA) + _HC_GAMMA
        return s.clamp(0.0, 1.0)

    def _harden_egress(self, g: torch.Tensor) -> torch.Tensor:
        """Straight-through top-``max_parents`` egress: forward uses the hard
        per-source top-N edge mask (the discrete structure), backward flows
        through the soft gates ``g``. Closes the soft->hard evaluation gap."""
        n = min(self.max_parents, g.size(2))
        hard = torch.zeros_like(g)
        idx = g.topk(n, dim=2).indices
        hard.scatter_(2, idx, 1.0)
        return hard + g - g.detach()

    def _root_pointer(self) -> torch.Tensor:
        """Softmax pointer over the top layer's ``W`` nodes, ``[H, W]`` (sums to
        1; hardens to a one-hot -> exactly one active root node)."""
        if bool(self.frozen):
            return self.frozen_root
        return F.softmax(self.root_select_logits / self.temperature.clamp_min(1e-4), dim=1)

    def _child_weights(self, g: torch.Tensor) -> torch.Tensor:
        """Normalize edge gates over sources so each destination's children sum
        to 1 (``[H, w_in, w_out]``). Orphan destinations (edge mass ~0) get a
        uniform placeholder; their node value is neutralized to 0.5 anyway via
        presence, so the placeholder never contributes information."""
        col = g.sum(dim=1, keepdim=True)                    # [H, 1, w_out]
        w = g / (col + 1e-8)
        dead = col < 1e-6
        if bool(dead.any()):
            w = torch.where(dead.expand_as(w),
                            g.new_full(w.shape, 1.0 / g.size(1)), w)
        return w                                            # [H, w_in, w_out]

    def _apply_negation(self, V: torch.Tensor) -> torch.Tensor:
        if not self.use_negation:
            return V
        if bool(self.transform_frozen):
            g = self.frozen_transform.unsqueeze(0)
        else:
            gate = F.softmax(
                self.transform_logits / self.transform_temperature.clamp_min(1e-4), dim=2)
            g = gate[..., 0].unsqueeze(0)
        return g * V + (1.0 - g) * (1.0 - V)

    # ---------------------------------------------------------------- forward
    def _layer_step(self, l: int, V: torch.Tensor, leaves: torch.Tensor):
        """One rectangular aggregation layer. Returns ``(V_out, parent_pen,
        compact_pen, bin_pen)`` -- the per-layer contributions to the structural
        penalties (zero tensors when not applicable). Factored out so the forward
        loop can wrap it in gradient checkpointing (recompute in backward)."""
        B, H = V.size(0), self.num_heads
        # source pool: previous-layer nodes, plus the K concept leaves when
        # leaf_shortcut lets a high-impact concept feed this upper layer.
        src = torch.cat([V, leaves], dim=2) if (self.leaf_shortcut and l >= 1) else V
        w_in, w_out = self.src_dims[l], self.widths[l + 1]
        g = self._edges(l)                              # [H, w_in, w_out] raw sigmoid gates
        # (0) edge-L1 sparsity on the RAW gates: unlike the soft-OR compactness
        # penalty (whose gradient vanishes once presence saturates), this gives
        # every edge a clean, non-vanishing pruning gradient -> actually narrows
        # fan-in AND node count. SUM edges per head then mean over heads so the
        # per-edge gradient is O(1/H) (independent of fan-in count); mean(g) would
        # divide by the huge edge count -> vanishing per-edge gradient. Pre-coef
        # so it targets structure, not the coefficient scale.
        l1_pen = g.sum(dim=(1, 2)).mean()
        l0_pen = g.new_zeros(())
        if self.use_l0 and not bool(self.frozen):
            # per-edge hard-concrete on/off gate: multiplies the edge (can hit 0)
            # and the L0 penalty counts expected-open edges -> learned fan-out.
            g = g * self._l0_gate(l)
            l0_pen = self._l0_open_prob(l).sum(dim=(1, 2)).mean()
        if self.use_coefficients:
            rel = torch.exp(self.coeff_logits[l].clamp(-10.0, 10.0))   # [H, w_in]
            g = g * rel.unsqueeze(2)
        # (2) fan-out & (3) compactness penalties use the SOFT gates.
        parent_pen = F.relu(g.sum(dim=2) - self.max_parents).mean()
        compact_pen = V.new_zeros(())
        bin_pen = V.new_zeros(())
        if l < self.depth - 1:
            compact_pen = self._presence(g).mean()
            if self.binarize_lam > 0:
                bin_pen = _bin_entropy(g).mean()
        # aggregation may use the straight-through HARD structure so the model
        # trains as the discrete top-N tree it will be hardened to.
        g_use = self._harden_egress(g) if self.straight_through else g
        presence = self._presence(g_use)               # [H, w_out] in [0,1]
        w_agg = self._child_weights(g_use)             # [H, w_in, w_out]

        X = src.permute(2, 0, 1).unsqueeze(-1).expand(w_in, B, H, w_out)
        Wn = w_agg.permute(1, 0, 2).unsqueeze(1)        # (w_in, 1, H, w_out)
        # per-node andness -> the SAME vectorized full-weight power mean the
        # other heads use (bias -> andness in (-1,2) via sigmoid*3-1).
        A = (torch.sigmoid(self.andness_bias[l]) * 3.0 - 1.0
             if self.normalize_andness else self.andness_bias[l]).unsqueeze(0)
        pm = lsp_power_mean(X, A, Wn, eps=1e-6)         # (B, H, w_out)
        # absent nodes (no live incoming edge) carry no information -> 0.5.
        V_out = presence.unsqueeze(0) * pm + (1.0 - presence.unsqueeze(0)) * 0.5
        return V_out, parent_pen, compact_pen, bin_pen, l1_pen, l0_pen

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            if x.size(1) != self.input_size:
                raise ValueError(f"expected input_size={self.input_size}, got {x.size(1)}")
            V = x.unsqueeze(1).expand(x.size(0), self.num_heads, self.input_size)
        elif x.dim() == 3:
            if x.size(1) != self.num_heads or x.size(2) != self.input_size:
                raise ValueError(
                    f"expected (batch, {self.num_heads}, {self.input_size}), got {tuple(x.shape)}")
            V = x
        else:
            raise ValueError(f"x must be 2D or 3D, got {x.dim()}D")

        V = self._apply_negation(V)
        B = V.size(0)
        H = self.num_heads
        leaves = V                                          # (B,H,K) for shortcut edges

        parent_pen = V.new_zeros(())
        compact_pen = V.new_zeros(())
        bin_pen = V.new_zeros(())
        l1_pen = V.new_zeros(())
        l0_pen = V.new_zeros(())
        for l in range(self.depth):
            # gradient checkpointing recomputes each layer in backward instead of
            # storing every layer's [w_in, B, H, w_out] activation -> lets deep /
            # wide rectangular stacks fit in memory (~depth x saving, +~25% compute).
            if self.use_checkpoint and self.training and torch.is_grad_enabled():
                V, pp, cp, bp, lp, l0 = checkpoint(self._layer_step, l, V, leaves,
                                           use_reentrant=False)
            else:
                V, pp, cp, bp, lp, l0 = self._layer_step(l, V, leaves)
            parent_pen = parent_pen + pp
            compact_pen = compact_pen + cp
            bin_pen = bin_pen + bp
            l1_pen = l1_pen + lp
            l0_pen = l0_pen + l0

        # (1) single active root: pointer-weighted pool of the top layer -> scalar.
        s = self._root_pointer()                            # [H, W], sums to 1
        # straight-through the pointer too: train with the HARD argmax node (what
        # harden() commits to) so hardening the pointer is a no-op -- otherwise a
        # still-soft softmax mixture collapses when snapped to one-hot at freeze.
        # BUT only hard-commit heads whose pointer already has a clear winner;
        # unsure pointers keep the SOFT mixture, else argmax flips on a near-
        # uniform softmax cause whole-eval crashes early in training. As the
        # temperature anneals, more pointers pass the threshold -> all hard.
        if self.straight_through and not bool(self.frozen):
            hard_s = F.one_hot(s.argmax(dim=1), self.wtop).float()    # [H, W]
            st = hard_s + s - s.detach()
            conf = s.max(dim=1, keepdim=True).values                  # [H, 1] pointer confidence
            s_use = torch.where(conf > 0.5, st, s)
        else:
            s_use = s
        out = (s_use.unsqueeze(0) * V).sum(dim=2)           # (B, H)
        # entropy of the (soft) pointer (normalized by log W) -> 0 when one-hot.
        ent = -(s.clamp_min(1e-9) * s.clamp_min(1e-9).log()).sum(dim=1)
        root_pen = (ent / math.log(max(self.wtop, 2))).mean()

        n_hidden = max(self.depth - 1, 1)
        self._reg = {
            "root": self.root_lam * root_pen,
            "parent": self.parent_lam * (parent_pen / self.depth),
            "compact": self.compact_lam * (compact_pen / n_hidden),
            "binarize": self.binarize_lam * (bin_pen / self.depth),
            "edge_l1": self.edge_l1_lam * (l1_pen / self.depth),
            "edge_l0": self.edge_l0_lam * (l0_pen / self.depth),
        }
        return out.clamp(self.eps, 1.0 - self.eps)

    # ---------------------------------------------------------------- penalties
    def regularization(self) -> torch.Tensor:
        """Scalar structural penalty from the most recent forward (add to loss)."""
        if not self._reg:
            return torch.zeros((), device=self.temperature.device)
        return sum(self._reg.values())

    def reg_terms(self) -> dict:
        """Individual penalty terms (floats) for logging."""
        return {k: float(v) for k, v in self._reg.items()}

    @torch.no_grad()
    def active_counts(self) -> list:
        """Per-head average number of present nodes per layer (presence > 0.5 on
        hidden layers; the root is always 1 via the pointer), for auditing."""
        hidden = [round(float((self._presence(self._edges(l)) > 0.5).float().sum().item())
                        / self.num_heads, 2)
                  for l in range(self.depth - 1)]
        return hidden + [1]

    # ---------------------------------------------------------------- hardening
    @torch.no_grad()
    def harden(self) -> None:
        """Commit the soft gates to a discrete DAG. Each source keeps its
        top-``max_parents`` outgoing edges (fan-out budget) among edges > 0.5;
        the root pointer becomes its argmax one-hot, and the selected root node
        is guaranteed >=1 incoming edge (its strongest child). Forward then reads
        the frozen buffers so evaluation matches the audited structure."""
        root_arg = self.root_select_logits.argmax(dim=1)             # [H]
        root1h = F.one_hot(root_arg, self.wtop).float()              # [H, W]
        for l in range(self.depth):
            g = self._edges(l)                              # [H, w_in, w_out]
            if self.use_l0:
                # LEARNED FAN-OUT: rank by GATED strength and drop edges whose L0
                # gate is not solidly open (z_test <= 0.5) -> a source keeps 0..N
                # parents (flexible), replacing the rigid top-N-by-rank commit.
                z_test = self._l0_z_test(l)
                g_rank = g * z_test
            else:
                z_test = None
                g_rank = g
            n = min(self.max_parents, g.size(2))
            # each source COMMITS to its top-N parents by edge strength (this is
            # the discrete structure the low-temperature soft egress approximates;
            # no 0.5 threshold, else nodes 'present' via many weak edges vanish).
            keep = torch.zeros_like(g)
            idx = g_rank.topk(n, dim=2).indices             # [H, w_in, n]
            keep.scatter_(2, idx, 1.0)
            if z_test is not None:
                keep = keep * (z_test > 0.5).float()        # learned prune of closed edges
            # absent source nodes (no live incoming edge) must not feed parents.
            # Use the ALREADY-COMMITTED frozen edges of the layer below (harden
            # runs bottom-up, so step l-1 is final here): a node is present iff it
            # kept >=1 hardened incoming edge. Using the SOFT presence instead
            # would leak phantom children (soft-present but hardened-empty nodes
            # feeding parents as neutral 0.5) -> unclean tree.
            # With leaf_shortcut, upper layers' source pool = [prev-layer nodes]
            # + [K concept leaves]; leaves are always present.
            if l == 0:
                src_present = g.new_ones(self.num_heads, self.widths[0])
            else:
                prev_present = (self._presence(getattr(self, f"frozen_edge_{l - 1}")) > 0.5).float()
                if self.leaf_shortcut:
                    leaf_present = g.new_ones(self.num_heads, self.input_size)
                    src_present = torch.cat([prev_present, leaf_present], dim=1)
                else:
                    src_present = prev_present
            keep = keep * src_present.unsqueeze(2)          # [H, w_in, w_out]
            if l == self.depth - 1:
                # only the selected root node survives; ensure it has a child.
                keep = keep * root1h.unsqueeze(1)           # [H, w_in, W] keep dests
                col = keep.sum(dim=1)                        # [H, W] incoming per dest
                orphan = (col < 1) & (root1h > 0)           # selected but childless
                if bool(orphan.any()):
                    gp = g * src_present.unsqueeze(2)       # only present sources
                    best_src = gp.argmax(dim=1)            # [H, W] strongest live child
                    h_idx, d_idx = orphan.nonzero(as_tuple=True)
                    keep[h_idx, best_src[h_idx, d_idx], d_idx] = 1.0
            getattr(self, f"frozen_edge_{l}").copy_(keep)
        self.frozen_root.copy_(root1h)
        if self.use_negation:
            # frozen_transform holds P(identity) as {0,1} (1 = identity, 0 = NOT),
            # matching _apply_negation's frozen path g*V + (1-g)*(1-V). Storing the
            # argmax INDEX here would invert it (identity is index 0 -> g=0 -> NOT).
            gate = F.softmax(self.transform_logits, dim=2)
            self.frozen_transform.copy_((gate[..., 0] >= 0.5).float())
            self.transform_frozen.fill_(True)
        self.frozen.fill_(True)


def _bin_entropy(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Binary entropy of gate probabilities (minimized -> polarizes to 0/1)."""
    p = p.clamp(eps, 1.0 - eps)
    return -(p * p.log() + (1.0 - p) * (1.0 - p).log())
