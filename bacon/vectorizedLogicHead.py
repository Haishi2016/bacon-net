"""Vectorized batched logic head for BACON.

A standard ``binaryTreeLogicNet`` evaluates a *single* graded-logic tree and
returns one truth value of shape ``(batch, 1)``. When many independent trees
are needed over the *same* inputs (e.g. one-vs-rest classification with one
head per class), running them as a Python loop of separate modules is slow:
each head launches thousands of tiny GPU kernels.

:class:`VectorLogicHead` evaluates ``num_heads`` independent logic trees in a
single vectorized pass by carrying a leading ``head`` dimension on its
parameters and intermediate tensors. All heads share the inputs but have
independent parameters, so each head learns its own rule.

This first implementation supports the ``"left"`` (left-associated), ``"full"``
(fully-connected), and ``"alternating"`` (coefficient/aggregation) tree layouts
with the ``gl.generic`` *static* aggregator semantics: every internal node
blends a fixed library of generalized-mean anchor operators
(``min, harmonic, geometric, mean, quadratic, max``) with a learned,
per-head, softmax-normalized weight vector. This reproduces the math of a
``binaryTreeLogicNet`` whose aggregator is a static ``GenericGLAggregator``
(which ignores per-node andness/weights and depends only on its anchor
weights). Because that aggregator ignores routing weights, the ``full`` and
``alternating`` layouts reduce to closed forms that vectorize cleanly:

* ``full``        — a single N-ary GL blend over *all* inputs.
* ``alternating`` — alternating per-head coefficient scaling and N-ary GL
  blends over shrinking widths ``n, n-1, ..., 2`` (exponents are not modelled;
  they default to 1 in BACON, matching ``alternating_learn_exponents=False``).
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from bacon.aggregators.lsp.generic_gl import ANCHOR_FUNCTIONS

# The six core generalized-mean anchors from the GL paper (Section 4.5.1).
DEFAULT_ANCHORS: tuple[str, ...] = (
    "min", "harmonic", "geometric", "mean", "quadratic", "max",
)

SUPPORTED_LAYOUTS = ("left", "full", "alternating")


def _weighted_anchor(name: str, u: torch.Tensor, g: torch.Tensor,
                     eps: float) -> torch.Tensor:
    """Importance-weighted version of a GL anchor (weighted generalized mean).

    Parameters
    ----------
    name : str
        One of the six core GL anchors.
    u : torch.Tensor
        Inputs ``(width, batch, num_heads)`` in ``[0, 1]``.
    g : torch.Tensor
        Per-element relevance gates ``(width, 1, num_heads)`` in ``(0, 1)``
        broadcasting over the batch. ``g_i = 1`` means concept ``i`` is fully
        relevant; ``g_i = 0`` removes it (toward the operator's identity).
    eps : float
        Numerical-stability clamp.

    Notes
    -----
    The mean-type anchors use **normalized** gates, giving the standard
    weighted generalized means. ``min`` (conjunction) and ``max`` (disjunction)
    discount each input toward the operator's identity (``1`` for AND, ``0``
    for OR), so an irrelevant concept no longer constrains the result. As all
    gates approach ``1`` every weighted anchor reduces to its plain GL form.
    """
    us = u.clamp(eps, 1.0 - eps)
    if name == "min":
        # Discount toward the AND-identity (1): irrelevant concepts don't lower
        # the conjunction.
        return (1.0 - g * (1.0 - us)).min(dim=0).values
    if name == "max":
        # Discount toward the OR-identity (0): irrelevant concepts don't raise
        # the disjunction.
        return (g * us).max(dim=0).values

    wn = g / g.sum(dim=0, keepdim=True).clamp(min=eps)  # normalized weights
    if name == "mean":
        return (wn * us).sum(dim=0)
    if name == "harmonic":
        return 1.0 / (wn / us).sum(dim=0).clamp(min=eps)
    if name == "geometric":
        return torch.exp((wn * us.log()).sum(dim=0))
    if name == "quadratic":
        return (wn * us ** 2).sum(dim=0).clamp(min=eps * eps).sqrt()
    raise ValueError(
        f"anchor '{name}' has no weighted form; weighted aggregation supports "
        f"only the six core anchors {DEFAULT_ANCHORS}."
    )



class VectorLogicHead(nn.Module):
    """Evaluate ``num_heads`` independent graded-logic trees in one pass.

    Parameters
    ----------
    input_size : int
        Number of input features (tree leaves) per head.
    num_heads : int
        Number of independent logic trees evaluated in parallel.
    anchors : sequence of str, optional
        Anchor operator names from :data:`ANCHOR_FUNCTIONS`. Defaults to the
        six core GL anchors.
    tau : float, optional
        Softmax temperature for the per-head anchor weights.
    layout : str, optional
        Tree layout: ``"left"``, ``"full"``, or ``"alternating"``.
    use_input_weights : bool, optional
        If ``True``, give each head a learnable per-concept **relevance gate**
        ``g[h, i] = sigmoid(weight_logits[h, i]) in (0, 1)`` folded into
        *weighted* GL anchors (weighted generalized means; ``min``/``max``
        discount irrelevant concepts toward the operator identity). This is the
        GL-native, explainable way for each one-vs-rest head to focus on the
        concepts that define its class ("concept ``i`` has relevance ``g[h,i]``
        for class ``h``"); without it every head sees identical inputs and can
        only differ by the anchor blend. Per-concept importance is a single
        weighted aggregation, so this is supported for ``layout="full"`` only.
        Defaults to ``False`` (plain GL anchors; exact tree equivalence kept).
    eps : float, optional
        Numerical-stability constant passed to anchor functions and used to
        clamp outputs away from ``{0, 1}`` for BCE compatibility.

    Forward input
    -------------
    ``x`` of shape ``(batch, input_size)`` (shared across heads) or
    ``(batch, num_heads, input_size)`` (per-head inputs).

    Forward output
    --------------
    Truth values of shape ``(batch, num_heads)`` in ``(0, 1)``.
    """

    def __init__(
        self,
        input_size: int,
        num_heads: int,
        anchors: Sequence[str] = DEFAULT_ANCHORS,
        tau: float = 0.5,
        layout: str = "left",
        use_input_weights: bool = False,
        eps: float = 1e-7,
    ):
        super().__init__()
        if input_size < 1:
            raise ValueError(f"input_size must be >= 1, got {input_size}")
        if num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {num_heads}")
        if layout not in SUPPORTED_LAYOUTS:
            raise NotImplementedError(
                f"layout '{layout}' not supported yet; "
                f"available: {SUPPORTED_LAYOUTS}"
            )
        if use_input_weights:
            if layout != "full":
                raise NotImplementedError(
                    "use_input_weights=True (per-concept relevance gates) is a "
                    "single weighted aggregation and is supported for "
                    f"layout='full' only, got '{layout}'."
                )
            unsupported = [n for n in anchors if n not in DEFAULT_ANCHORS]
            if unsupported:
                raise ValueError(
                    "weighted aggregation supports only the six core anchors "
                    f"{DEFAULT_ANCHORS}; got unsupported {unsupported}."
                )
        for name in anchors:
            if name not in ANCHOR_FUNCTIONS:
                raise ValueError(
                    f"Unknown anchor '{name}'. "
                    f"Choose from: {list(ANCHOR_FUNCTIONS)}"
                )

        self.input_size = input_size
        self.num_heads = num_heads
        self.layout = layout
        self.eps = eps
        self.tau = max(float(tau), 1e-4)
        self.use_input_weights = use_input_weights

        self._anchor_names = tuple(anchors)
        self._anchor_fns = [ANCHOR_FUNCTIONS[n] for n in anchors]
        self.k = len(self._anchor_fns)

        # Independent per-head anchor weights: (num_heads, k).
        self.alpha_logits = nn.Parameter(torch.zeros(num_heads, self.k))

        # Optional per-head, per-concept relevance gates (full layout only).
        # sigmoid(0)=0.5 -> all concepts half-relevant at init; small per-head
        # noise breaks the symmetry between heads (otherwise every head is
        # identical and argmax is arbitrary, pinning task accuracy at chance).
        self.weight_logits = None
        if use_input_weights:
            self.weight_logits = nn.Parameter(
                0.01 * torch.randn(num_heads, input_size)
            )

        # The ``alternating`` layout interleaves per-head coefficient scaling
        # (one coefficient per node) with N-ary blends over shrinking widths.
        # Coefficient widths follow ``binaryTreeLogicNet``'s AlternatingTree:
        # ``[n, n-1, ..., 2]`` (one coeff layer per aggregation), or ``[1]``
        # when there is a single input. Log-space init at 0 -> coefficient 1,
        # matching BACON's default coefficient initialization.
        self.coeff_log = None
        if layout == "alternating":
            if input_size == 1:
                widths = [1]
            else:
                widths = list(range(input_size, 1, -1))  # n, n-1, ..., 2
            self.coeff_log = nn.ParameterList(
                nn.Parameter(torch.zeros(num_heads, w)) for w in widths
            )

    def anchor_weights(self) -> torch.Tensor:
        """Per-head convex anchor weights ``(num_heads, k)``."""
        return F.softmax(self.alpha_logits / self.tau, dim=-1)

    def _blend(self, u: torch.Tensor) -> torch.Tensor:
        """N-ary GL blend of ``u`` of shape ``(width, batch, num_heads)``.

        Anchor functions reduce over dim 0 (the width); the per-head anchor
        weights then convex-combine the anchor outputs. Returns
        ``(batch, num_heads)`` clamped to ``[0, 1]``.
        """
        anchors = torch.stack(
            [fn(u, self.eps) for fn in self._anchor_fns], dim=0
        )  # (k, batch, num_heads)
        # weights (num_heads, k) -> (k, 1, num_heads) for broadcasting.
        w = self.anchor_weights().transpose(0, 1).unsqueeze(1)
        out = (w * anchors).sum(dim=0)  # (batch, num_heads)
        return out.clamp(0.0, 1.0)

    def _aggregate_pair(self, acc: torch.Tensor, nxt: torch.Tensor) -> torch.Tensor:
        """GL-aggregate two ``(batch, num_heads)`` tensors elementwise per head."""
        # u: (2, batch, num_heads); anchor fns reduce over dim 0.
        return self._blend(torch.stack([acc, nxt], dim=0))

    def _weighted_blend(self, u: torch.Tensor) -> torch.Tensor:
        """Importance-weighted N-ary GL blend of ``u`` ``(width, batch, heads)``.

        Uses the per-head, per-concept relevance gates so each head aggregates
        only the concepts relevant to its class. Returns ``(batch, num_heads)``.
        """
        # gates: (width, num_heads) -> (width, 1, num_heads) to broadcast batch.
        g = torch.sigmoid(self.weight_logits).transpose(0, 1).unsqueeze(1)
        anchors = torch.stack(
            [_weighted_anchor(n, u, g, self.eps) for n in self._anchor_names],
            dim=0,
        )  # (k, batch, num_heads)
        w = self.anchor_weights().transpose(0, 1).unsqueeze(1)  # (k, 1, heads)
        out = (w * anchors).sum(dim=0)
        return out.clamp(0.0, 1.0)

    def _expand_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """Validate ``x`` and return ``(batch, num_heads, input_size)``."""
        if x.dim() == 2:
            if x.size(1) != self.input_size:
                raise ValueError(
                    f"expected input_size={self.input_size}, got {x.size(1)}"
                )
            return x.unsqueeze(1).expand(x.size(0), self.num_heads, self.input_size)
        if x.dim() == 3:
            if x.size(1) != self.num_heads or x.size(2) != self.input_size:
                raise ValueError(
                    f"expected (batch, {self.num_heads}, {self.input_size}), "
                    f"got {tuple(x.shape)}"
                )
            return x
        raise ValueError(f"x must be 2D or 3D, got {x.dim()}D")

    def _forward_left(self, xb: torch.Tensor) -> torch.Tensor:
        # Left-associated fold: acc = GL(GL(GL(x0, x1), x2), ...).
        acc = xb[..., 0]  # (batch, num_heads)
        for j in range(1, self.input_size):
            acc = self._aggregate_pair(acc, xb[..., j])
        return acc

    def _forward_full(self, xb: torch.Tensor) -> torch.Tensor:
        # Static GL ignores routing weights, so every node in the fully
        # connected tree blends all inputs identically and deeper layers are
        # idempotent -> the whole tree is one N-ary blend over all inputs.
        u = xb.permute(2, 0, 1)  # (input_size, batch, num_heads)
        if self.use_input_weights:
            return self._weighted_blend(u)
        return self._blend(u)

    def _forward_alternating(self, xb: torch.Tensor) -> torch.Tensor:
        # Alternating coefficient/aggregation tree. With static GL the routing
        # is ignored, so each aggregation layer collapses to a single N-ary
        # blend whose result is broadcast to the next width; the per-head
        # coefficient layers between aggregations re-diversify the values.
        cur = xb  # (batch, num_heads, width)
        if self.input_size == 1:
            coeff = torch.exp(self.coeff_log[0].clamp(-10.0, 10.0))  # (num_heads, 1)
            cur = (cur * coeff.unsqueeze(0)).clamp(-1e4, 1e4)
            return cur[..., 0]
        for i in range(self.input_size - 1):
            coeff = torch.exp(self.coeff_log[i].clamp(-10.0, 10.0))  # (num_heads, w)
            cur = (cur * coeff.unsqueeze(0)).clamp(-1e4, 1e4)
            o = self._blend(cur.permute(2, 0, 1)).clamp(-1e4, 1e4)  # (batch, num_heads)
            out_width = (self.input_size - i) - 1
            cur = o.unsqueeze(2).expand(o.size(0), self.num_heads, out_width)
        return cur[..., 0]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xb = self._expand_inputs(x)

        if self.layout == "left":
            acc = self._forward_left(xb)
        elif self.layout == "full":
            acc = self._forward_full(xb)
        else:  # "alternating"
            acc = self._forward_alternating(xb)

        # Clamp away from {0, 1} for BCE numerical stability (matches BACON).
        return acc.clamp(self.eps, 1.0 - self.eps)


class VectorTreeLogicHead(nn.Module):
    r"""Faithful multi-head lift of ``binaryTreeLogicNet``'s ``left`` tree.

    This evaluates ``num_heads`` independent left-associated graded-logic trees
    in a single vectorized pass, using the **same** LSP aggregator math as the
    scalar :class:`~bacon.binaryTreeLogicNet.binaryTreeLogicNet`. It is the
    direct "switch numbers to vectors" extension of BACON: every per-node scalar
    of the scalar tree -- the input routing (permutation), the per-node andness,
    and the per-node weights -- simply gains a leading ``head`` axis, and the
    identical fold/aggregator computation runs broadcast across heads.

    Unlike :class:`VectorLogicHead` (which reproduces only the ``gl.generic``
    *static anchor blend* and is therefore monotone with no real routing), this
    head calls the real aggregator (``lsp.full_weight`` / ``lsp.half_weight``)
    through its element-wise :func:`~bacon.aggregators.lsp.full_weight.lsp_power_mean`
    core, so each head has a genuine per-node andness (conjunction<->disjunction)
    and per-node weights, plus an independent soft input permutation. An optional
    per-head transformation layer (identity / negation) lets each class head use
    *negated* concepts, which a purely monotone aggregator cannot express.

    Parameters
    ----------
    input_size : int
        Number of input concepts (tree leaves) per head.
    num_heads : int
        Number of independent logic trees evaluated in parallel.
    aggregator : object
        An LSP aggregator instance exposing ``_F_many(X, a, w_norm)`` (the
        element-wise andness power mean), e.g. ``FullWeightAggregator``.
    normalize_andness : bool, optional
        If ``True`` (default) map the raw bias to andness via
        ``sigmoid(bias) * 3 - 1`` in ``(-1, 2)`` (matches BACON); else use the
        raw bias directly.
    use_permutation_layer : bool, optional
        If ``True`` (default) give every head an independent soft input
        permutation (batched Sinkhorn), mirroring ``inputToLeafSinkhorn``.
    use_transformation_layer : bool, optional
        If ``True`` give every head a per-concept identity/negation choice so it
        can use absent concepts. Defaults to ``False``.
    sinkhorn_iters, sinkhorn_temperature, use_gumbel : optional
        Sinkhorn normalization controls (match ``inputToLeafSinkhorn`` defaults).
    sinkhorn_final_temperature : float, optional
        Final Sinkhorn temperature reached at the end of annealing. BACON trains
        the soft permutation with a temperature *schedule* -- broad exploration
        first (high temperature => near-uniform routing that blends all concepts)
        then progressively sharpened toward a near-permutation (low temperature)
        as training proceeds, via :meth:`anneal`. Without annealing the routing
        stays near-uniform and every leaf collapses to ``mean(concepts)``, so the
        head becomes input-independent. Matches the full/alternating tree's
        ``final_temperature`` (default ``0.1``).
    eps : float, optional
        Clamp keeping outputs in ``(0, 1)`` for BCE stability.
    """

    def __init__(
        self,
        input_size: int,
        num_heads: int,
        aggregator,
        *,
        normalize_andness: bool = True,
        use_permutation_layer: bool = True,
        use_transformation_layer: bool = False,
        sinkhorn_iters: int = 20,
        sinkhorn_temperature: float = 3.0,
        sinkhorn_final_temperature: float = 0.1,
        transform_final_temperature: float = 0.1,
        use_gumbel: bool = True,
        eps: float = 1e-7,
    ):
        super().__init__()
        if input_size < 1:
            raise ValueError(f"input_size must be >= 1, got {input_size}")
        if num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {num_heads}")
        if not hasattr(aggregator, "_F_many"):
            raise TypeError(
                "VectorTreeLogicHead requires an LSP aggregator exposing "
                "_F_many(X, a, w_norm) (e.g. FullWeightAggregator / "
                "HalfWeightAggregator). For gl.generic use VectorLogicHead."
            )

        self.input_size = input_size
        self.num_heads = num_heads
        self.aggregator = aggregator  # stateless LSP aggregator (no params)
        self.normalize_andness = normalize_andness
        self.use_permutation_layer = use_permutation_layer
        self.use_transformation_layer = use_transformation_layer
        self.sinkhorn_iters = sinkhorn_iters
        self.sinkhorn_temperature = sinkhorn_temperature
        self.sinkhorn_final_temperature = sinkhorn_final_temperature
        self.transform_final_temperature = transform_final_temperature
        self.use_gumbel = use_gumbel
        self.eps = eps

        # Mutable exploration state driven by :meth:`anneal` (BACON schedule).
        # ``temperature`` starts broad (``sinkhorn_temperature``) and is annealed
        # toward ``sinkhorn_final_temperature``; ``gumbel_noise_scale`` starts at
        # 1.0 and is annealed toward 0.0. Registered as buffers so the *final*
        # annealed routing state is saved in ``state_dict`` and restored on load
        # -- otherwise a reloaded checkpoint would evaluate at the initial broad
        # temperature (near-uniform routing) and collapse to chance accuracy.
        self.register_buffer("temperature", torch.tensor(float(sinkhorn_temperature)))
        self.register_buffer("gumbel_noise_scale", torch.tensor(1.0))
        # Identity/negation gate sharpening temperature. Starts at 1.0 (a plain
        # softmax) and is annealed toward ``transform_final_temperature`` so the
        # gate commits to a clean identity or negation by the end of training
        # instead of settling in the diluting 50/50 middle (which squashes every
        # truth value toward 0.5). Buffered so the sharp final state is saved.
        self.register_buffer(
            "transform_temperature", torch.tensor(1.0)
        )
        # Number of left-fold aggregation nodes (N-1; 0 when there is one leaf).
        self.num_nodes = max(input_size - 1, 0)

        # Per-head, per-node andness bias and pair weights. Random init breaks
        # head symmetry (every head sees the same inputs, so distinct parameters
        # are what make the heads learn distinct class rules). Matches the scalar
        # tree's bias init ``rand * 3 - 1`` and trainable 2-vector weights.
        if self.num_nodes > 0:
            self.bias = nn.Parameter(torch.rand(num_heads, self.num_nodes) * 3 - 1)
            self.weight_logits = nn.Parameter(torch.rand(num_heads, self.num_nodes, 2))
        else:
            self.register_parameter("bias", None)
            self.register_parameter("weight_logits", None)

        # Per-head soft input permutation (batched analogue of inputToLeafSinkhorn).
        if use_permutation_layer and input_size > 1:
            self.perm_logits = nn.Parameter(torch.randn(num_heads, input_size, input_size))
            # Confidence-triggered hard freeze (matches baconNet's locked_perm):
            # once routing is confident, freeze_permutation() Hungarian-hardens the
            # soft permutation into `frozen_perm` and forward routes through it.
            self.register_buffer("perm_frozen", torch.tensor(False))
            self.register_buffer("frozen_perm",
                                 torch.zeros(num_heads, input_size, input_size))
        else:
            self.register_parameter("perm_logits", None)
            self.register_buffer("perm_frozen", torch.tensor(False))
            self.frozen_perm = None

        # Per-head per-concept transformation gate over {identity, negation}.
        # Identity is favored at init (logit +2) with small noise to break ties.
        if use_transformation_layer:
            t = torch.zeros(num_heads, input_size, 2)
            t[..., 0] = 2.0
            t = t + 0.01 * torch.randn(num_heads, input_size, 2)
            self.transform_logits = nn.Parameter(t)
        else:
            self.register_parameter("transform_logits", None)

    # -- helpers -----------------------------------------------------------
    def _andness(self, bias: torch.Tensor) -> torch.Tensor:
        """Map a per-head bias to andness, matching BACON's scalar tree."""
        if self.normalize_andness:
            return torch.sigmoid(bias) * 3.0 - 1.0
        return bias

    def anneal_temperature(self, progress: float) -> None:
        """Anneal the Sinkhorn temperature, matching BACON's tree schedule.

        ``progress`` runs from ``0.0`` (start of training) to ``1.0`` (end).
        Linearly interpolates ``temperature`` from ``sinkhorn_temperature`` down
        to ``sinkhorn_final_temperature`` (same formula as
        :meth:`AlternatingTree.anneal_temperature`): broad/soft routing first,
        sharpened toward a near-permutation as training proceeds.
        """
        p = min(max(float(progress), 0.0), 1.0)
        self.temperature.fill_(
            self.sinkhorn_temperature
            - p * (self.sinkhorn_temperature - self.sinkhorn_final_temperature)
        )
        # Sharpen the identity/negation gate on the same schedule (1.0 -> final).
        self.transform_temperature.fill_(
            1.0 - p * (1.0 - self.transform_final_temperature)
        )

    def anneal_gumbel(self, progress: float, initial: float = 1.0, final: float = 0.0) -> None:
        """Anneal the Gumbel noise scale from ``initial`` to ``final``.

        Matches :meth:`AlternatingTree.anneal_gumbel`: exploration noise is high
        early and decays to ``final`` (0 by default) so routing settles.
        """
        p = min(max(float(progress), 0.0), 1.0)
        self.gumbel_noise_scale.fill_(initial - p * (initial - final))

    def anneal(self, progress: float) -> None:
        """Advance both exploration schedules (temperature + Gumbel noise)."""
        self.anneal_temperature(progress)
        self.anneal_gumbel(progress)

    def _sinkhorn(self, logits: torch.Tensor) -> torch.Tensor:
        """Batched Sinkhorn -> doubly-stochastic ``(num_heads, N, N)`` per head.

        Uses the *current* annealed ``temperature`` and scales the exploration
        Gumbel noise by ``gumbel_noise_scale`` (both set by :meth:`anneal`),
        mirroring ``inputToLeafSinkhorn.gumbel_sinkhorn``.
        """
        temp = self.temperature.clamp(min=1e-6)
        lg = (logits / temp).clamp(-10.0, 10.0)
        if self.training and self.use_gumbel and float(self.gumbel_noise_scale) > 0.0:
            u = torch.rand_like(lg)
            g = -torch.log(-torch.log(u + 1e-20) + 1e-20)
            lg = lg + g * self.gumbel_noise_scale
        A = lg.exp()
        for _ in range(self.sinkhorn_iters):
            A = A / (A.sum(dim=2, keepdim=True) + 1e-10)  # rows
            A = A / (A.sum(dim=1, keepdim=True) + 1e-10)  # cols
        return torch.nan_to_num(A, nan=0.0, posinf=1.0, neginf=0.0)

    def _expand_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """Validate ``x`` and return ``(batch, num_heads, input_size)``."""
        if x.dim() == 2:
            if x.size(1) != self.input_size:
                raise ValueError(
                    f"expected input_size={self.input_size}, got {x.size(1)}"
                )
            return x.unsqueeze(1).expand(x.size(0), self.num_heads, self.input_size)
        if x.dim() == 3:
            if x.size(1) != self.num_heads or x.size(2) != self.input_size:
                raise ValueError(
                    f"expected (batch, {self.num_heads}, {self.input_size}), "
                    f"got {tuple(x.shape)}"
                )
            return x
        raise ValueError(f"x must be 2D or 3D, got {x.dim()}D")

    def _pair(self, left, right, a, w):
        """GL-aggregate two ``(batch, num_heads)`` tensors per head.

        ``a`` is the per-head andness ``(num_heads,)`` and ``w`` the per-head
        pair weights ``(num_heads, 2)`` (already summing to 1). Delegates to the
        real aggregator's element-wise power mean.
        """
        X = torch.stack([left, right], dim=0)          # (2, batch, num_heads)
        w_norm = w.transpose(0, 1).unsqueeze(1)         # (2, 1, num_heads)
        return self.aggregator._F_many(X, a, w_norm)    # (batch, num_heads)

    # -- forward -----------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xb = self._expand_inputs(x)  # (batch, num_heads, input_size)

        # Per-head transformation (identity / negation) over each concept.
        if self.transform_logits is not None:
            t_temp = self.transform_temperature.clamp(min=1e-6)
            tw = F.softmax(self.transform_logits / t_temp, dim=-1)  # (heads, N, 2)
            keep = tw[..., 0].unsqueeze(0)                     # (1, heads, N)
            neg = tw[..., 1].unsqueeze(0)
            xb = keep * xb + neg * (1.0 - xb)

        # Per-head soft permutation of the leaves (batched inputToLeafSinkhorn).
        if self.perm_logits is not None:
            if bool(self.perm_frozen):
                P = self.frozen_perm                           # hard, locked permutation
            else:
                P = self._sinkhorn(self.perm_logits)           # (heads, N, N)
            leaves = torch.einsum("bhn,hln->bhl", xb, P)       # reorder per head
        else:
            leaves = xb

        if self.num_nodes == 0:
            acc = leaves[..., 0]
        else:
            acc = leaves[..., 0]                               # (batch, heads)
            for i in range(self.num_nodes):
                right = leaves[..., i + 1]
                a = self._andness(self.bias[:, i])             # (heads,)
                w = F.softmax(self.weight_logits[:, i, :], dim=-1)  # (heads, 2)
                acc = self._pair(acc, right, a, w)

        return acc.clamp(self.eps, 1.0 - self.eps)

    @torch.no_grad()
    def forward_pruned(self, x: torch.Tensor, keep_mask: torch.Tensor) -> torch.Tensor:
        """Faithful structural prune-and-infer on the frozen left-fold tree.

        ``keep_mask`` is ``[num_heads, input_size]`` (bool/float) over the
        ORIGINAL concept indices: the leaves to KEEP per head. A pruned leaf is
        bypassed at the node where it enters -- its pair weight is set to 0 and
        the sibling takes weight 1 -- which is the exact GL "remove an input"
        for a 2-input power mean (``w=[1,0]`` returns the left input for ANY
        andness), matching ``binaryTreeLogicNet.prune_features``. With an
        all-ones mask this reproduces :meth:`forward` bit-for-bit.
        """
        if self.perm_logits is not None and not bool(self.perm_frozen):
            raise RuntimeError("forward_pruned requires a frozen permutation")
        xb = self._expand_inputs(x)                            # (B, H, N)
        if self.transform_logits is not None:
            t_temp = self.transform_temperature.clamp(min=1e-6)
            tw = F.softmax(self.transform_logits / t_temp, dim=-1)
            keep = tw[..., 0].unsqueeze(0)
            neg = tw[..., 1].unsqueeze(0)
            xb = keep * xb + neg * (1.0 - xb)
        km = keep_mask.to(xb.device).float()                  # [H, N] over concepts
        if tuple(km.shape) != (self.num_heads, self.input_size):
            raise ValueError(
                f"keep_mask must be [{self.num_heads}, {self.input_size}], "
                f"got {tuple(km.shape)}")
        if self.perm_logits is not None:
            P = self.frozen_perm                              # [H, N, N] hard
            leaves = torch.einsum("bhn,hln->bhl", xb, P)      # reorder per head
            leaf_keep = torch.einsum("hln,hn->hl", P, km)     # keep -> leaf positions
        else:
            leaves = xb
            leaf_keep = km
        if self.num_nodes == 0:
            return leaves[..., 0].clamp(self.eps, 1.0 - self.eps)
        acc = leaves[..., 0]                                  # (B, H)
        acc_alive = leaf_keep[:, 0]                           # (H,)
        ceps = 1e-6
        bypass = leaves.new_tensor([1.0, 0.0])
        for i in range(self.num_nodes):
            right = leaves[..., i + 1]
            right_alive = leaf_keep[:, i + 1]                 # (H,)
            a = self._andness(self.bias[:, i])                # (H,)
            w = F.softmax(self.weight_logits[:, i, :], dim=-1)  # (H, 2)
            m = torch.stack([acc_alive, right_alive], dim=1)  # (H, 2) alive gate
            wm = w * m
            s = wm.sum(dim=1, keepdim=True)                   # (H, 1)
            w_eff = torch.where(s > ceps, wm / s.clamp_min(ceps),
                                bypass.expand_as(w))          # renorm survivors
            acc = self._pair(acc, right, a, w_eff)
            acc_alive = torch.clamp(acc_alive + right_alive, max=1.0)
        return acc.clamp(self.eps, 1.0 - self.eps)
    @torch.no_grad()
    def permutation_confidence(self) -> float:
        """Mean peak routing weight per leaf (== baconNet freeze confidence)."""
        if self.perm_logits is None or bool(self.perm_frozen):
            return 1.0
        was_training = self.training
        self.eval()
        P = self._sinkhorn(self.perm_logits)
        self.train(was_training)
        return P.max(dim=2).values.mean().item()

    def permutation_sparsity_loss(self) -> torch.Tensor:
        """Mean row-entropy of the soft permutation; minimizing it drives the
        routing toward a peaked, near-hard (bijective) permutation."""
        if self.perm_logits is None or bool(self.perm_frozen):
            return self.temperature.new_zeros(())
        P = self._sinkhorn(self.perm_logits)
        H = -(P.clamp_min(1e-9) * P.clamp_min(1e-9).log()).sum(dim=2)   # (heads, N)
        return H.mean()

    @torch.no_grad()
    def freeze_permutation(self) -> None:
        """Hungarian-harden the soft permutation into a locked hard permutation
        and stop training the routing logits (== baconNet's freeze step)."""
        if self.perm_logits is None or bool(self.perm_frozen):
            return
        from scipy.optimize import linear_sum_assignment
        was_training = self.training
        self.eval()
        P = self._sinkhorn(self.perm_logits)
        self.train(was_training)
        hard = torch.zeros_like(P)
        Pc = P.detach().cpu().numpy()
        for h in range(P.shape[0]):
            rows, cols = linear_sum_assignment(-Pc[h])          # maximize routed weight
            hard[h, rows, cols] = 1.0
        self.frozen_perm.copy_(hard.to(self.frozen_perm.device))
        self.perm_frozen.fill_(True)
        self.perm_logits.requires_grad_(False)


