r"""AIGCD over an array of *full_weight* aggregators.

This is the AIGCD architecture (Bai & Wang, *Learning Interpretable Graded Logic
Aggregators via Adaptive Anchor Composition*) with one change: the anchor bank is
replaced by an **array of continuous-andness ``full_weight`` power means**
(:func:`lsp_power_mean`) instead of a matrix of fixed anchor operators
(min/harmonic/.../max). Each "expert" is therefore a genuine GL aggregator whose
andness is learnable and can reach hard-AND (``a=2``) / hard-OR (``a=-1``),
which the fixed anchor set cannot.

Unlike a mode switch, a **single** instance learns *which* behaviour it needs:

.. math::

    A(x \mid c) = \sum_{i=1}^{k} \alpha_i(\psi(u,c))\, F_i(u), \qquad u = R\,x

* **plain full_weight** -- routing collapses to one expert (``\alpha`` one-hot),
  gates stay at their zero-init, ``R = I``: the node is a single weighted power
  mean at that expert's andness.
* **value-based aggregation** -- the routing weights ``\alpha_i`` become a
  function of the input summary features ``\psi(u)`` (and/or external context
  ``c``), so the effective andness varies across the input space (e.g. min in one
  region, max in another).
* **partial absorption** -- the row-stochastic transform ``R`` mixes inputs so a
  composite like ``P(x,y)=H(x, A(x,y))`` emerges in a single layer.

At initialisation the gate networks emit zeros and ``R`` starts at identity, so
the aggregator computes a plain ``full_weight`` blend; training then enables only
whatever lowers the loss (``||R-I||^2`` regularisation keeps absorption off
unless it helps).

Partial absorption is intrinsically a **two-input** construction. This module
therefore engages ``R`` only to the extent that the node effectively has *two
active inputs*: the deviation of ``R`` from identity is scaled by a smooth bump
on the **participation ratio** ``n_eff = 1 / sum_j w_j^2`` of the (learned) input
weights, peaked at ``n_eff = 2``. Extra inputs with weight ~0 barely change
``n_eff``, so "two active inputs among several inactive ones" still triggers
absorption, while three-or-more active inputs switch it off.
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from bacon.aggregators.base import AggregatorBase
from bacon.aggregators.lsp.full_weight import lsp_power_mean

_PSI_DIM = 5  # psi(u) = [mean, min, max, std, prod]


def _psi(u: torch.Tensor) -> torch.Tensor:
    """Summary features of ``u`` ``[N, B]`` -> ``[B, 5]`` (mean/min/max/std/prod)."""
    f_mean = u.mean(dim=0)
    f_min = u.min(dim=0).values
    f_max = u.max(dim=0).values
    f_std = u.std(dim=0) if u.size(0) > 1 else torch.zeros_like(f_mean)
    f_prod = u.prod(dim=0)
    return torch.stack([f_mean, f_min, f_max, f_std, f_prod], dim=-1)


def _andness_to_logit(a: float) -> float:
    """Invert ``a = sigmoid(logit) * 3 - 1`` so an expert inits at andness ``a``."""
    p = min(max((a + 1.0) / 3.0, 1e-4), 1.0 - 1e-4)
    return float(math.log(p / (1.0 - p)))


class AIGCDFullWeightAggregator(nn.Module, AggregatorBase):
    r"""AIGCD over an array of ``full_weight`` power-mean experts (one node).

    Parameters
    ----------
    num_inputs : int
        Number of inputs ``N`` this node aggregates (fixed per instance).
    num_experts : int
        Number ``k`` of ``full_weight`` experts (the andness bank). Their
        andnesses are learnable; they initialise spread across ``andness_range``.
    andness_range : (float, float)
        Initial spread of the experts' andness in ``[-1, 2]``
        (``(-1, 2)`` spans hard-OR .. hard-AND).
    context_dim : int
        Width of the optional external context ``c`` (0 disables context gating).
    use_transform : bool
        Build the partial-absorption transform ``R`` (only when ``num_inputs > 1``).
        ``R`` starts at identity and is engaged only near two active inputs.
    gate_use_values : bool
        Allow routing weights to depend on ``psi(u)`` (value-based aggregation).
        The gate is zero-initialised, so this is *available* but off until trained.
    gate_use_context : bool
        Allow routing weights to depend on context ``c`` (conditional aggregation).
    hidden_dim : int
        Hidden width of the routing gate MLP.
    tau : float
        Softmax temperature for the routing weights.
    pa_sigma : float
        Width of the "two active inputs" bump on ``n_eff`` that scales ``R - I``.
    identity_reg : float
        Weight of the ``||R - I||^2`` regulariser (keeps absorption off by default).
    eps : float
        Numerical-stability clamp for the power mean.
    """

    def __init__(
        self,
        num_inputs: int,
        num_experts: int = 5,
        andness_range: tuple[float, float] = (-1.0, 2.0),
        context_dim: int = 0,
        use_transform: bool = True,
        gate_use_values: bool = True,
        gate_use_context: bool = False,
        hidden_dim: int = 16,
        tau: float = 1.0,
        pa_sigma: float = 0.5,
        identity_reg: float = 1e-3,
        eps: float = 1e-6,
    ):
        nn.Module.__init__(self)
        if num_inputs < 1:
            raise ValueError(f"num_inputs must be >= 1, got {num_inputs}")
        if num_experts < 1:
            raise ValueError(f"num_experts must be >= 1, got {num_experts}")
        if gate_use_context and context_dim <= 0:
            raise ValueError("gate_use_context=True requires context_dim > 0")

        self.num_inputs = int(num_inputs)
        self.num_experts = int(num_experts)
        self.context_dim = int(context_dim)
        self._tau = max(float(tau), 1e-4)
        self.pa_sigma = float(pa_sigma)
        self.identity_reg = float(identity_reg)
        self.eps = float(eps)
        self.gate_use_values = bool(gate_use_values)
        self.gate_use_context = bool(gate_use_context)
        # Partial absorption is a 2-input construct -> only build R for N > 1.
        self.use_transform = bool(use_transform) and self.num_inputs > 1

        # ---- expert andness bank (learnable, spread across the range) -------
        lo, hi = andness_range
        if self.num_experts == 1:
            targets = [0.5 * (lo + hi)]
        else:
            step = (hi - lo) / (self.num_experts - 1)
            targets = [lo + step * i for i in range(self.num_experts)]
        self.expert_andness_logits = nn.Parameter(
            torch.tensor([_andness_to_logit(t) for t in targets], dtype=torch.float32)
        )

        # ---- static input weights (which inputs matter; uniform at init) ----
        self.input_weight_logits = nn.Parameter(torch.zeros(self.num_inputs))

        # ---- static routing over experts (uniform at init) ------------------
        self.routing_logits = nn.Parameter(torch.zeros(self.num_experts))

        # ---- routing gate (value / context dependent; zero-init -> off) -----
        feat_dim = (_PSI_DIM if self.gate_use_values else 0) + (
            self.context_dim if self.gate_use_context else 0
        )
        self._feat_dim = feat_dim
        self.gate_net = None
        if feat_dim > 0:
            self.gate_net = nn.Sequential(
                nn.Linear(feat_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.num_experts),
            )
            nn.init.zeros_(self.gate_net[-1].weight)  # start as pure static routing
            nn.init.zeros_(self.gate_net[-1].bias)

        # ---- partial-absorption transform R (identity init) -----------------
        if self.use_transform:
            r0 = torch.zeros(self.num_inputs, self.num_inputs)
            diag = 5.0 + math.log(max(self.num_inputs - 1, 1))
            r0.fill_diagonal_(diag)  # softmax(rows) ~ identity for any N
            self.r_logits = nn.Parameter(r0)
        else:
            self.register_parameter("r_logits", None)

    # ------------------------------------------------------------------ pieces
    def input_weights(self) -> torch.Tensor:
        """Normalised per-input weights ``w`` ``[N]`` (sum to 1)."""
        return F.softmax(self.input_weight_logits, dim=0)

    def expert_andness(self) -> torch.Tensor:
        """Experts' andness ``[k]`` in ``[-1, 2]``."""
        return torch.sigmoid(self.expert_andness_logits) * 3.0 - 1.0

    def n_eff(self, w: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Effective number of active inputs = participation ratio ``1/sum(w^2)``."""
        if w is None:
            w = self.input_weights()
        return 1.0 / (w.pow(2).sum() + self.eps)

    def partial_absorption_gate(self, w: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Scalar in ``[0, 1]`` scaling ``R - I``; a bump on ``n_eff`` peaked at 2.

        ~1 when exactly two inputs are active, ->0 when one dominates or three or
        more are active. Extra inputs with weight ~0 barely move ``n_eff``, so a
        genuine active *pair* (amid inactive inputs) still triggers absorption.
        """
        if not self.use_transform:
            return torch.zeros((), device=self.input_weight_logits.device)
        ne = self.n_eff(w)
        return torch.exp(-((ne - 2.0) ** 2) / (2.0 * self.pa_sigma ** 2))

    def transform_matrix(self, w: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Effective row-stochastic transform ``R_eff = g*R + (1-g)*I`` ``[N, N]``.

        ``g`` is the two-active-inputs gate, so ``R_eff = I`` unless the node
        effectively has two active inputs. Still row-stochastic (convex mix of two
        row-stochastic matrices)."""
        if not self.use_transform or self.r_logits is None:
            n = self.num_inputs
            return torch.eye(n, device=self.input_weight_logits.device)
        R = F.softmax(self.r_logits, dim=1)
        eye = torch.eye(R.size(0), device=R.device, dtype=R.dtype)
        g = self.partial_absorption_gate(w)
        return g * R + (1.0 - g) * eye

    def _features(self, u: torch.Tensor, context: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if self._feat_dim == 0:
            return None
        B = u.size(1)
        parts = []
        if self.gate_use_values:
            parts.append(_psi(u))  # [B, 5]
        if self.gate_use_context:
            if context is None:
                context = torch.zeros(B, self.context_dim, device=u.device, dtype=u.dtype)
            if context.dim() == 1:
                context = context.unsqueeze(-1)
            parts.append(context)
        return torch.cat(parts, dim=-1)  # [B, feat_dim]

    def routing_weights(self, u: torch.Tensor,
                        context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Convex weights over experts ``alpha`` ``[k, B]`` (static + gate)."""
        logits = self.routing_logits.unsqueeze(1)              # [k, 1]
        feats = self._features(u, context)
        if self.gate_net is not None and feats is not None:
            gl = self.gate_net(feats).transpose(0, 1)          # [k, B]
            logits = logits + gl
        return F.softmax(logits / self._tau, dim=0)            # [k, B] (broadcasts)

    # ------------------------------------------------------------------ forward
    def forward(self, x, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Aggregate ``x`` (``[N, B]``, ``[N]``, or list of ``[B]``) -> ``[B]``."""
        if isinstance(x, (list, tuple)):
            X = torch.stack([xi if torch.is_tensor(xi) else torch.as_tensor(xi)
                             for xi in x], dim=0)
        else:
            X = x
        squeeze = X.dim() == 1
        if squeeze:
            X = X.unsqueeze(1)
        if X.size(0) != self.num_inputs:
            raise ValueError(
                f"expected {self.num_inputs} inputs, got {X.size(0)}")
        N, B = X.shape[0], X.shape[1]

        w = self.input_weights()                                # [N]
        # partial absorption (gated to ~2 active inputs)
        if self.use_transform:
            R = self.transform_matrix(w)                        # [N, N]
            u = R @ X                                           # [N, B]
        else:
            u = X

        # array of full_weight experts, all andnesses at once
        a = self.expert_andness()                              # [k]
        w_norm = w.view(N, 1, 1)                                # [N, 1, 1]
        Xe = u.unsqueeze(1).expand(N, self.num_experts, B)      # [N, k, B]
        ops = lsp_power_mean(Xe, a.view(self.num_experts, 1), w_norm, eps=self.eps)  # [k, B]

        alpha = self.routing_weights(u, context)               # [k, B] or [k, 1]
        out = (alpha * ops).sum(dim=0)                         # [B]
        out = out.clamp(0.0, 1.0)
        return out.squeeze(0) if squeeze else out

    # ---------------------------------------------------- AggregatorBase compat
    def aggregate_tensor(self, values, andness=None, weights=None):
        """Tensor path. ``andness``/``weights`` are ignored (owned internally)."""
        return self.forward(values)

    def aggregate_float(self, values, andness=None, weights=None):
        out = self.forward([torch.as_tensor(float(v)) for v in values])
        return float(out.item())

    # -------------------------------------------------------------- regularisers
    def transform_regularization(self) -> torch.Tensor:
        """``identity_reg * ||R - I||^2`` on the *raw* R (0 if transform off)."""
        if not self.use_transform or self.r_logits is None or self.identity_reg == 0:
            return torch.zeros((), device=self.input_weight_logits.device)
        R = F.softmax(self.r_logits, dim=1)
        eye = torch.eye(R.size(0), device=R.device, dtype=R.dtype)
        return self.identity_reg * ((R - eye) ** 2).sum()

    # -------------------------------------------------------------- diagnostics
    def effective_andness(self, u=None, context=None) -> float:
        """MAT effective andness = mean routing weight . expert andness."""
        with torch.no_grad():
            if u is None:
                u = torch.full((self.num_inputs, 1), 0.5)
            alpha = self.routing_weights(u, context)           # [k, B]
            a = self.expert_andness()                          # [k]
            return float((alpha.mean(dim=1) * a).sum())

    def describe(self, u=None, context=None) -> dict:
        with torch.no_grad():
            if u is None:
                u = torch.full((self.num_inputs, 1), 0.5)
            alpha = self.routing_weights(u, context).mean(dim=1)  # [k]
            info = {
                "num_inputs": self.num_inputs,
                "num_experts": self.num_experts,
                "input_weights": self.input_weights().tolist(),
                "expert_andness": self.expert_andness().tolist(),
                "routing_weights": alpha.tolist(),
                "dominant_expert": int(torch.argmax(alpha)),
                "effective_andness": self.effective_andness(u, context),
                "n_eff": float(self.n_eff()),
                "use_transform": self.use_transform,
                "partial_absorption_gate": float(self.partial_absorption_gate()),
            }
            if self.use_transform:
                info["R"] = self.transform_matrix().tolist()
            return info

    def __repr__(self) -> str:
        return (
            f"AIGCDFullWeightAggregator(num_inputs={self.num_inputs}, "
            f"num_experts={self.num_experts}, use_transform={self.use_transform}, "
            f"gate_use_values={self.gate_use_values}, "
            f"gate_use_context={self.gate_use_context}, tau={self._tau:.3f})"
        )
