r"""Generic Full-Weight aggregator.

Combines the two BACON aggregators into one *generic* form whose behaviour is
**selected by training** rather than fixed a priori:

  * ``full_weight`` core  -> a **continuous** andness-parameterised weighted
    power mean ``lsp_power_mean(u, a, w)`` with ``a in [-1, 2]`` (reaches
    hard-AND / hard-OR, unlike the anchor-blend of ``gl.generic``).
  * ``gl.generic`` capabilities layered on top:
      - **partial absorption** via a learnable row-stochastic coordinate
        transform ``u = R x`` (identity at init);
      - **value-based** aggregation: andness ``a`` and per-input weights ``w``
        are modulated by summary features ``psi(u) = [mean, min, max, std, prod]``;
      - **conditional** aggregation: ``a`` / ``w`` also modulated by an external
        context vector ``c``.

Everything is a *superset* of ``full_weight``: the gates initialise to zero and
``R`` initialises to identity, so at init the aggregator computes exactly
``full_weight`` (static andness + static weights). Training then turns on
whatever it needs -- if the value/context gates stay near zero it remains
"regular"; if ``R`` moves off identity it discovers partial absorption; etc.

The reduction core is the same vectorizable ``lsp_power_mean``, so this drops
into the vectorized full tree (per-head / per-node ``a`` and ``w`` tensors).
"""

from __future__ import annotations

import math
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from bacon.aggregators.base import AggregatorBase
from bacon.aggregators.lsp.full_weight import lsp_power_mean

_PSI_DIM = 5  # [mean, min, max, std, prod]


def _psi(u: torch.Tensor) -> torch.Tensor:
    """Summary-statistic features of ``u`` ``[N, batch]`` -> ``[batch, 5]``."""
    f_mean = u.mean(dim=0)
    f_min = u.min(dim=0).values
    f_max = u.max(dim=0).values
    f_std = u.std(dim=0) if u.size(0) > 1 else torch.zeros_like(f_mean)
    f_prod = u.prod(dim=0)
    return torch.stack([f_mean, f_min, f_max, f_std, f_prod], dim=-1)


class GenericFullWeightAggregator(nn.Module, AggregatorBase):
    r"""One aggregation node over ``num_inputs`` inputs, generic GL form.

    Parameters
    ----------
    num_inputs : int
        Number of inputs ``N`` this node aggregates (fixed per instance).
    weight_mode : str
        ``'static'``        -> plain ``full_weight`` (no gates).
        ``'value_dependent'`` -> gates conditioned on ``psi(u)``.
        ``'conditional'``   -> gates conditioned on context ``c``.
        ``'full'``          -> gates conditioned on both.
        ``'generic'``       -> ``'full'`` **and** ``use_transform=True``
                               (partial absorption on): the "everything,
                               training-selects" mode.
    use_transform : bool or None
        Enable the row-stochastic coordinate transform ``R`` (partial
        absorption). ``None`` -> True iff ``weight_mode == 'generic'``.
    gate_andness, gate_weights : bool
        Whether the dynamic gates modulate andness / per-input weights.
    context_dim : int
        Width of the external context ``c`` (0 if unused).
    hidden_dim : int
        Hidden width of the gate MLPs.
    init_andness : float
        Static andness base in ``[-1, 2]`` (0.5 = neutral mean).
    identity_reg : float
        Weight of the ``||R - I||^2`` regulariser (keeps absorption off unless
        it helps).
    eps : float
        Clamp for ``lsp_power_mean``.
    """

    def __init__(
        self,
        num_inputs: int,
        weight_mode: str = "generic",
        use_transform: Optional[bool] = None,
        gate_andness: bool = True,
        gate_weights: bool = True,
        context_dim: int = 0,
        hidden_dim: int = 16,
        init_andness: float = 0.5,
        identity_reg: float = 1e-3,
        eps: float = 1e-6,
    ):
        nn.Module.__init__(self)
        if num_inputs < 1:
            raise ValueError(f"num_inputs must be >= 1, got {num_inputs}")
        valid = {"static", "value_dependent", "conditional", "full", "generic"}
        if weight_mode not in valid:
            raise ValueError(f"weight_mode must be one of {valid}, got {weight_mode!r}")

        self.num_inputs = int(num_inputs)
        self.weight_mode = weight_mode
        self.context_dim = int(context_dim)
        self.identity_reg = float(identity_reg)
        self.eps = float(eps)

        self.use_value = weight_mode in ("value_dependent", "full", "generic")
        self.use_context = weight_mode in ("conditional", "full", "generic")
        if use_transform is None:
            use_transform = (weight_mode == "generic")
        self.use_transform = bool(use_transform)
        self.gate_andness = bool(gate_andness)
        self.gate_weights = bool(gate_weights)

        # --- static parameters (this reproduces full_weight on its own) ------
        if not (-1.0 <= init_andness <= 2.0):
            raise ValueError("init_andness must be in [-1, 2]")
        # invert a = sigmoid(logit) * 3 - 1  =>  logit = logit_of((a + 1) / 3)
        p = min(max((init_andness + 1.0) / 3.0, 1e-4), 1 - 1e-4)
        self.andness_logit = nn.Parameter(torch.tensor(float(torch.logit(torch.tensor(p)))))
        self.weight_logits = nn.Parameter(torch.zeros(self.num_inputs))

        # --- partial absorption: row-stochastic R (identity init) -----------
        if self.use_transform and self.num_inputs > 1:
            r0 = torch.zeros(self.num_inputs, self.num_inputs)
            # N-independent near-identity: softmax(diag)/off = e^5 : 1 per off
            # cell -> diagonal weight ~0.993 for any N (learnable, reg pulls to I).
            diag = 5.0 + math.log(max(self.num_inputs - 1, 1))
            r0.fill_diagonal_(diag)
            self.r_logits = nn.Parameter(r0)
        else:
            self.register_parameter("r_logits", None)

        # --- dynamic gates (init to zero output -> start as static) ---------
        feat_dim = (_PSI_DIM if self.use_value else 0) + (
            self.context_dim if self.use_context else 0
        )
        self._feat_dim = feat_dim
        self.andness_gate = None
        self.weight_gate = None
        if feat_dim > 0 and self.gate_andness:
            self.andness_gate = self._make_gate(feat_dim, hidden_dim, 1)
        if feat_dim > 0 and self.gate_weights:
            self.weight_gate = self._make_gate(feat_dim, hidden_dim, self.num_inputs)

    @staticmethod
    def _make_gate(feat_dim: int, hidden_dim: int, out_dim: int) -> nn.Sequential:
        net = nn.Sequential(
            nn.Linear(feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim),
        )
        nn.init.zeros_(net[-1].weight)  # zero output at init -> pure static core
        nn.init.zeros_(net[-1].bias)
        return net

    # ------------------------------------------------------------------ core
    def _transform(self, X: torch.Tensor) -> torch.Tensor:
        if self.use_transform and self.r_logits is not None:
            R = F.softmax(self.r_logits, dim=1)  # row-stochastic [N, N]
            return R @ X
        return X

    def _features(self, u: torch.Tensor, context: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
        if self._feat_dim == 0:
            return None
        B = u.size(1)
        parts = []
        if self.use_value:
            parts.append(_psi(u))  # [B, 5]
        if self.use_context:
            if context is None:
                context = torch.zeros(B, self.context_dim, device=u.device, dtype=u.dtype)
            if context.dim() == 1:
                context = context.unsqueeze(-1)
            parts.append(context)
        return torch.cat(parts, dim=-1)  # [B, feat_dim]

    def effective_andness_weights(self, X: torch.Tensor, context=None):
        """Return ``(u, a, w)`` used by the power mean (for tests / interpretability).

        ``u`` is the (optionally absorbed) inputs ``[N, batch]``; ``a`` is the
        andness (scalar or ``[batch]``); ``w`` the per-input weights ``[N, batch]``.
        """
        if X.dim() == 1:
            X = X.unsqueeze(1)
        N, B = X.shape
        u = self._transform(X)
        a_logit = self.andness_logit
        w_logits = self.weight_logits.unsqueeze(1).expand(N, B).clone()  # [N, B]

        feats = self._features(u, context)
        if feats is not None:
            if self.andness_gate is not None:
                a_logit = a_logit + self.andness_gate(feats).squeeze(-1)  # [B]
            if self.weight_gate is not None:
                w_logits = w_logits + self.weight_gate(feats).transpose(0, 1)  # [N, B]

        a = torch.sigmoid(a_logit) * 3.0 - 1.0            # [-1, 2], scalar or [B]
        w = F.softmax(w_logits, dim=0)                    # [N, B], sums to 1 over N
        return u, a, w

    def forward(self, x, context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Aggregate ``x`` (``[N, batch]`` or list of ``[batch]``) -> ``[batch]``."""
        if isinstance(x, (list, tuple)):
            X = torch.stack(list(x), dim=0)
        else:
            X = x
        squeeze = X.dim() == 1
        if squeeze:
            X = X.unsqueeze(1)
        u, a, w = self.effective_andness_weights(X, context)
        out = lsp_power_mean(u, a, w, eps=self.eps)  # [batch]
        return out.squeeze(0) if squeeze else out

    # -------------------------------------------------- AggregatorBase compat
    def aggregate_tensor(self, values, andness=None, weights=None):
        """AggregatorBase tensor path. ``andness``/``weights`` are ignored here
        because this aggregator owns its (static + dynamic) andness/weights."""
        return self.forward(values)

    def aggregate_float(self, values, andness=None, weights=None):
        out = self.forward([torch.as_tensor(float(v)) for v in values])
        return float(out.item())

    def transform_regularization(self) -> torch.Tensor:
        """``identity_reg * ||R - I||^2`` (0 when transform disabled)."""
        if not self.use_transform or self.r_logits is None or self.identity_reg == 0:
            return torch.zeros((), device=self.andness_logit.device)
        R = F.softmax(self.r_logits, dim=1)
        eye = torch.eye(R.size(0), device=R.device, dtype=R.dtype)
        return self.identity_reg * ((R - eye) ** 2).sum()
