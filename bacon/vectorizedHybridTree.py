r"""Hybrid tree head: a left-associative binary **spine** over the most
important features, fed at its deepest node by a shallow fully-connected
sub-tree that pools the remaining features.

Motivation
----------
The permutation-free :class:`~bacon.vectorizedFullTree.VectorFullTreeHead`
hardens best when it is *shallow* (few egress layers to discretise). A pure
left-associative binary tree, on the other hand, lets a handful of important
features "surface" to positions near the root where they exert direct influence.
This head combines both:

  * the first ``n_bin = round(bin_frac * K)`` features form a **binary spine** --
    a left fold ``(((full_agg op f_{n-1}) op f_{n-2}) ... op f_0)`` where the
    most important feature ``f_0`` lands at the root (top) and the bulk aggregate
    enters at the deepest node;
  * the remaining ``K - n_bin`` features are pooled by a shallow
    ``VectorFullTreeHead`` whose single output becomes that deepest node.

Leveraging existing aggregators
-------------------------------
Nothing here re-implements aggregation. The bulk sub-tree *is* a
``VectorFullTreeHead`` (built on :func:`lsp_power_mean`), and each binary fold
step is the **same** ``lsp_power_mean`` continuous-andness power mean used by
``FullWeightAggregator`` / ``GenericFullWeightAggregator`` -- with the identical
andness parameterisation ``a = sigmoid(bias) * 3 - 1`` in ``[-1, 2]``.

Hardening
---------
Only the full sub-tree carries learned routing that must be discretised; the
binary spine is a fixed deterministic left fold (no permutation, no routing), so
it is always faithful. All egress control (sparsity loss, confidence, freeze,
candidate-scan freeze, anneal) is delegated to the full sub-tree.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from bacon.aggregators.lsp.full_weight import lsp_power_mean
from bacon.vectorizedFullTree import VectorFullTreeHead


class VectorHybridTreeHead(nn.Module):
    """Binary spine (important features) + full sub-tree (bulk), per head.

    Parameters
    ----------
    input_size : int
        Number of input concepts, ``K``.
    num_heads : int
        Number of independent trees (e.g. classes).
    bin_frac : float
        Fraction of inputs routed to the binary spine (default ``0.25``). The
        remaining inputs feed the full sub-tree. Clamped so both parts get at
        least one input.
    branching, max_egress, use_coefficients, final_temperature, temperature,
    use_gumbel, normalize_andness, eps
        Forwarded to the bulk :class:`VectorFullTreeHead` (and, where relevant,
        used by the binary spine's andness mapping).
    """

    def __init__(
        self,
        input_size: int,
        num_heads: int,
        *,
        bin_frac: float = 0.25,
        branching: int = 8,
        max_egress: int = 2,
        use_coefficients: bool = True,
        normalize_andness: bool = True,
        final_temperature: float = 0.2,
        temperature: float = 3.0,
        use_gumbel: bool = True,
        eps: float = 1e-7,
        **fulltree_kwargs,
    ):
        super().__init__()
        if input_size < 2:
            raise ValueError(f"input_size must be >= 2 for a hybrid tree, got {input_size}")
        if num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {num_heads}")

        self.input_size = int(input_size)
        self.num_heads = int(num_heads)
        self.normalize_andness = bool(normalize_andness)
        self.eps = float(eps)

        # Split: at least 1 feature to each side.
        n_bin = int(round(bin_frac * input_size))
        n_bin = max(1, min(input_size - 1, n_bin))
        self.n_bin = n_bin
        self.n_full = input_size - n_bin

        # Bulk sub-tree over the remaining features (existing full-tree aggregator).
        self.full = VectorFullTreeHead(
            self.n_full, num_heads,
            branching=branching, max_egress=max_egress,
            use_coefficients=use_coefficients, normalize_andness=normalize_andness,
            final_temperature=final_temperature, temperature=temperature,
            use_gumbel=use_gumbel, eps=eps, **fulltree_kwargs,
        )

        # Binary spine: one continuous-andness power-mean node per fold step.
        # rand*3-1 in (-1, 2) matches the full tree's andness init; the weight
        # logit (init 0 -> 0.5) balances the accumulated node vs the new feature.
        self.bin_andness = nn.Parameter(torch.rand(num_heads, self.n_bin) * 3 - 1)
        self.bin_weight_logit = nn.Parameter(torch.zeros(num_heads, self.n_bin))

    # ------------------------------------------------------------------ split
    def _split(self, x: torch.Tensor):
        """Return ``(x_bin, x_full)`` slices of the last (feature) dim."""
        return x[..., : self.n_bin], x[..., self.n_bin:]

    # ---------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() not in (2, 3):
            raise ValueError(f"x must be 2D or 3D, got {x.dim()}D")
        if x.size(-1) != self.input_size:
            raise ValueError(f"expected input_size={self.input_size}, got {x.size(-1)}")

        x_bin, x_full = self._split(x)
        node = self.full(x_full)                          # (B, H)

        # Broadcast the binary features to per-head values (B, H, n_bin).
        if x_bin.dim() == 2:
            xb = x_bin.unsqueeze(1).expand(x_bin.size(0), self.num_heads, self.n_bin)
        else:
            xb = x_bin

        # Left-associative fold: deepest feature (index n_bin-1) folds in first,
        # the most important feature (index 0) folds in last at the root.
        for i in range(self.n_bin - 1, -1, -1):
            feat = xb[..., i]                             # (B, H)
            a_logit = self.bin_andness[:, i]             # (H,)
            a = (torch.sigmoid(a_logit) * 3.0 - 1.0) if self.normalize_andness else a_logit
            wf = torch.sigmoid(self.bin_weight_logit[:, i])           # (H,) weight on new feat
            X = torch.stack([node, feat], dim=0)                       # (2, B, H)
            w = torch.stack([1.0 - wf, wf], dim=0).unsqueeze(1)       # (2, 1, H) sums to 1
            node = lsp_power_mean(X, a.unsqueeze(0), w, eps=1e-6)     # (B, H)

        return node.clamp(self.eps, 1.0 - self.eps)

    # ------------------------------------------- egress control (delegated)
    @property
    def egress_frozen(self):
        return self.full.egress_frozen

    def egress_sparsity_loss(self) -> torch.Tensor:
        return self.full.egress_sparsity_loss()

    def egress_confidence(self) -> torch.Tensor:
        return self.full.egress_confidence()

    def freeze_egress(self) -> None:
        self.full.freeze_egress()

    def unfreeze_egress(self) -> None:
        self.full.unfreeze_egress()

    def freeze_egress_scan(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Candidate-scan freeze on the bulk sub-tree only (the spine is fixed).

        The full sub-tree's scalar output is what feeds the binary spine, so
        preserving that output faithfully preserves the hybrid output; the scan
        therefore calibrates on the bulk slice of ``x``.
        """
        _, x_full = self._split(x)
        return self.full.freeze_egress_scan(x_full, **kwargs)

    def transform_regularization(self) -> torch.Tensor:
        return self.full.transform_regularization()

    def anneal(self, progress: float) -> None:
        self.full.anneal(progress)
