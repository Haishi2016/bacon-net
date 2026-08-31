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
import torch.nn.functional as F

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
        use_negation: bool = False,
        use_permutation_layer: bool = True,
        normalize_andness: bool = True,
        final_temperature: float = 0.2,
        temperature: float = 3.0,
        sinkhorn_iters: int = 20,
        sinkhorn_temperature: float = 3.0,
        sinkhorn_final_temperature: float = 0.1,
        transform_final_temperature: float = 0.1,
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
        self.use_negation = bool(use_negation)
        self.use_permutation_layer = bool(use_permutation_layer)
        self.sinkhorn_iters = int(sinkhorn_iters)
        self.sinkhorn_temperature = float(sinkhorn_temperature)
        self.sinkhorn_final_temperature = float(sinkhorn_final_temperature)
        self._transform_temp_final = float(transform_final_temperature)
        self.use_gumbel = bool(use_gumbel)

        # Split: at least 1 feature to each side.
        n_bin = int(round(bin_frac * input_size))
        n_bin = max(1, min(input_size - 1, n_bin))
        self.n_bin = n_bin
        # PERMUTATION mode partitions the K concepts (spine gets n_bin, full gets
        # the rest). PERMUTATION-FREE mode overlays: the full sub-tree pools ALL
        # K concepts (== the clean-freezing full tree, a >=71% floor since a spine
        # node with weight [1,0] is the identity), and the spine SELECTS n_bin
        # concepts to surface at the root via egress-style hard selection (no
        # Sinkhorn -> clean freeze, avoiding the permutation-freeze penalty).
        self.n_full = (input_size - n_bin) if self.use_permutation_layer else input_size

        # Bulk sub-tree. Negation is handled at the hybrid level over ALL K
        # concepts (so the spine gets it too), so the sub-tree keeps use_negation=False.
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

        # Per-head soft input permutation over ALL K concepts (batched Sinkhorn,
        # same machinery as the binary tree's inputToLeafSinkhorn). This is what
        # lets each head LEARN which concepts land on its binary spine (the first
        # ``n_bin`` permuted leaves) vs its bulk full sub-tree (the rest), instead
        # of a fixed first-n_bin-by-index split. Hardened to a hard permutation at
        # freeze so the frozen tree is faithful.
        self.register_buffer("perm_temperature", torch.tensor(float(sinkhorn_temperature)))
        self.register_buffer("perm_gumbel", torch.tensor(1.0))
        if self.use_permutation_layer and input_size > 1:
            self.perm_logits = nn.Parameter(torch.randn(num_heads, input_size, input_size))
            self.register_buffer("perm_frozen", torch.tensor(False))
            self.register_buffer("frozen_perm",
                                 torch.zeros(num_heads, input_size, input_size))
        else:
            self.register_parameter("perm_logits", None)
            self.register_buffer("perm_frozen", torch.tensor(False))
            self.frozen_perm = None

        # Permutation-FREE spine selection: each of the n_bin spine slots picks a
        # concept via a row-softmax over the K concepts (annealed), hardened to a
        # one-hot argmax at freeze -- egress-style routing that commits cleanly
        # (unlike a Sinkhorn permutation). Slots may repeat a concept; that is
        # acceptable (a redundant surfaced literal), the point is the clean freeze.
        if not self.use_permutation_layer:
            self.spine_select_logits = nn.Parameter(
                torch.randn(num_heads, self.n_bin, input_size) * 0.1)
            self.register_buffer("select_frozen", torch.tensor(False))
            self.register_buffer("frozen_select",
                                 torch.zeros(num_heads, self.n_bin, input_size))
        else:
            self.register_parameter("spine_select_logits", None)
            self.register_buffer("select_frozen", torch.tensor(False))
            self.frozen_select = None

        # Per-head per-concept identity/negation gate over ALL K concepts, so any
        # leaf (spine or bulk) may read ``c`` or ``1-c``. Softmax over
        # {identity, negation}, identity favoured at init (+2), hardened at freeze.
        self.register_buffer("transform_temperature", torch.tensor(1.0))
        if self.use_negation:
            t = torch.zeros(num_heads, input_size, 2)
            t[..., 0] = 2.0
            t = t + 0.01 * torch.randn_like(t)
            self.transform_logits = nn.Parameter(t)
            self.register_buffer("transform_frozen", torch.tensor(False))
            self.register_buffer("frozen_transform",
                                 torch.zeros(num_heads, input_size))
        else:
            self.register_parameter("transform_logits", None)

    # -- permutation / selection / negation helpers ------------------------
    def _sinkhorn(self, logits: torch.Tensor) -> torch.Tensor:
        """Batched Sinkhorn -> doubly-stochastic ``(num_heads, K, K)`` per head."""
        temp = self.perm_temperature.clamp(min=1e-6)
        lg = (logits / temp).clamp(-10.0, 10.0)
        if self.training and self.use_gumbel and float(self.perm_gumbel) > 0.0:
            u = torch.rand_like(lg)
            g = -torch.log(-torch.log(u + 1e-20) + 1e-20)
            lg = lg + g * self.perm_gumbel
        A = lg.exp()
        for _ in range(self.sinkhorn_iters):
            A = A / (A.sum(dim=2, keepdim=True) + 1e-10)
            A = A / (A.sum(dim=1, keepdim=True) + 1e-10)
        return torch.nan_to_num(A, nan=0.0, posinf=1.0, neginf=0.0)

    def _spine_selection(self) -> torch.Tensor:
        """Per-head spine selection ``(H, n_bin, K)``: soft row-softmax over the
        K concepts (annealed via ``perm_temperature``) or the frozen one-hot."""
        if bool(self.select_frozen):
            return self.frozen_select
        temp = self.perm_temperature.clamp(min=1e-6)
        lg = (self.spine_select_logits / temp).clamp(-10.0, 10.0)
        if self.training and self.use_gumbel and float(self.perm_gumbel) > 0.0:
            u = torch.rand_like(lg)
            g = -torch.log(-torch.log(u + 1e-20) + 1e-20)
            lg = lg + g * self.perm_gumbel
        return F.softmax(lg, dim=2)                              # (H, n_bin, K)

    def _apply_negation(self, V: torch.Tensor) -> torch.Tensor:
        """Per-head per-concept identity/negation over ``V`` = (B, H, K)."""
        if not self.use_negation:
            return V
        if bool(self.transform_frozen):
            g = self.frozen_transform.unsqueeze(0)                # (1,H,K) in {0,1}
        else:
            gate = F.softmax(
                self.transform_logits / self.transform_temperature.clamp_min(1e-4),
                dim=2)
            g = gate[..., 0].unsqueeze(0)                         # (1,H,K) P(identity)
        return g * V + (1.0 - g) * (1.0 - V)



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

        # Broadcast to per-head values (B, H, K).
        if x.dim() == 2:
            xb = x.unsqueeze(1).expand(x.size(0), self.num_heads, self.input_size)
        else:
            xb = x
        xb = self._apply_negation(xb)                     # identity/negation leaves

        if self.use_permutation_layer:
            # Permutation mode: reorder all K per head, split first n_bin -> spine.
            P = self.frozen_perm if bool(self.perm_frozen) else self._sinkhorn(self.perm_logits)
            leaves = torch.einsum("bhn,hln->bhl", xb, P)  # (B, H, K) reordered per head
            x_bin, x_full = leaves[..., : self.n_bin], leaves[..., self.n_bin:]
            node = self.full(x_full)                      # (B, H) bulk over the rest
        else:
            # Permutation-free mode: full sub-tree pools ALL K; spine SELECTS
            # n_bin concepts to surface at the root via egress-style selection.
            node = self.full(xb)                          # (B, H) full tree over all K
            Sel = self._spine_selection()                 # (H, n_bin, K)
            x_bin = torch.einsum("hik,bhk->bhi", Sel, xb)  # (B, H, n_bin) selected

        # Left-associative fold: deepest feature (index n_bin-1) folds in first,
        # the most important feature (index 0) folds in last at the root.
        for i in range(self.n_bin - 1, -1, -1):
            feat = x_bin[..., i]                          # (B, H)
            a_logit = self.bin_andness[:, i]             # (H,)
            a = (torch.sigmoid(a_logit) * 3.0 - 1.0) if self.normalize_andness else a_logit
            wf = torch.sigmoid(self.bin_weight_logit[:, i])           # (H,) weight on new feat
            X = torch.stack([node, feat], dim=0)                       # (2, B, H)
            w = torch.stack([1.0 - wf, wf], dim=0).unsqueeze(1)       # (2, 1, H) sums to 1
            node = lsp_power_mean(X, a.unsqueeze(0), w, eps=1e-6)     # (B, H)

        return node.clamp(self.eps, 1.0 - self.eps)


    # -- permutation / negation freezing (called from freeze_egress) --------
    @torch.no_grad()
    def _freeze_permutation(self) -> None:
        """Hungarian-harden the soft permutation into a locked hard permutation."""
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
            rows, cols = linear_sum_assignment(-Pc[h])
            hard[h, rows, cols] = 1.0
        self.frozen_perm.copy_(hard.to(self.frozen_perm.device))
        self.perm_frozen.fill_(True)
        self.perm_logits.requires_grad_(False)

    @torch.no_grad()
    def _freeze_selection(self) -> None:
        """Argmax-harden the spine selection into one-hot (egress-style, clean)."""
        if self.spine_select_logits is None or bool(self.select_frozen):
            return
        was_training = self.training
        self.eval()
        Sel = self._spine_selection()                            # (H, n_bin, K)
        self.train(was_training)
        hard = torch.zeros_like(Sel)
        hard.scatter_(2, Sel.argmax(dim=2, keepdim=True), 1.0)
        self.frozen_select.copy_(hard.to(self.frozen_select.device))
        self.select_frozen.fill_(True)
        self.spine_select_logits.requires_grad_(False)


    @torch.no_grad()
    def _freeze_transform(self) -> None:
        """Commit each concept's gate to identity or negation (argmax)."""
        if not self.use_negation or bool(self.transform_frozen):
            return
        gate = F.softmax(
            self.transform_logits / self.transform_temperature.clamp_min(1e-4), dim=2)
        self.frozen_transform.copy_((gate[..., 0] >= 0.5).float())
        self.transform_frozen = torch.tensor(True, device=self.transform_frozen.device)
        self.transform_logits.requires_grad_(False)


    # ------------------------------------------- egress control (delegated)
    @property
    def egress_frozen(self):
        return self.full.egress_frozen

    def egress_sparsity_loss(self) -> torch.Tensor:
        """Full sub-tree egress peaking + binary-spine routing peaking (permutation
        entropy, or selection entropy in the permutation-free variant)."""
        loss = self.full.egress_sparsity_loss()
        if self.use_permutation_layer:
            if self.perm_logits is not None and not bool(self.perm_frozen):
                P = self._sinkhorn(self.perm_logits)
                ent = -(P.clamp_min(1e-9) * P.clamp_min(1e-9).log()).sum(dim=2)   # (H, K)
                loss = loss + ent.mean()
        else:
            if not bool(self.select_frozen):
                Sel = self._spine_selection()
                ent = -(Sel.clamp_min(1e-9) * Sel.clamp_min(1e-9).log()).sum(dim=2)  # (H,n_bin)
                loss = loss + ent.mean()
        return loss

    def egress_confidence(self) -> torch.Tensor:
        """Min of the full sub-tree's egress confidence and the spine routing's
        peak weight, so freezing waits until BOTH are committed."""
        c = self.full.egress_confidence()
        if not torch.is_tensor(c):
            c = torch.as_tensor(float(c), device=self.perm_temperature.device)
        if self.use_permutation_layer:
            if self.perm_logits is not None and not bool(self.perm_frozen):
                P = self._sinkhorn(self.perm_logits)
                c = torch.minimum(c, P.max(dim=2).values.mean())
        else:
            if not bool(self.select_frozen):
                Sel = self._spine_selection()
                c = torch.minimum(c, Sel.max(dim=2).values.mean())
        return c

    def freeze_egress(self) -> None:
        """Freeze the full sub-tree egress, the spine routing (permutation or
        egress-style selection), and the negation gate together -> faithful frozen."""
        self.full.freeze_egress()
        self._freeze_permutation()
        self._freeze_selection()
        self._freeze_transform()


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
        """Advance the full sub-tree's schedule AND the permutation / negation
        temperature schedules (broad -> peaked, matching the binary tree)."""
        p = min(max(float(progress), 0.0), 1.0)
        self.perm_temperature.fill_(
            self.sinkhorn_temperature
            - p * (self.sinkhorn_temperature - self.sinkhorn_final_temperature))
        self.perm_gumbel.fill_(1.0 - p)
        self.transform_temperature.fill_(1.0 - p * (1.0 - self._transform_temp_final))
        self.full.anneal(progress)

