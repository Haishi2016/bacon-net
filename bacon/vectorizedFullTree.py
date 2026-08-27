r"""Vectorized full (fully-connected) logic tree with egress-hardened routing.

A permutation-free alternative to :class:`VectorTreeLogicHead`. Evaluates
``num_heads`` independent trees in one vectorized pass. Instead of a fixed
left-fold plus a soft input permutation, every layer learns a **routing** matrix
that is hardened on the **egress** side only:

  * **Egress** (a node's outgoing edges): each source distributes over the next
    layer's destinations via a row-softmax; a row-entropy penalty peaks it and a
    confidence-triggered freeze commits each source to at most ``max_egress``
    parents (default 1 -> a clean tree). This is the only side that is hardened.
  * **Ingress** (a node's incoming edges) is *free*: a destination aggregates all
    sources routed to it (N-ary), with the child weights **normalized** to sum
    to 1 for the weighted power mean.

The per-node aggregation uses the ``full_weight`` continuous-andness power mean
(:func:`lsp_power_mean`) with a per-head, per-node andness. Static andness /
weights to start; the generic gates / partial-absorption transform can be added
later (the core math is shared with ``GenericFullWeightAggregator``).

Triangle shape: widths ``[K, K-1, ..., 1]`` over ``K-1`` layers, so the
parameters converge to a single root per head.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from bacon.aggregators.lsp.full_weight import lsp_power_mean


class VectorFullTreeHead(nn.Module):
    """Fully-connected graded-logic tree head, egress-hardened, no permutation.

    Parameters
    ----------
    input_size : int
        Number of input concepts (leaves), ``K``.
    num_heads : int
        Number of independent trees (e.g. classes).
    branching : int
        Width-reduction factor per layer (the funnel). Each layer narrows
        ``w_out = ceil(w_in / branching)`` so depth is ``~log_b(K)`` and total
        routing is ``O(K^2)`` instead of the ``O(K^3)`` reduce-by-1 triangle.
        ``branching=1`` recovers the reduce-by-1 triangle (``[K, K-1, ..., 1]``).
        Nodes stay N-ary (a parent may aggregate many children).
    max_egress : int
        Max parents a node may feed after hardening (default 1 -> tree).
    normalize_andness : bool
        Map bias -> andness via ``sigmoid(bias)*3 - 1`` in ``(-1, 2)``.
    temperature, final_temperature : float
        Routing softmax temperature schedule (annealed via :meth:`anneal`).
    use_gumbel : bool
        Add annealed Gumbel noise to routing logits during training.
    eps : float
        Output clamp for BCE stability.
    """

    def __init__(
        self,
        input_size: int,
        num_heads: int,
        *,
        branching: int = 4,
        max_egress: int = 1,
        normalize_andness: bool = True,
        weight_mode: str = "static",
        straight_through: bool = False,
        use_coefficients: bool = False,
        use_negation: bool = False,
        gate_hidden: int = 16,
        identity_reg: float = 1e-3,
        temperature: float = 3.0,
        final_temperature: float = 0.1,
        transform_final_temperature: float = 0.1,
        use_gumbel: bool = True,
        eps: float = 1e-7,
    ):
        super().__init__()
        if input_size < 1:
            raise ValueError(f"input_size must be >= 1, got {input_size}")
        if num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {num_heads}")
        if weight_mode not in ("static", "value", "full"):
            raise ValueError(f"weight_mode must be static|value|full, got {weight_mode!r}")

        self.input_size = int(input_size)
        self.num_heads = int(num_heads)
        # `branching` may be a scalar (uniform funnel) or a per-layer schedule
        # (list) so the tree can be deep-then-wide, wide-then-deep, etc.
        if isinstance(branching, (list, tuple)):
            self.branching_schedule = [int(b) for b in branching]
            self.branching = int(self.branching_schedule[0])
        else:
            self.branching_schedule = None
            self.branching = int(branching)
        self.max_egress = int(max_egress)
        self.normalize_andness = bool(normalize_andness)
        self.straight_through = bool(straight_through)
        self.use_coefficients = bool(use_coefficients)
        self.use_negation = bool(use_negation)
        self.use_gumbel = bool(use_gumbel)
        self.eps = float(eps)

        # Funnel width schedule: narrow by `branching` each layer down to 1.
        # branching=1 recovers the reduce-by-1 triangle [K, K-1, ..., 1].
        widths = [self.input_size]
        w = self.input_size
        li = 0
        while w > 1:
            b = (self.branching_schedule[min(li, len(self.branching_schedule) - 1)]
                 if self.branching_schedule else self.branching)
            w = (w - 1) if b <= 1 else max(1, math.ceil(w / b))
            widths.append(w)
            li += 1
        self.widths = widths                           # [K, ceil(K/b), ..., 1]
        self.depth = len(widths) - 1

        self._temp0 = float(temperature)
        self._temp_final = float(final_temperature)
        self.register_buffer("temperature", torch.tensor(float(temperature)))
        self.register_buffer("gumbel_noise_scale", torch.tensor(1.0))
        self.register_buffer("egress_frozen", torch.tensor(False))
        self._transform_temp_final = float(transform_final_temperature)
        self.register_buffer("transform_temperature", torch.tensor(1.0))

        self.route_logits = nn.ParameterList()
        self.andness_bias = nn.ParameterList()
        for l in range(self.depth):
            w_in, w_out = self.widths[l], self.widths[l + 1]
            self.route_logits.append(nn.Parameter(torch.randn(num_heads, w_in, w_out) * 0.1))
            # rand*3-1 in (-1, 2), matching the scalar tree's andness init.
            self.andness_bias.append(nn.Parameter(torch.rand(num_heads, w_out) * 3 - 1))
            self.register_buffer(f"frozen_route_{l}",
                                 torch.zeros(num_heads, w_in, w_out))

        # Per-head, per-concept identity/negation gate (the "transformation
        # layer" the left-fold tree already uses): each leaf may read the concept
        # ``c`` or its negation ``1-c``, so rules can express "has X AND NOT Y".
        # Softmax over {identity, negation}, identity favoured at init (+2), soft
        # during training and hardened (argmax) at freeze -> stays readable.
        if self.use_negation:
            t = torch.zeros(num_heads, self.input_size, 2)
            t[..., 0] = 2.0
            t = t + 0.01 * torch.randn_like(t)
            self.transform_logits = nn.Parameter(t)
            self.register_buffer("transform_frozen", torch.tensor(False))
            self.register_buffer("frozen_transform",
                                 torch.zeros(num_heads, self.input_size))

        # Coefficient layers (alternating-tree style): a smooth per-head,
        # per-source relevance weight applied inside the child-weights. This is
        # a continuous, always-differentiable gradient path alongside the
        # (possibly hard) routing -- it lets the model learn which inputs matter
        # even when egress is straight-through hard, stabilising deep-tree
        # training. Log-space, init 0 -> relevance 1 (no-op at start).
        if self.use_coefficients:
            self.coeff_logits = nn.ParameterList(
                nn.Parameter(torch.zeros(num_heads, self.widths[l]))
                for l in range(self.depth))

        # ---- "fancier aggregator" options (generic form), off at init ------
        self.weight_mode = weight_mode
        self.dynamic_andness = weight_mode in ("value", "full")   # value-based
        self.use_transform = weight_mode == "full"                # partial absorption
        self.identity_reg = float(identity_reg)
        if self.dynamic_andness:
            # shared per-node gate: [weighted mean, std, geo-mean] -> andness delta
            self.andness_gate = nn.Sequential(
                nn.Linear(3, gate_hidden), nn.ReLU(), nn.Linear(gate_hidden, 1))
            nn.init.zeros_(self.andness_gate[-1].weight)          # start static
            nn.init.zeros_(self.andness_gate[-1].bias)
        if self.use_transform:
            # per-layer row-stochastic R[H,w_in,w_in], identity init (partial
            # absorption mixes a layer's inputs before routing/aggregation).
            self.r_logits = nn.ParameterList()
            for l in range(self.depth):
                w_in = self.widths[l]
                diag = 5.0 + math.log(max(w_in - 1, 1))
                r0 = (torch.eye(w_in) * diag).unsqueeze(0).repeat(num_heads, 1, 1)
                self.r_logits.append(nn.Parameter(r0))

    # ---------------------------------------------------------------- routing
    def _egress(self, l: int) -> torch.Tensor:
        """Per-head routing weights ``[H, w_in, w_out]`` (rows = sources)."""
        if bool(self.egress_frozen):
            return getattr(self, f"frozen_route_{l}")
        logits = self.route_logits[l]
        if self.training and self.use_gumbel and float(self.gumbel_noise_scale) > 0:
            u = torch.rand_like(logits).clamp_(1e-20, 1.0)
            g = -torch.log(-torch.log(u))
            logits = logits + g * self.gumbel_noise_scale
        # Egress: each source (dim=1 row) distributes over destinations (dim=2).
        soft = F.softmax(logits / self.temperature.clamp_min(1e-4), dim=2)
        if self.straight_through:
            # Forward uses HARD one-hot egress (each source -> its argmax parent)
            # so the model trains/evals AS the discrete tree (no soft->hard gap);
            # backward uses the soft gradient (straight-through estimator).
            hard = torch.zeros_like(soft)
            hard.scatter_(2, soft.argmax(dim=2, keepdim=True), 1.0)
            if self.training:
                return hard - soft.detach() + soft
            return hard
        return soft

    def _child_weights(self, E: torch.Tensor) -> torch.Tensor:
        """Normalize routing over ingress (sources) so each dest's children sum to 1.

        A destination orphaned by hard egress (no source routed to it -> column
        sum ~0) would divide ~0/0 -> garbage; fall back to a uniform mean over
        all sources so the node stays well-defined (bounded in [0,1])."""
        col = E.sum(dim=1, keepdim=True)                    # [H, 1, w_out]
        w = E / (col + 1e-8)
        dead = col < 1e-6
        if bool(dead.any()):
            w = torch.where(dead.expand_as(w),
                            E.new_full(w.shape, 1.0 / E.size(1)), w)
        return w                                             # [H, w_in, w_out]

    def _apply_negation(self, V: torch.Tensor) -> torch.Tensor:
        """Per-concept identity/negation gate: replace each leaf ``c`` with
        ``g*c + (1-g)*(1-c)``, where ``g`` is the (soft, or frozen 0/1) prob of
        identity. ``V`` is ``(B, H, K)``."""
        if not self.use_negation:
            return V
        if bool(self.transform_frozen):
            g = self.frozen_transform.unsqueeze(0)               # (1,H,K) in {0,1}
        else:
            gate = F.softmax(
                self.transform_logits / self.transform_temperature.clamp_min(1e-4),
                dim=2)                                           # (H,K,2)
            g = gate[..., 0].unsqueeze(0)                        # (1,H,K) P(identity)
        return g * V + (1.0 - g) * (1.0 - V)

    # ---------------------------------------------------------------- forward
    def forward(self, x: torch.Tensor,
                egress_override: Optional[list] = None) -> torch.Tensor:
        """Evaluate the tree.

        ``egress_override`` (optional): a list of ``depth`` hard routing tensors
        ``[H, w_in, w_out]`` to use in place of the learned egress. Used by the
        candidate-scan freeze to evaluate sampled discrete structures without
        mutating the frozen buffers.
        """
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

        V = self._apply_negation(V)                       # identity/negation leaves
        B = V.size(0)
        for l in range(self.depth):
            w_in, w_out = self.widths[l], self.widths[l + 1]
            if self.use_transform:
                R = F.softmax(self.r_logits[l], dim=2)        # [H, w_in, w_in]
                V = torch.einsum("hji,bhi->bhj", R, V)        # partial absorption
            E = egress_override[l] if egress_override is not None else self._egress(l)
            if self.use_coefficients:
                rel = torch.exp(self.coeff_logits[l].clamp(-10.0, 10.0))  # [H, w_in]
                E = E * rel.unsqueeze(2)                      # smooth per-source relevance
            w_agg = self._child_weights(E)                   # [H, w_in, w_out]
            # X: (w_in, B, H, w_out) -- broadcast each source over destinations
            X = V.permute(2, 0, 1).unsqueeze(-1).expand(w_in, B, self.num_heads, w_out)
            Wn = w_agg.permute(1, 0, 2).unsqueeze(1)         # (w_in, 1, H, w_out)
            a_logit = self.andness_bias[l]                   # [H, w_out]
            if self.dynamic_andness:
                # value-based andness: gate on each node's weighted child stats.
                m1 = (Wn * X).sum(0)                          # (B,H,w_out) w-mean
                m2 = (Wn * X * X).sum(0)
                std = ((m2 - m1 * m1).clamp_min(0.0) + 1e-6).sqrt()   # safe grad
                gm = ((X.clamp_min(1e-6).log()) * Wn).sum(0).exp()   # w-geo-mean
                feats = torch.stack([m1, std, gm], dim=-1)   # (B,H,w_out,3)
                delta = self.andness_gate(feats).squeeze(-1)  # (B,H,w_out)
                A = torch.sigmoid(a_logit.unsqueeze(0) + delta) * 3.0 - 1.0
            elif self.normalize_andness:
                A = (torch.sigmoid(a_logit) * 3.0 - 1.0).unsqueeze(0)   # (1,H,w_out)
            else:
                A = a_logit.unsqueeze(0)
            V = lsp_power_mean(X, A, Wn, eps=1e-6)           # (B, H, w_out)
        out = V[..., 0]                                      # (B, H)
        return out.clamp(self.eps, 1.0 - self.eps)

    def transform_regularization(self) -> torch.Tensor:
        """``identity_reg * sum_l ||R_l - I||^2`` (0 when partial absorption off)."""
        if not self.use_transform:
            return torch.zeros((), device=self.temperature.device)
        tot = torch.zeros((), device=self.temperature.device)
        for l in range(self.depth):
            R = F.softmax(self.r_logits[l], dim=2)
            eye = torch.eye(R.size(1), device=R.device).unsqueeze(0)
            tot = tot + ((R - eye) ** 2).sum()
        return self.identity_reg * tot

    # ---------------------------------------------------------- egress control
    def egress_sparsity_loss(self) -> torch.Tensor:
        """Egress peaking (each source -> one parent) + ingress coverage (each
        destination keeps >=1 child, so hard-freezing does not orphan nodes)."""
        if bool(self.egress_frozen) or self.depth == 0:
            return torch.zeros((), device=self.temperature.device)
        total = torch.zeros((), device=self.temperature.device)
        for l in range(self.depth):
            E = self._egress(l)                              # [H, w_in, w_out]
            w_out = E.size(2)
            if w_out <= 1:
                continue
            # (a) egress peaking: concentrate each source's mass on its top-
            # `max_egress` destination(s). For a tree (max_egress=1) this is the
            # row-entropy -> 0 penalty; for a relaxed DAG (max_egress>1) drive the
            # top-k routing mass -> 1 so a source commits to k parents, not one.
            if self.max_egress <= 1:
                ent = -(E * (E + 1e-8).log()).sum(dim=2)     # per-source entropy
                peak = (ent / math.log(w_out)).mean()
            else:
                k = min(self.max_egress, w_out)
                topk_mass = E.topk(k, dim=2).values.sum(dim=2)   # [H, w_in]
                peak = (1.0 - topk_mass).mean()
            # (b) ingress coverage: penalize starved destinations (col sum < 1),
            # so the hard argmax does not orphan any node (-> no dead subtrees).
            col = E.sum(dim=1)                               # [H, w_out] exp. #children
            starve = F.relu(1.0 - col).pow(2).mean()
            total = total + peak + starve
        return total / max(self.depth, 1)

    def egress_confidence(self) -> torch.Tensor:
        """Mean top-1 SOFT routing probability across sources/layers/heads.

        Uses the underlying softmax (not the straight-through one-hot forward),
        so it measures how peaked the routing *logits* are -- otherwise ST would
        report confidence 1.0 from epoch 1 and trigger an immediate freeze."""
        confs = []
        for l in range(self.depth):
            E = F.softmax(self.route_logits[l] / self.temperature.clamp_min(1e-4), dim=2)
            if self.max_egress <= 1:
                confs.append(E.max(dim=2).values.mean())
            else:
                k = min(self.max_egress, E.size(2))
                confs.append(E.topk(k, dim=2).values.sum(dim=2).mean())
        if not confs:
            return torch.ones((), device=self.temperature.device)
        return torch.stack(confs).mean()

    def freeze_egress(self) -> None:
        """Commit each source to its top-``max_egress`` parent(s); hard routing."""
        with torch.no_grad():
            for l in range(self.depth):
                E = F.softmax(self.route_logits[l] / self.temperature.clamp_min(1e-4), dim=2)
                hard = torch.zeros_like(E)
                if self.max_egress <= 1:
                    idx = E.argmax(dim=2, keepdim=True)      # [H, w_in, 1]
                    hard.scatter_(2, idx, 1.0)
                else:
                    k = min(self.max_egress, E.size(2))
                    topk = E.topk(k, dim=2).indices           # [H, w_in, k]
                    hard.scatter_(2, topk, 1.0)
                getattr(self, f"frozen_route_{l}").copy_(hard)
        self.egress_frozen = torch.tensor(True, device=self.egress_frozen.device)
        self._freeze_transform()

    def _freeze_transform(self) -> None:
        """Commit each concept's gate to identity or negation (argmax)."""
        if not self.use_negation:
            return
        with torch.no_grad():
            gate = F.softmax(
                self.transform_logits / self.transform_temperature.clamp_min(1e-4),
                dim=2)
            self.frozen_transform.copy_((gate[..., 0] >= 0.5).float())
        self.transform_frozen = torch.tensor(True, device=self.transform_frozen.device)

    def unfreeze_egress(self) -> None:
        self.egress_frozen = torch.tensor(False, device=self.egress_frozen.device)
        if self.use_negation:
            self.transform_frozen = torch.tensor(False, device=self.transform_frozen.device)

    @torch.no_grad()
    def _sample_routes(self, temperature: float, greedy: bool = False) -> list:
        """One candidate discrete routing: a list of hard one-hots ``[H,w_in,w_out]``.

        Each source picks a single destination -- greedily (argmax) or sampled
        from the routing categorical at ``temperature`` (the soft egress is a
        *distribution over trees*; sampling explores that candidate range)."""
        routes = []
        for l in range(self.depth):
            logits = self.route_logits[l] / max(temperature, 1e-4)
            probs = F.softmax(logits, dim=2)                 # [H, w_in, w_out]
            H, w_in, w_out = probs.shape
            if greedy or w_out == 1:
                idx = probs.argmax(dim=2, keepdim=True)      # [H, w_in, 1]
            else:
                idx = torch.multinomial(probs.reshape(H * w_in, w_out), 1
                                        ).reshape(H, w_in, 1)
            hard = torch.zeros_like(probs)
            hard.scatter_(2, idx, 1.0)
            routes.append(hard)
        return routes

    @torch.no_grad()
    def freeze_egress_scan(self, x: torch.Tensor, *, num_candidates: int = 64,
                           temperature: Optional[float] = None,
                           batch_cap: int = 2048, chunk: int = 256) -> torch.Tensor:
        """Candidate-scan freeze: search the soft distribution for a hard tree.

        Rather than greedily collapsing every source to its argmax parent (which
        jointly orphans nodes and can badly distort the function), treat the soft
        egress as a **distribution over discrete trees** and *scan* candidates:
        sample ``num_candidates`` hard routings, evaluate each on ``x``, and for
        **each head independently** keep the candidate whose hard output best
        reproduces that head's soft output. The search decomposes per head (heads
        do not interact in the forward), so 200 heads = 200 cheap searches; bad
        (orphaning) candidates simply score poorly and are rejected.

        Returns the per-head faithfulness MSE of the selected trees (mean).
        """
        if self.depth == 0:
            self.egress_frozen = torch.tensor(True, device=self.egress_frozen.device)
            self._freeze_transform()
            return torch.zeros((), device=self.temperature.device)
        if self.max_egress > 1:
            # The scan samples single-parent routings; for a relaxed DAG
            # (max_egress>1) use the deterministic top-k freeze instead.
            self.freeze_egress()
            return torch.zeros((), device=self.temperature.device)

        was_training = self.training
        was_frozen = bool(self.egress_frozen)
        self.eval()
        self.egress_frozen = torch.tensor(False, device=self.egress_frozen.device)
        if x.size(0) > batch_cap:
            x = x[:batch_cap]
        temp = float(self.temperature) if temperature is None else float(temperature)

        # Evaluate the forward in calibration sub-batches: the per-node tensor is
        # (w_in, chunk, H, w_out), which for deep/wide trees can blow up VRAM at
        # the full 2048-sample cap -- chunking keeps it bounded (exact SSE).
        xs = list(torch.split(x, max(chunk, 1)))
        B = x.size(0)

        # Soft target: the good continuous function we want to preserve, per head.
        soft_chunks = [self.forward(xc) for xc in xs]        # each (b, H)

        H = self.num_heads
        best_mse = torch.full((H,), float("inf"), device=x.device)
        best_routes = [torch.zeros_like(getattr(self, f"frozen_route_{l}"))
                       for l in range(self.depth)]

        # Candidate 0 is the deterministic greedy argmax (scan never loses to it);
        # the rest are sampled from the routing categorical.
        for c in range(max(num_candidates, 1)):
            routes = self._sample_routes(temp, greedy=(c == 0))
            sse = torch.zeros(H, device=x.device)            # per-head SSE
            for xc, sc in zip(xs, soft_chunks):
                out = self.forward(xc, egress_override=routes)   # (b, H)
                sse = sse + (out - sc).pow(2).sum(dim=0)     # accumulate over batch
            mse = sse / max(B, 1)                             # (H,) per-head faithfulness
            improved = mse < best_mse                        # (H,)
            if bool(improved.any()):
                best_mse = torch.where(improved, mse, best_mse)
                for l in range(self.depth):
                    m = improved.view(H, 1, 1).expand_as(best_routes[l])
                    best_routes[l] = torch.where(m, routes[l], best_routes[l])

        for l in range(self.depth):
            getattr(self, f"frozen_route_{l}").copy_(best_routes[l])
        self.egress_frozen = torch.tensor(True, device=self.egress_frozen.device)
        self._freeze_transform()
        if was_training and not was_frozen:
            self.train()
        return best_mse.mean()

    def anneal(self, progress: float) -> None:
        """Anneal routing temperature (broad->sharp) and Gumbel noise (1->0)."""
        p = min(max(float(progress), 0.0), 1.0)
        log_t = (1 - p) * math.log(self._temp0) + p * math.log(self._temp_final)
        self.temperature = torch.tensor(math.exp(log_t), device=self.temperature.device)
        self.gumbel_noise_scale = torch.tensor(1.0 - p, device=self.gumbel_noise_scale.device)
        # Sharpen the identity/negation gate on the same schedule (1.0 -> final).
        if self.use_negation:
            self.transform_temperature = torch.tensor(
                1.0 - p * (1.0 - self._transform_temp_final),
                device=self.transform_temperature.device)
