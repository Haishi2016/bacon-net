"""BACON logic head for the CIBM CBM.

Drop-in replacement for the official ``[n_concepts -> n_classes]`` classifier in
``models.BasicMLP``. The training protocol is kept byte-for-byte identical to the
official CIBM baseline (frozen InceptionV3 embeddings, concept encoder, softmax
CE task loss + summed concept BCE, Adam + cosine); ONLY the concept->label map is
swapped for BACON's vectorized graded-logic head -- one logic tree per class
reading the shared concept bottleneck. Prediction is ``argmax`` over the 200
per-class truth values.

Same model contract as ``BasicMLP`` so the existing train/eval/intervention code
works unchanged:

* ``forward(x) -> (class_logits, None, concept_logits)``  (``None`` == not stochastic)
* ``forward_tti(x, gt_concepts, allowed_idx)``            (test-time intervention)

Two BACON-intrinsic mechanisms are exposed for the train loop:

* ``anneal(progress)``      -- Sinkhorn routing schedule (broad -> near-permutation)
* ``param_groups(lr, mult)`` -- optional higher LR for the fresh logic head
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

# Make the ``bacon`` package importable. This file lives at
# lab/cbm/cibm/src/bacon_head.py, so the repo root (c:/School/bacon-net) is four
# parents up.
_REPO_ROOT = Path(__file__).resolve().parents[4]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bacon.binaryTreeLogicNet import binaryTreeLogicNet  # noqa: E402
from bacon.baconNet import _aggregator_registry  # noqa: E402

from models import StochasticMLP  # noqa: E402

# Truth values are clamped before the logit transform for numerical stability.
_EPS = 1e-6


def _make_aggregator(name: str):
    if name not in _aggregator_registry:
        raise KeyError(
            f"Unknown aggregator '{name}'. Available: {sorted(_aggregator_registry)}"
        )
    return _aggregator_registry[name]()


def _build_tree_bank(num_trees, num_concepts, aggregator, tree_layout,
                     use_permutation_layer, use_transformation_layer,
                     normalize_andness, sinkhorn_temperature,
                     sinkhorn_final_temperature, extra_kwargs):
    """Construct ``num_trees`` independent scalar ``binaryTreeLogicNet`` trees.

    Each class gets its **own** full BACON tree, so the *structure* (routing /
    edge selection / per-node andness) is learned per class -- not just per-class
    parameters over a shared topology like the vectorized head. Every layout the
    scalar net supports (``left``, ``balanced``, ``paired``, ``full``,
    ``alternating``) works here, with all of that layout's native mechanisms
    (full-tree egress/ingress, alternating coefficients, transformation layer).

    Returns an ``nn.ModuleList`` of trees with their viz cache disabled (the
    cache is only read by ``visualization.py``; skipping it avoids N per-node
    tensor clones per forward).
    """
    trees = nn.ModuleList()
    for _ in range(num_trees):
        # A fresh aggregator instance per tree: stateful aggregators (e.g. the
        # full-tree / per-node parameterized ones) must not share parameters
        # across classes, or every class would be tied to the same operators.
        tree = binaryTreeLogicNet(
            input_size=num_concepts,
            aggregator=_make_aggregator(aggregator),
            tree_layout=tree_layout,
            weight_mode="trainable",
            normalize_andness=normalize_andness,
            use_permutation_layer=use_permutation_layer,
            use_transformation_layer=use_transformation_layer,
            head_type="binary",
            **extra_kwargs,
        )
        tree.cache_layer_outputs = False
        # Match the vector head's Sinkhorn endpoints where the routing layer
        # exposes them (left/balanced/paired soft permutation).
        leaf = getattr(tree, "input_to_leaf", None)
        if leaf is not None and hasattr(leaf, "temperature"):
            leaf.temperature = float(sinkhorn_temperature)
        trees.append(tree)
    return trees


def _bank_truths(trees, probs):
    """Evaluate a bank of scalar trees on shared concept truths ``probs``.

    ``probs`` is ``(batch, num_concepts)``; returns ``(batch, num_trees)`` truth
    values (each tree outputs ``(batch, 1)``). Trees are independent modules so
    this is a Python loop, but each tree's own forward is fully batched over the
    sample dimension.
    """
    cols = [tree(probs) for tree in trees]  # each (batch, 1)
    return torch.cat(cols, dim=1)            # (batch, num_trees)


class BaconCBM(nn.Module):
    """Concept encoder + one BACON graded-logic tree per class (one-vs-rest)."""

    def __init__(
        self,
        arch,
        activation: str = "relu",
        backbone=None,
        aggregator: str = "lsp.full_weight",
        tree_layout: str = "left",
        use_permutation_layer: bool = True,
        use_transformation_layer: bool = True,
        sinkhorn_temperature: float = 3.0,
        sinkhorn_final_temperature: float = 0.1,
        concept_dropout: float = 0.0,
    ):
        super().__init__()
        self.arch = arch
        self.backbone = backbone
        self.num_concepts = arch[-2]
        self.num_classes = arch[-1]
        self.concept_dropout = float(concept_dropout)
        activation_class = nn.ReLU if activation == "relu" else nn.Tanh

        # Concept encoder: identical structure to BasicMLP.enc (maps the input
        # embedding through arch[0..-2] down to the n_concepts bottleneck).
        layers = [nn.Linear(arch[0], arch[1])]
        for l in range(1, len(arch) - 2):
            layers += [activation_class(), nn.Linear(arch[l], arch[l + 1])]
        self.enc = nn.Sequential(*layers)

        # BACON logic head: one tree per class over the shared concept layer.
        self.vector_net = binaryTreeLogicNet(
            input_size=self.num_concepts,
            aggregator=_make_aggregator(aggregator),
            tree_layout=tree_layout,
            weight_mode="trainable",
            normalize_andness=True,
            use_permutation_layer=use_permutation_layer,
            use_transformation_layer=use_transformation_layer,
            head_type="vector",
            num_heads=self.num_classes,
        )
        head = getattr(self.vector_net, "vector_head", None)
        if head is not None and hasattr(head, "sinkhorn_final_temperature"):
            head.sinkhorn_temperature = float(sinkhorn_temperature)
            head.sinkhorn_final_temperature = float(sinkhorn_final_temperature)
            head.temperature.fill_(float(sinkhorn_temperature))

    def _apply(self, fn, recurse: bool = True):
        # ``binaryTreeLogicNet`` caches ``.device`` for tensor allocation during
        # forward; keep it in sync whenever the module is moved (``model.to``).
        out = super()._apply(fn, recurse)
        self.vector_net.device = self.enc[0].weight.device
        return out

    def _concepts(self, x):
        if self.backbone is not None:
            x = self.backbone(x)
        return self.enc(x)

    def _logits_from_concepts(self, concept_logits):
        # Concepts enter the logic head as truth values in [0, 1].
        probs = torch.sigmoid(concept_logits)
        # Neutral concept dropout (head regularization): during training replace a
        # fraction of concept truths with 0.5 -- the graded-logic "unknown" value --
        # so the per-class routing cannot memorize exact concept signatures of the
        # ~24 training images/class. Concept supervision is unaffected (the concept
        # loss uses concept_logits, computed pre-dropout in forward()).
        if self.training and self.concept_dropout > 0.0:
            drop = torch.rand_like(probs) < self.concept_dropout
            probs = torch.where(drop, torch.full_like(probs, 0.5), probs)
        truths = self.vector_net(probs)
        truths = truths.clamp(_EPS, 1.0 - _EPS)
        # Logit transform so argmax/softmax-CE behave; argmax is preserved.
        return torch.log(truths) - torch.log1p(-truths)

    def get_activations(self, x):
        # Parity with BasicMLP for any MI estimation hooks (unused when
        # collect_MIs=False, which is the BACON-CBM setting).
        return [self._concepts(x)]

    def forward(self, x):
        concept_logits = self._concepts(x)
        class_logits = self._logits_from_concepts(concept_logits)
        return class_logits, None, concept_logits

    def forward_tti(self, x, gt_concepts, allowed_idx):
        concept_logits = self._concepts(x)
        gt_logits = (gt_concepts - 0.5) * 6.0  # {0,1} -> {-3,+3} ~ sigmoid logits
        concept_logits = concept_logits.clone()
        concept_logits[:, allowed_idx] = gt_logits[:, allowed_idx]
        return self._logits_from_concepts(concept_logits)

    def anneal(self, progress: float) -> None:
        """Advance the Sinkhorn routing schedule (broad -> near-permutation)."""
        if hasattr(self.vector_net, "anneal_vector_tree"):
            self.vector_net.anneal_vector_tree(progress)

    def param_groups(self, base_lr: float, head_lr_mult: float = 1.0):
        """Optimizer param groups; logic head at ``base_lr * head_lr_mult``.

        ``head_lr_mult == 1.0`` (default) reproduces a single shared LR identical
        to the official baseline. With frozen embeddings the concept encoder is
        also freshly initialized, so a single LR is the faithful default; the
        multiplier exists only as a knob if the logic head's routing needs to
        train faster than the concept encoder.
        """
        if head_lr_mult == 1.0:
            return [{"params": list(self.parameters()), "lr": base_lr}]
        enc_params = list(self.enc.parameters())
        if self.backbone is not None:
            enc_params += list(self.backbone.parameters())
        enc_ids = {id(p) for p in enc_params}
        head_params = [p for p in self.parameters() if id(p) not in enc_ids]
        return [
            {"params": enc_params, "lr": base_lr},
            {"params": head_params, "lr": base_lr * head_lr_mult},
        ]


class StochasticBaconCBM(StochasticMLP):
    """IBE stochastic concept encoder + BACON logic head (IB-regularized BACON-CBM).

    Combines the two orthogonal regularizers:

    * **IBE** regularizes the X->C map: a stochastic concept encoder samples
      ``z = mu + sigma * eps`` and the train loop penalizes ``I(X; C)`` via the
      Lagrangian dual update, yielding minimal-sufficient concepts. (Inherited
      verbatim from :class:`StochasticMLP` -- ``pred_mu`` / ``pred_sigma`` /
      ``log_marg_prob`` / ``uncertainties`` / ``get_activations`` are reused, so
      the existing MI estimator, beta dual update and intervention-uncertainty
      code all work unchanged.)
    * **BACON** regularizes the C->Y map: the sampled concepts feed a vectorized
      graded-logic head (one logic tree per class) instead of a linear ``cls``.

    The stochastic sampling doubles as principled, learned concept noise for the
    logic head (cf. fixed ``concept_dropout``), directly countering the per-class
    signature memorization observed in the deterministic BACON-CBM.

    Exposes the same stochastic interface as ``StochasticMLP``
    (``forward`` returns ``(class_logits, stds.mean().item(), z)``;
    ``forward(x, return_log_prob=True)`` returns ``(z, logprob)``), so
    ``is_stochastic=True`` training works with no train-loop changes.
    """

    def __init__(
        self,
        arch,
        activation: str = "relu",
        backbone=None,
        aggregator: str = "lsp.full_weight",
        tree_layout: str = "left",
        use_permutation_layer: bool = True,
        use_transformation_layer: bool = True,
        sinkhorn_temperature: float = 3.0,
        sinkhorn_final_temperature: float = 0.1,
        concept_dropout: float = 0.0,
    ):
        super().__init__(arch, activation, backbone=backbone)
        self.num_concepts = arch[-2]
        self.num_classes = arch[-1]
        self.concept_dropout = float(concept_dropout)

        # Replace the linear classifier (z -> num_classes) with the BACON head.
        self.cls = None
        self.vector_net = binaryTreeLogicNet(
            input_size=self.num_concepts,
            aggregator=_make_aggregator(aggregator),
            tree_layout=tree_layout,
            weight_mode="trainable",
            normalize_andness=True,
            use_permutation_layer=use_permutation_layer,
            use_transformation_layer=use_transformation_layer,
            head_type="vector",
            num_heads=self.num_classes,
        )
        head = getattr(self.vector_net, "vector_head", None)
        if head is not None and hasattr(head, "sinkhorn_final_temperature"):
            head.sinkhorn_temperature = float(sinkhorn_temperature)
            head.sinkhorn_final_temperature = float(sinkhorn_final_temperature)
            head.temperature.fill_(float(sinkhorn_temperature))

    def _apply(self, fn, recurse: bool = True):
        out = super()._apply(fn, recurse)
        self.vector_net.device = self.pred_mu[0].weight.device
        return out

    def _head(self, z):
        # Sampled concept logits z -> truth values in [0, 1] -> logic head.
        probs = torch.sigmoid(z)
        if self.training and self.concept_dropout > 0.0:
            drop = torch.rand_like(probs) < self.concept_dropout
            probs = torch.where(drop, torch.full_like(probs, 0.5), probs)
        truths = self.vector_net(probs).clamp(_EPS, 1.0 - _EPS)
        return torch.log(truths) - torch.log1p(-truths)

    def forward(self, x, return_log_prob=False):
        if self.backbone is not None:
            x = self.backbone(x)
        means = self.pred_mu(x)
        stds = self.pred_sigma(x).clamp(min=1e-7)
        eps = torch.randn_like(means)
        z = means + stds * eps
        if return_log_prob:
            distr = torch.distributions.normal.Normal(means, stds)
            logprob = distr.log_prob(z).sum(dim=1)
            return z, logprob
        return self._head(z), stds.mean().item(), z

    def forward_tti(self, x, gt_concepts, allowed_idx):
        if self.backbone is not None:
            x = self.backbone(x)
        means = self.pred_mu(x)
        stds = self.pred_sigma(x).clamp(min=1e-7)
        eps = torch.randn_like(means)
        z = means + stds * eps
        # Match StochasticMLP.forward_tti: inject {0,1} concepts scaled to z's range.
        gt_logits = 2 * (gt_concepts - 0.5)
        gt_logits = gt_logits * (torch.max(z) - torch.mean(z))
        z = z.clone()
        z[:, allowed_idx] = gt_logits[:, allowed_idx]
        return self._head(z)

    def anneal(self, progress: float) -> None:
        if hasattr(self.vector_net, "anneal_vector_tree"):
            self.vector_net.anneal_vector_tree(progress)

    def param_groups(self, base_lr: float, head_lr_mult: float = 1.0):
        if head_lr_mult == 1.0:
            return [{"params": list(self.parameters()), "lr": base_lr}]
        enc_params = list(self.pred_mu.parameters()) + list(self.pred_sigma.parameters())
        if self.backbone is not None:
            enc_params += list(self.backbone.parameters())
        enc_ids = {id(p) for p in enc_params}
        head_params = [p for p in self.parameters() if id(p) not in enc_ids]
        return [
            {"params": enc_params, "lr": base_lr},
            {"params": head_params, "lr": base_lr * head_lr_mult},
        ]


class MultiTreeBaconCBM(nn.Module):
    """Concept encoder + ``num_classes`` **independent** BACON trees.

    Unlike :class:`BaconCBM` (one *vectorized* head where all classes share a
    single tree topology and differ only by per-head parameters), this builds one
    full scalar :class:`~bacon.binaryTreeLogicNet.binaryTreeLogicNet` per class.
    Each class therefore learns its **own aggregation structure** (routing / edge
    selection / per-node andness), which is the point: different bird classes can
    use genuinely different logic trees. Every scalar layout and all of its native
    mechanisms (soft input permutation, transformation layer, full-tree
    egress/ingress routing, alternating coefficients) are available unchanged.

    Trade-off: ``num_classes`` separate modules are evaluated in a Python loop, so
    this is heavier than the batched vector head (mitigated by disabling the
    per-node viz cache via ``cache_layer_outputs=False``).

    Same model contract as ``BasicMLP`` / :class:`BaconCBM` so the existing
    train/eval/intervention code is unchanged.
    """

    def __init__(
        self,
        arch,
        activation: str = "relu",
        backbone=None,
        aggregator: str = "lsp.full_weight",
        tree_layout: str = "left",
        use_permutation_layer: bool = True,
        use_transformation_layer: bool = True,
        normalize_andness: bool = True,
        sinkhorn_temperature: float = 3.0,
        sinkhorn_final_temperature: float = 0.1,
        concept_dropout: float = 0.0,
        use_negative_sampling: bool = False,
        tree_kwargs=None,
    ):
        super().__init__()
        self.arch = arch
        self.backbone = backbone
        self.num_concepts = arch[-2]
        self.num_classes = arch[-1]
        self.concept_dropout = float(concept_dropout)
        self.use_negative_sampling = use_negative_sampling
        self.sinkhorn_temperature = float(sinkhorn_temperature)
        self.sinkhorn_final_temperature = float(sinkhorn_final_temperature)
        activation_class = nn.ReLU if activation == "relu" else nn.Tanh

        # Concept encoder: identical structure to BasicMLP.enc.
        layers = [nn.Linear(arch[0], arch[1])]
        for l in range(1, len(arch) - 2):
            layers += [activation_class(), nn.Linear(arch[l], arch[l + 1])]
        self.enc = nn.Sequential(*layers)

        # One independent BACON tree per class.
        self.trees = _build_tree_bank(
            self.num_classes, self.num_concepts, aggregator, tree_layout,
            use_permutation_layer, use_transformation_layer, normalize_andness,
            sinkhorn_temperature, sinkhorn_final_temperature,
            dict(tree_kwargs or {}),
        )

    def _apply(self, fn, recurse: bool = True):
        out = super()._apply(fn, recurse)
        dev = self.enc[0].weight.device
        for tree in self.trees:
            tree.to(dev)
        return out

    def _concepts(self, x):
        if self.backbone is not None:
            x = self.backbone(x)
        return self.enc(x)

    def _logits_from_concepts(self, concept_logits, targets=None):
        probs = torch.sigmoid(concept_logits)
        if self.training and self.concept_dropout > 0.0:
            drop = torch.rand_like(probs) < self.concept_dropout
            probs = torch.where(drop, torch.full_like(probs, 0.5), probs)
        
        # During training with targets: use negative sampling (2 trees per sample)
        # During evaluation or without targets: evaluate all 200 trees
        if self.training and targets is not None:
            truths = self._evaluate_negative_sampled(probs, targets)
        else:
            # Evaluation: evaluate all trees
            truths = _bank_truths(self.trees, probs)
        
        truths = truths.clamp(_EPS, 1.0 - _EPS)
        return torch.log(truths) - torch.log1p(-truths)

    def _evaluate_negative_sampled(self, probs, targets):
        """Evaluate only positive class tree and 1 random negative per sample.
        
        For homogeneous per-class batches, evaluate exactly two trees total:
        the positive class tree once on the full batch and one negative tree once
        on the full batch. Fall back to per-sample sampling only for mixed-label
        batches.
        """
        batch_size = probs.shape[0]
        device = probs.device
        truths = torch.zeros(batch_size, self.num_classes, device=device, dtype=probs.dtype)

        if batch_size == 0:
            return truths

        target_classes = targets.reshape(-1).to(device=device, dtype=torch.long)
        if torch.all(target_classes == target_classes[0]):
            pos_class = int(target_classes[0].item())
            neg_class = torch.randint(0, self.num_classes, (1,), device=device).item()
            while neg_class == pos_class and self.num_classes > 1:
                neg_class = torch.randint(0, self.num_classes, (1,), device=device).item()

            truths[:, pos_class] = self.trees[pos_class](probs).squeeze(-1)
            truths[:, neg_class] = self.trees[neg_class](probs).squeeze(-1)
            return truths
        
        for i in range(batch_size):
            pos_class = int(target_classes[i].item())
            
            # Evaluate positive tree for this sample
            pos_truth = self.trees[pos_class](probs[i:i+1]).squeeze()  # (1, 1) -> scalar
            truths[i, pos_class] = pos_truth
            
            # Sample and evaluate 1 random negative tree
            neg_class = torch.randint(0, self.num_classes, (1,), device=device).item()
            while neg_class == pos_class and self.num_classes > 1:
                neg_class = torch.randint(0, self.num_classes, (1,), device=device).item()
            
            neg_truth = self.trees[neg_class](probs[i:i+1]).squeeze()  # (1, 1) -> scalar
            truths[i, neg_class] = neg_truth
        
        return truths

    def get_activations(self, x):
        return [self._concepts(x)]

    def forward(self, x, targets=None):
        concept_logits = self._concepts(x)
        class_logits = self._logits_from_concepts(concept_logits, targets)
        return class_logits, None, concept_logits

    def forward_tti(self, x, gt_concepts, allowed_idx):
        concept_logits = self._concepts(x)
        gt_logits = (gt_concepts - 0.5) * 6.0
        concept_logits = concept_logits.clone()
        concept_logits[:, allowed_idx] = gt_logits[:, allowed_idx]
        return self._logits_from_concepts(concept_logits)

    def anneal(self, progress: float) -> None:
        """Advance each tree's native routing schedule (all layouts)."""
        for tree in self.trees:
            tree.anneal_routing(
                progress,
                sinkhorn_temperature=self.sinkhorn_temperature,
                sinkhorn_final_temperature=self.sinkhorn_final_temperature,
            )

    def param_groups(self, base_lr: float, head_lr_mult: float = 1.0):
        if head_lr_mult == 1.0:
            return [{"params": list(self.parameters()), "lr": base_lr}]
        enc_params = list(self.enc.parameters())
        if self.backbone is not None:
            enc_params += list(self.backbone.parameters())
        enc_ids = {id(p) for p in enc_params}
        head_params = [p for p in self.parameters() if id(p) not in enc_ids]
        return [
            {"params": enc_params, "lr": base_lr},
            {"params": head_params, "lr": base_lr * head_lr_mult},
        ]


class StochasticMultiTreeBaconCBM(StochasticMLP):
    """IBE stochastic concept encoder + per-class independent BACON trees.

    The multi-tree analogue of :class:`StochasticBaconCBM`: combines IBE's
    stochastic X->C encoder (inherited verbatim from :class:`StochasticMLP`) with
    a bank of ``num_classes`` independent scalar BACON trees for C->Y, so each
    class learns its own logic structure *and* the encoder is IB-regularized.
    """

    def __init__(
        self,
        arch,
        activation: str = "relu",
        backbone=None,
        aggregator: str = "lsp.full_weight",
        tree_layout: str = "left",
        use_permutation_layer: bool = True,
        use_transformation_layer: bool = True,
        normalize_andness: bool = True,
        sinkhorn_temperature: float = 3.0,
        sinkhorn_final_temperature: float = 0.1,
        concept_dropout: float = 0.0,
        use_negative_sampling: bool = False,
        tree_kwargs=None,
    ):
        super().__init__(arch, activation, backbone=backbone)
        self.num_concepts = arch[-2]
        self.num_classes = arch[-1]
        self.concept_dropout = float(concept_dropout)
        self.use_negative_sampling = use_negative_sampling
        self.sinkhorn_temperature = float(sinkhorn_temperature)
        self.sinkhorn_final_temperature = float(sinkhorn_final_temperature)

        self.cls = None
        self.trees = _build_tree_bank(
            self.num_classes, self.num_concepts, aggregator, tree_layout,
            use_permutation_layer, use_transformation_layer, normalize_andness,
            sinkhorn_temperature, sinkhorn_final_temperature,
            dict(tree_kwargs or {}),
        )

    def _apply(self, fn, recurse: bool = True):
        out = super()._apply(fn, recurse)
        # Properly move trees to the target device
        dev = self.pred_mu[0].weight.device
        for tree in self.trees:
            tree.to(dev)  # This updates tree.device internally
        return out

    def _head(self, z, targets=None):
        probs = torch.sigmoid(z)
        if self.training and self.concept_dropout > 0.0:
            drop = torch.rand_like(probs) < self.concept_dropout
            probs = torch.where(drop, torch.full_like(probs, 0.5), probs)
        
        # Negative sampling: evaluate only positive + 1 random negative tree per sample
        if self.training and self.use_negative_sampling and targets is not None:
            truths = self._evaluate_negative_sampled(probs, targets)
        else:
            # Standard: evaluate all trees
            truths = _bank_truths(self.trees, probs)
        
        truths = truths.clamp(_EPS, 1.0 - _EPS)
        return torch.log(truths) - torch.log1p(-truths)

    def _evaluate_negative_sampled(self, probs, targets):
        """Evaluate only positive class tree and 1 random negative per sample.
        
        For homogeneous per-class batches, evaluate exactly two trees total:
        the positive class tree once on the full batch and one negative tree once
        on the full batch. Fall back to per-sample sampling only for mixed-label
        batches.
        """
        batch_size = probs.shape[0]
        device = probs.device
        truths = torch.zeros(batch_size, self.num_classes, device=device, dtype=probs.dtype)

        if batch_size == 0:
            return truths

        target_classes = targets.reshape(-1).to(device=device, dtype=torch.long)
        if torch.all(target_classes == target_classes[0]):
            pos_class = int(target_classes[0].item())
            neg_class = torch.randint(0, self.num_classes, (1,), device=device).item()
            while neg_class == pos_class and self.num_classes > 1:
                neg_class = torch.randint(0, self.num_classes, (1,), device=device).item()

            truths[:, pos_class] = self.trees[pos_class](probs).squeeze(-1)
            truths[:, neg_class] = self.trees[neg_class](probs).squeeze(-1)
            return truths
        
        for i in range(batch_size):
            pos_class = int(target_classes[i].item())
            
            # Evaluate positive tree for this sample
            pos_truth = self.trees[pos_class](probs[i:i+1]).squeeze()  # (1, 1) -> scalar
            truths[i, pos_class] = pos_truth
            
            # Sample and evaluate 1 random negative tree
            neg_class = torch.randint(0, self.num_classes, (1,), device=device).item()
            while neg_class == pos_class and self.num_classes > 1:
                neg_class = torch.randint(0, self.num_classes, (1,), device=device).item()
            
            neg_truth = self.trees[neg_class](probs[i:i+1]).squeeze()  # (1, 1) -> scalar
            truths[i, neg_class] = neg_truth
        
        return truths

    def forward(self, x, targets=None, return_log_prob=False):
        if self.backbone is not None:
            x = self.backbone(x)
        means = self.pred_mu(x)
        stds = self.pred_sigma(x).clamp(min=1e-7)
        eps = torch.randn_like(means)
        z = means + stds * eps
        if return_log_prob:
            distr = torch.distributions.normal.Normal(means, stds)
            logprob = distr.log_prob(z).sum(dim=1)
            return z, logprob
        return self._head(z, targets), stds.mean().item(), z

    def forward_tti(self, x, gt_concepts, allowed_idx):
        if self.backbone is not None:
            x = self.backbone(x)
        means = self.pred_mu(x)
        stds = self.pred_sigma(x).clamp(min=1e-7)
        eps = torch.randn_like(means)
        z = means + stds * eps
        gt_logits = 2 * (gt_concepts - 0.5)
        gt_logits = gt_logits * (torch.max(z) - torch.mean(z))
        z = z.clone()
        z[:, allowed_idx] = gt_logits[:, allowed_idx]
        return self._head(z)

    def anneal(self, progress: float) -> None:
        for tree in self.trees:
            tree.anneal_routing(
                progress,
                sinkhorn_temperature=self.sinkhorn_temperature,
                sinkhorn_final_temperature=self.sinkhorn_final_temperature,
            )

    def param_groups(self, base_lr: float, head_lr_mult: float = 1.0):
        if head_lr_mult == 1.0:
            return [{"params": list(self.parameters()), "lr": base_lr}]
        enc_params = list(self.pred_mu.parameters()) + list(self.pred_sigma.parameters())
        if self.backbone is not None:
            enc_params += list(self.backbone.parameters())
        enc_ids = {id(p) for p in enc_params}
        head_params = [p for p in self.parameters() if id(p) not in enc_ids]
        return [
            {"params": enc_params, "lr": base_lr},
            {"params": head_params, "lr": base_lr * head_lr_mult},
        ]
