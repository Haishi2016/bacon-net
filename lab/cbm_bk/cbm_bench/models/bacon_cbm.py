"""One-vs-rest BACON Concept Bottleneck Model.

Architecture::

    image --[backbone CNN]--> features
          --[concept head]--> concept_logits (n_concepts)
          --[sigmoid]-------> concept_probs  in [0, 1]
          --[200 BACON heads, one per class] --> per-class truth values
          --[stack]---------> class_logits (n_classes)

Each class gets its **own** ``binaryTreeLogicNet`` graded-logic head that reads
the *shared* concept bottleneck and outputs a single truth value in ``[0, 1]``
("is this bird class *k*?"). The heads are trained **one-vs-rest** with binary
cross-entropy against the one-hot label, so every head must learn a concise
logical rule over the concepts for its species. Because all heads share the one
concept layer, this acts as a strong regularizer pushing the concepts toward
clear, reusable predicates. Prediction is ``argmax`` over the 200 truth values.

For speed, set ``head_type='vector'`` to evaluate all ``n_classes`` rules in a
single batched pass through one vectorized ``binaryTreeLogicNet`` instead of
looping over 200 serial heads. The vector path supports the ``left``, ``full``,
and ``alternating`` tree layouts (``gl.generic`` semantics).

Note: 200 logic trees over ~100 concepts is compute-heavy; prefer a GPU and a
moderate ``n_concepts`` (see ``min_class_count`` in the CUB dataset config).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from bacon.binaryTreeLogicNet import binaryTreeLogicNet
from bacon.baconNet import _aggregator_registry
from bacon.frozonInputToLeaf import frozenInputToLeaf
from bacon.transformationLayer import (
    IdentityTransformation,
    NegationTransformation,
)

from .base import CBMOutput
from .baseline_cbm import _make_backbone  # reuse backbone factory (unmodified)
from ..registry import register_model

# Truth values are clamped before the logit transform for numerical stability.
_EPS = 1e-6


def _make_aggregator(name: str):
    """Instantiate a *fresh* aggregator object from its registry name.

    Each BACON head needs its own aggregator instance because ``attach_to_tree``
    creates per-node parameters.
    """
    if name not in _aggregator_registry:
        raise KeyError(
            f"Unknown aggregator '{name}'. Available: {sorted(_aggregator_registry)}"
        )
    return _aggregator_registry[name]()


class BaconCBM(nn.Module):
    """Shared concept bottleneck with one BACON head per class (one-vs-rest)."""

    def __init__(
        self,
        n_concepts: int,
        n_classes: int,
        backbone: str = "resnet34",
        pretrained: bool = True,
        aggregator: str = "gl.generic",
        tree_layout: str = "alternating",
        weight_mode: str = "trainable",
        normalize_andness: bool = True,
        use_permutation_layer: bool = False,
        use_transformation_layer: bool = False,
        sinkhorn_temperature: float = 3.0,
        sinkhorn_final_temperature: float = 0.1,
        head_type: str = "binary",
        concept_dropout: float = 0.0,
    ):
        super().__init__()
        self.n_concepts = n_concepts
        self.n_classes = n_classes
        self.head_type = head_type
        self.concept_dropout = float(concept_dropout)

        self.backbone, feat_dim = _make_backbone(backbone, pretrained)
        self.concept_head = nn.Linear(feat_dim, n_concepts)

        if head_type == "vector":
            # A single vectorized logic net evaluates all ``n_classes`` rules in
            # one batched pass (much faster than 200 serial heads). Supports the
            # left, full, and alternating layouts (gl.generic semantics).
            self.heads = None
            self.vector_net = binaryTreeLogicNet(
                input_size=n_concepts,
                aggregator=_make_aggregator(aggregator),
                tree_layout=tree_layout,
                weight_mode=weight_mode,
                normalize_andness=normalize_andness,
                use_permutation_layer=use_permutation_layer,
                use_transformation_layer=use_transformation_layer,
                head_type="vector",
                num_heads=n_classes,
            )
            # Configure the LSP tree head's Sinkhorn annealing schedule (broad
            # exploration -> sharp near-permutation). Endpoints match BACON's
            # full/alternating tree (3.0 -> 0.1); exposed so the schedule can be
            # tuned to the training budget (e.g. a lower start sharpens sooner).
            tree_head = getattr(self.vector_net, "vector_head", None)
            if tree_head is not None and hasattr(tree_head, "sinkhorn_final_temperature"):
                tree_head.sinkhorn_temperature = float(sinkhorn_temperature)
                tree_head.sinkhorn_final_temperature = float(sinkhorn_final_temperature)
                tree_head.temperature.fill_(float(sinkhorn_temperature))
        else:
            # One graded-logic head per class, each reading the full concept layer.
            self.vector_net = None
            self.heads = nn.ModuleList(
                binaryTreeLogicNet(
                    input_size=n_concepts,
                    aggregator=_make_aggregator(aggregator),
                    tree_layout=tree_layout,
                    weight_mode=weight_mode,
                    normalize_andness=normalize_andness,
                    use_permutation_layer=use_permutation_layer,
                )
                for _ in range(n_classes)
            )

    def _apply(self, fn, recurse: bool = True):
        # ``binaryTreeLogicNet`` caches a ``.device`` used to allocate tensors
        # during forward; keep it in sync whenever the module is moved
        # (e.g. ``model.to(device)``) so heads match the input device.
        out = super()._apply(fn, recurse)
        device = self.concept_head.weight.device
        if self.heads is not None:
            for head in self.heads:
                head.device = device
        if self.vector_net is not None:
            self.vector_net.device = device
        return out

    @torch.no_grad()
    def init_head_gates_from_prototypes(
        self, prototypes: torch.Tensor, scale: float = 8.0
    ) -> bool:
        """Warm-start the per-class relevance gates from concept prototypes.

        The vectorized head gives every class the *same* concept vector, so all
        heads init to nearly identical outputs (gates ``g approx 0.5`` -> truth
        ``approx 0.5``); ``argmax`` is then effectively constant and task accuracy
        collapses to ``1 / n_classes``. Gradient descent cannot break this
        symmetric saddle (the gated min/max over many concepts saturate near
        ``0.5`` with vanishing gradient).

        Seeding each head's gates from its class's concept signature
        ``g[k, i] = sigmoid(scale * (prototype[k, i] - 0.5))`` places every head
        near the explainable rule "class ``k`` = graded-AND of the concepts that
        characterize class ``k``", which both fixes the optimization and makes
        the learned per-class rules interpretable from the start.

        Only affects the vectorized head with per-concept gates
        (``head_type='vector'`` + ``use_permutation_layer=True``). Returns
        ``True`` if gates were initialized, ``False`` otherwise.
        """
        if self.vector_net is None:
            return False
        head = getattr(self.vector_net, "vector_head", None)
        if head is None or getattr(head, "weight_logits", None) is None:
            return False
        if not getattr(head, "use_input_weights", False):
            # Only the gl.generic anchor head exposes per-concept relevance gates
            # shaped (n_classes, n_concepts). The LSP tree head parameterizes
            # routing differently, so prototype gate warm-start does not apply.
            return False
        if prototypes.shape != head.weight_logits.shape:
            raise ValueError(
                f"prototypes shape {tuple(prototypes.shape)} must match "
                f"weight_logits {tuple(head.weight_logits.shape)} "
                f"(n_classes, n_concepts)."
            )
        p = prototypes.to(head.weight_logits.device).clamp(0.0, 1.0)
        head.weight_logits.copy_(scale * (p - 0.5))
        return True

    def anneal(self, progress: float) -> None:
        """Advance the BACON head's exploration schedule (no-op for non-vector).

        ``progress`` runs 0.0 -> 1.0 over training. For the vectorized LSP tree
        head this anneals the soft-permutation Sinkhorn temperature (broad
        exploration -> sharp near-permutation) and the Gumbel noise, matching
        BACON's scalar-tree schedule. Without it the routing stays near-uniform
        and the head is input-independent (task accuracy stuck at chance).
        """
        net = self.vector_net
        if net is not None and hasattr(net, "anneal_vector_tree"):
            net.anneal_vector_tree(progress)

    # -- per-class tree extraction ----------------------------------------
    @torch.no_grad()
    def extract_head_as_scalar_tree(self, k: int) -> binaryTreeLogicNet:
        """Reconstruct class ``k``'s rule as a standalone scalar ``binaryTreeLogicNet``.

        The vectorized ``VectorTreeLogicHead`` stores all ``n_classes`` rules in
        batched tensors (``bias`` / ``weight_logits`` / ``perm_logits`` /
        ``transform_logits`` with a leading head axis), so the existing
        single-tree visualization / analysis helpers
        (:func:`bacon.visualization.visualize_tree_structure`,
        :func:`bacon.utils.export_tree_structure_to_json`, ...) cannot read it
        directly. This copies head ``k``'s slice into a fresh scalar left-fold
        tree those tools understand:

        * ``bias[k]``            -> ``tree.biases``      (same ``sigmoid*3-1`` andness)
        * ``weight_logits[k]``   -> ``tree.weights``     (``weight_normalization='softmax'``
          so the scalar tree applies the *same* softmax over pair weights as the
          vector head)
        * ``perm_logits[k]``     -> a frozen hard permutation (Hungarian assignment
          on the annealed Sinkhorn doubly-stochastic matrix) stored as
          ``tree.input_to_leaf`` (``frozenInputToLeaf``) + ``tree.locked_perm``
        * ``transform_logits[k]``-> a 2-way identity/negation ``TransformationLayer``

        Returns a CPU-resident, eval-mode scalar tree. Only valid for the
        vectorized LSP tree head (``head_type='vector'`` + ``lsp.*`` aggregator).
        """
        if self.vector_net is None:
            raise RuntimeError(
                "extract_head_as_scalar_tree requires head_type='vector'."
            )
        head = getattr(self.vector_net, "vector_head", None)
        if head is None or not hasattr(head, "_F_many") and not hasattr(head, "bias"):
            raise RuntimeError(
                "The vector head does not expose an LSP tree to extract "
                "(expected a VectorTreeLogicHead)."
            )
        if not (0 <= k < self.n_classes):
            raise IndexError(f"class index {k} out of range [0, {self.n_classes}).")

        n = head.input_size
        use_tl = getattr(head, "transform_logits", None) is not None
        transforms = (
            [IdentityTransformation(n), NegationTransformation(n)] if use_tl else None
        )

        tree = binaryTreeLogicNet(
            input_size=n,
            aggregator=type(head.aggregator)(),  # fresh stateless LSP aggregator
            tree_layout="left",
            weight_mode="trainable",
            weight_normalization="softmax",  # match the head's softmax pair weights
            normalize_andness=bool(head.normalize_andness),
            use_permutation_layer=False,  # frozen permutation attached below
            use_transformation_layer=use_tl,
            transformations=transforms,
            device=torch.device("cpu"),
        )
        tree.eval()

        # Per-node andness bias and pair-weight logits (head k's slice).
        if head.bias is not None:
            for i in range(tree.num_layers):
                tree.biases[i].data = head.bias[k, i].detach().cpu().reshape(1).clone()
                tree.weights[i].data = head.weight_logits[k, i].detach().cpu().clone()

        # Frozen hard input->leaf permutation. Harden the annealed Sinkhorn
        # routing for head k into a true permutation via the Hungarian
        # assignment (guarantees a bijection even if a row argmax collides).
        # ``assignment[l]`` is the concept index routed into leaf ``l``.
        assignment = list(range(n))
        if getattr(head, "perm_logits", None) is not None:
            was_training = head.training
            head.eval()
            P = head._sinkhorn(head.perm_logits.detach())[k].cpu()
            if was_training:
                head.train()
            from scipy.optimize import linear_sum_assignment

            rows, cols = linear_sum_assignment(-P.numpy())
            assignment = [int(c) for _, c in sorted(zip(rows.tolist(), cols.tolist()))]
            tree.input_to_leaf = frozenInputToLeaf(assignment, n).to(tree.device)
            tree.locked_perm = torch.tensor(assignment, dtype=torch.long)
            tree.is_frozen = True

        # Identity/negation gate. The vector head applies the gate in *concept*
        # space (before its permutation); the scalar tree applies it in *leaf*
        # space (after ``input_to_leaf``). Reorder the per-concept logits into
        # leaf order so leaf ``l`` receives the gate the vector head applied to
        # concept ``assignment[l]``. Column order matches (0=identity, 1=negation).
        if use_tl:
            src = head.transform_logits[k].detach().cpu()
            tree.transformation_layer.logits.data = src[assignment].clone()
            # Match the head's (annealed) gate softmax temperature so the scalar
            # tree's forward reproduces the head; the argmax used by the viz
            # tooling is temperature-invariant either way.
            tree.transformation_layer.temperature = float(head.transform_temperature)

        return tree

    @torch.no_grad()
    def extract_all_scalar_trees(self):
        """Extract every class's rule as a scalar tree (list indexed by class)."""
        return [self.extract_head_as_scalar_tree(k) for k in range(self.n_classes)]

    def forward(self, x: torch.Tensor) -> CBMOutput:
        concept_logits = self.concept_head(self.backbone(x))
        concept_probs = torch.sigmoid(concept_logits)  # BACON inputs in [0, 1]

        # Neutral concept dropout (head regularization): during training, replace
        # a fraction of concept truths with 0.5 -- the graded-logic "unknown"
        # value -- before the logic head. The head's hardened per-class routing
        # (the ~1.7M-param permutation that drives the train/test gap) is forced
        # to stay robust to any single concept instead of memorizing the exact
        # concept signatures of the ~30 training images/class. Concept
        # supervision is unaffected: the concept loss uses ``concept_logits``
        # (pre-dropout), so concept accuracy is preserved.
        head_probs = concept_probs
        if self.training and self.concept_dropout > 0.0:
            drop = torch.rand_like(head_probs) < self.concept_dropout
            head_probs = torch.where(
                drop, torch.full_like(head_probs, 0.5), head_probs
            )

        if self.vector_net is not None:
            # Single batched pass -> (B, n_classes) truth values.
            truths = self.vector_net(head_probs)
        else:
            # Each head returns (B, 1) truth values; stack to (B, n_classes).
            truths = torch.cat([head(head_probs) for head in self.heads], dim=1)

        # Logit-transform so argmax/eval and an optional softmax fallback work;
        # argmax over logits == argmax over truth values.
        truths = truths.clamp(_EPS, 1.0 - _EPS)
        class_logits = torch.log(truths) - torch.log1p(-truths)
        return CBMOutput(concept_logits=concept_logits, class_logits=class_logits)

    def param_groups(self, base_lr: float, head_lr_mult: float = 1.0):
        """Split parameters so the fresh heads can train faster than the backbone.

        The backbone is pretrained and wants a small learning rate, but the
        concept head and the BACON logic head(s) are randomly initialized and
        need a much larger one -- sharing a single LR forces a compromise that
        leaves the logic head badly undertrained (task accuracy crawls while
        concept accuracy is already high). Returns optimizer param groups with
        the backbone at ``base_lr`` and every other parameter (concept head +
        logic heads) at ``base_lr * head_lr_mult``. ``head_lr_mult == 1.0``
        reproduces a single shared learning rate.
        """
        backbone_params = list(self.backbone.parameters())
        backbone_ids = {id(p) for p in backbone_params}
        head_params = [
            p for p in self.parameters() if id(p) not in backbone_ids
        ]
        return [
            {"params": backbone_params, "lr": base_lr},
            {"params": head_params, "lr": base_lr * head_lr_mult},
        ]

    def task_loss(self, class_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Balanced one-vs-rest task loss.

        Each sample contributes its single **positive** head (the true class)
        weighted equally against the **average** of its ``n_classes - 1``
        negative heads::

            loss = mean_batch[ BCE(positive head) + mean(BCE(negative heads)) ]

        Why not a plain mean/sum over heads:

        * Plain ``mean`` over ``batch x n_classes`` divides every head's gradient
          by ``n_classes``; cold-started heads barely move and ``argmax``
          collapses to a constant class (accuracy ~ ``1 / n_classes``).
        * Plain ``sum`` over heads restores per-head gradient but, with only
          ~``batch / n_classes`` positives per head per batch, the loss is
          dominated by the many negatives: every head learns to predict
          "negative" (outputs collapse toward 0) and accuracy stays at chance.
          It also scales the task term ~``n_classes`` x larger than the concept
          term, starving concept learning.

        Balancing the single positive against the mean negative removes the
        positive/negative imbalance *and* keeps the loss magnitude O(1) and
        independent of ``n_classes``, so it stays comparable to the concept loss
        for any class count.

        Detected by the engine via ``hasattr(model, 'task_loss')``; models
        without it (e.g. the baseline) fall back to softmax cross-entropy.
        """
        targets = F.one_hot(labels, num_classes=self.n_classes).float()
        bce = F.binary_cross_entropy_with_logits(
            class_logits, targets, reduction="none"
        )  # (batch, n_classes)
        pos = (bce * targets).sum(dim=1)  # the single true-class head
        if self.n_classes > 1:
            neg = (bce * (1.0 - targets)).sum(dim=1) / (self.n_classes - 1)
        else:
            neg = torch.zeros_like(pos)
        return (pos + neg).mean()


def save_bacon_trees(
    model: BaconCBM,
    out_dir,
    concept_names=None,
    class_names=None,
):
    """Materialize every class's logic tree next to a checkpoint.

    For each of the ``n_classes`` heads this extracts the rule as a scalar
    ``binaryTreeLogicNet`` (see :meth:`BaconCBM.extract_head_as_scalar_tree`) and
    serializes it with the **existing** analysis helper
    :func:`bacon.utils.save_tree_structure_to_json`, so the saved trees load
    straight into the standard visualization / analysis tooling. Writes one
    ``class_<idx>[_<name>].json`` per class under ``out_dir`` and returns the
    list of written paths. No-op (returns ``[]``) for non-vector heads.
    """
    import os
    import re

    from bacon.utils import save_tree_structure_to_json

    if model.vector_net is None:
        return []

    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for k in range(model.n_classes):
        tree = model.extract_head_as_scalar_tree(k)
        if class_names is not None and k < len(class_names):
            safe = re.sub(r"[^0-9A-Za-z._-]+", "_", str(class_names[k])).strip("_")
            fname = f"class_{k:03d}_{safe}.json"
        else:
            fname = f"class_{k:03d}.json"
        path = os.path.join(out_dir, fname)
        save_tree_structure_to_json(tree, path, feature_names=concept_names)
        paths.append(path)
    return paths


@register_model("bacon_cbm")
def build_bacon_cbm(cfg, n_concepts: int, n_classes: int) -> BaconCBM:
    """Factory used by the training driver. ``cfg`` is the model config."""
    return BaconCBM(
        n_concepts=n_concepts,
        n_classes=n_classes,
        backbone=cfg.get("backbone", "resnet34"),
        pretrained=bool(cfg.get("pretrained", True)),
        aggregator=cfg.get("aggregator", "lsp.full_weight"),
        tree_layout=cfg.get("tree_layout", "left"),
        weight_mode=cfg.get("weight_mode", "trainable"),
        normalize_andness=bool(cfg.get("normalize_andness", True)),
        use_permutation_layer=bool(cfg.get("use_permutation_layer", True)),
        use_transformation_layer=bool(cfg.get("use_transformation_layer", False)),
        sinkhorn_temperature=float(cfg.get("sinkhorn_temperature", 3.0)),
        sinkhorn_final_temperature=float(cfg.get("sinkhorn_final_temperature", 0.1)),
        head_type=cfg.get("head_type", "binary"),
        concept_dropout=float(cfg.get("concept_dropout", 0.0)),
    )
