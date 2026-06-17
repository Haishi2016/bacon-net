"""Baseline (joint) Concept Bottleneck Model.

Architecture (Koh et al., 2020 "Concept Bottleneck Models"):

    image --[backbone CNN]--> features
          --[concept head]--> concept_logits  (n_concepts)
          --[sigmoid]-------> concept_probs
          --[task head]-----> class_logits    (n_classes)

This is the *joint* bottleneck variant: the task head reads only the concept
layer, so all class information must flow through the concepts. Trained with a
weighted sum of concept BCE loss and task cross-entropy.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models

from .base import CBMOutput
from ..registry import register_model


_BACKBONES = {
    "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT),
    "resnet34": (models.resnet34, models.ResNet34_Weights.DEFAULT),
    "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
}


def _make_backbone(name: str, pretrained: bool) -> tuple[nn.Module, int]:
    """Return a feature-extractor backbone and its output feature dim."""
    if name not in _BACKBONES:
        raise KeyError(
            f"Unknown backbone '{name}'. Available: {sorted(_BACKBONES)}"
        )
    ctor, weights = _BACKBONES[name]
    net = ctor(weights=weights if pretrained else None)
    feat_dim = net.fc.in_features
    net.fc = nn.Identity()  # expose pooled features
    return net, feat_dim


class BaselineCBM(nn.Module):
    """Joint concept bottleneck model with a linear task head."""

    def __init__(
        self,
        n_concepts: int,
        n_classes: int,
        backbone: str = "resnet18",
        pretrained: bool = True,
        bottleneck: bool = True,
        concept_activation: str = "sigmoid",
    ):
        super().__init__()
        self.bottleneck = bottleneck
        # Activation applied to the concept logits before the task head reads
        # them. ``"sigmoid"`` (default, legacy) feeds concept *probabilities*;
        # ``"identity"`` feeds the raw soft concept *logits*, which is what the
        # CIBM paper does (App. D.1: "on top of concept logits we stack the
        # label predictor"; D.7: soft, non-binary concepts). The sigmoid gate
        # saturates on sparse-attribute data (CUB positives ~0.20 -> negative
        # logits -> sigmoid~0 -> near-constant task input and a vanished task
        # back-gradient), which collapses the soft-joint baseline; the identity
        # head avoids that.
        act = concept_activation.lower()
        if act not in ("sigmoid", "identity", "none"):
            raise ValueError(
                f"concept_activation must be 'sigmoid' or 'identity', got {act!r}"
            )
        self.concept_activation = "identity" if act == "none" else act
        self.backbone, feat_dim = _make_backbone(backbone, pretrained)
        self.concept_head = nn.Linear(feat_dim, n_concepts)
        # In a true bottleneck the task head sees only the concept layer.
        task_in = n_concepts if bottleneck else feat_dim
        self.task_head = nn.Linear(task_in, n_classes)

    def forward(self, x: torch.Tensor) -> CBMOutput:
        feats = self.backbone(x)
        concept_logits = self.concept_head(feats)
        if self.bottleneck:
            if self.concept_activation == "sigmoid":
                concept_feats = torch.sigmoid(concept_logits)
            else:
                concept_feats = concept_logits
            class_logits = self.task_head(concept_feats)
        else:
            class_logits = self.task_head(feats)
        return CBMOutput(concept_logits=concept_logits, class_logits=class_logits)


@register_model("baseline_cbm")
def build_baseline_cbm(cfg, n_concepts: int, n_classes: int) -> BaselineCBM:
    """Factory used by the training driver. ``cfg`` is the model config."""
    return BaselineCBM(
        n_concepts=n_concepts,
        n_classes=n_classes,
        backbone=cfg.get("backbone", "resnet18"),
        pretrained=bool(cfg.get("pretrained", True)),
        bottleneck=bool(cfg.get("bottleneck", True)),
        concept_activation=str(cfg.get("concept_activation", "sigmoid")),
    )
