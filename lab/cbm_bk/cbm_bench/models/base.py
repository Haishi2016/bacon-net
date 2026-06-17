"""Shared abstractions for CBM models.

A Concept Bottleneck Model maps ``image -> concept_logits -> class_logits``.
Every model returns a :class:`CBMOutput` so the training loop can compute both
the concept loss and the task loss uniformly.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CBMOutput:
    """Standard forward output for all CBM models.

    Attributes
    ----------
    concept_logits : tensor ``(B, n_concepts)`` pre-sigmoid concept scores.
    class_logits   : tensor ``(B, n_classes)`` pre-softmax task scores.
    """

    concept_logits: torch.Tensor
    class_logits: torch.Tensor
