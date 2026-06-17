"""Metric helpers for CBM evaluation."""

from __future__ import annotations

import torch


@torch.no_grad()
def task_accuracy(class_logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Top-1 classification accuracy."""
    preds = class_logits.argmax(dim=1)
    return (preds == labels).float().mean().item()


@torch.no_grad()
def concept_accuracy(
    concept_logits: torch.Tensor, concepts: torch.Tensor, threshold: float = 0.5
) -> float:
    """Mean per-concept binary accuracy over all concepts and samples."""
    preds = (torch.sigmoid(concept_logits) >= threshold).float()
    return (preds == concepts).float().mean().item()


class AverageMeter:
    """Running average of a scalar, weighted by batch size."""

    def __init__(self) -> None:
        self.sum = 0.0
        self.count = 0

    def update(self, value: float, n: int = 1) -> None:
        self.sum += value * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / self.count if self.count else 0.0
