"""Shared training / evaluation loops used by the CLI drivers.

Keeping the loops here lets ``train.py`` and ``evaluate.py`` stay thin and share
identical forward / loss / metric logic.
"""

from __future__ import annotations

import random
from typing import Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .metrics import AverageMeter, concept_accuracy, task_accuracy
from .models.base import CBMOutput


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def make_loaders(bundle, train_cfg) -> Dict[str, DataLoader]:
    batch_size = int(train_cfg.get("batch_size", 64))
    num_workers = int(train_cfg.get("num_workers", 4))
    loaders = {
        "train": DataLoader(
            bundle.train,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        ),
        "test": DataLoader(
            bundle.test,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        ),
    }
    if bundle.val is not None:
        loaders["val"] = DataLoader(
            bundle.val,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )
    return loaders


def compute_losses(
    out: CBMOutput,
    concepts: torch.Tensor,
    labels: torch.Tensor,
    concept_loss_weight: float,
    model: nn.Module | None = None,
    concept_reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (total, concept_loss, task_loss).

    If ``model`` defines a ``task_loss(class_logits, labels)`` method (e.g. the
    one-vs-rest BACON head), it is used for the task term; otherwise the default
    multi-class softmax cross-entropy is applied.

    ``concept_reduction`` selects how the per-concept BCE terms are reduced:

    * ``"mean"`` (default): mean over both batch and concepts. The concept term
      is then ~``1 / n_concepts`` the size of a single summed CE, so a
      ``concept_loss_weight`` of ~5-10 is needed to balance it against the task
      term (the tuned harness configs).
    * ``"sum"``: sum over concepts, mean over batch -- the form written in the
      CBM objective (Koh et al. eq. 6 / the CIBM paper): the task CE and every
      per-concept CE each enter with weight 1. This is what makes
      ``concept_loss_weight: 1.0`` literally mean "weight the CE terms equally"
      and is required for the faithful paper reproduction (otherwise the concept
      head collapses to the all-absent prior because its gradient is ~112x too
      weak).
    """
    if concept_reduction == "sum":
        # Sum the per-concept BCE over concepts, then average over the batch.
        concept_loss = nn.functional.binary_cross_entropy_with_logits(
            out.concept_logits, concepts, reduction="none"
        ).sum(dim=1).mean()
    elif concept_reduction == "mean":
        concept_loss = nn.functional.binary_cross_entropy_with_logits(
            out.concept_logits, concepts
        )
    else:
        raise ValueError(
            f"concept_reduction must be 'mean' or 'sum', got {concept_reduction!r}"
        )
    if model is not None and hasattr(model, "task_loss"):
        task_loss = model.task_loss(out.class_logits, labels)
    else:
        task_loss = nn.functional.cross_entropy(
            out.class_logits, labels, label_smoothing=label_smoothing
        )
    total = task_loss + concept_loss_weight * concept_loss
    # Optional information-bottleneck regulariser (e.g. CIBM). The model
    # returns an already-signed term to add to the total loss.
    if model is not None and hasattr(model, "ib_loss"):
        total = total + model.ib_loss()
    return total, concept_loss, task_loss


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    concept_loss_weight: float,
    optimizer: torch.optim.Optimizer | None = None,
    grad_clip: float | None = None,
    concept_reduction: str = "mean",
    label_smoothing: float = 0.0,
) -> Dict[str, float]:
    """Run one pass. If ``optimizer`` is given, train; otherwise evaluate.

    ``grad_clip`` (max global gradient norm) guards against the gradient
    explosions that cause NaN losses; ``None`` disables clipping.
    ``concept_reduction`` is forwarded to :func:`compute_losses` ("mean" or
    "sum"); use "sum" for the faithful CBM objective. ``label_smoothing``
    (0 disables) is applied to the default cross-entropy task loss only.
    """
    is_train = optimizer is not None
    model.train(is_train)

    loss_m, c_acc_m, t_acc_m = AverageMeter(), AverageMeter(), AverageMeter()

    torch.set_grad_enabled(is_train)
    for images, concepts, labels in loader:
        images = images.to(device, non_blocking=True)
        concepts = concepts.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        out = model(images)
        total, _, _ = compute_losses(
            out, concepts, labels, concept_loss_weight, model, concept_reduction,
            label_smoothing,
        )

        if is_train:
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            if grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            # Constrained-IB models (CIBM dual-beta) adapt their Lagrange
            # multiplier after each optimiser step; no-op for other models.
            if hasattr(model, "ib_dual_step"):
                model.ib_dual_step()

        bs = images.size(0)
        loss_m.update(total.item(), bs)
        c_acc_m.update(concept_accuracy(out.concept_logits, concepts), bs)
        t_acc_m.update(task_accuracy(out.class_logits, labels), bs)
    torch.set_grad_enabled(True)

    return {
        "loss": loss_m.avg,
        "concept_acc": c_acc_m.avg,
        "task_acc": t_acc_m.avg,
    }


@torch.no_grad()
def collect_concepts(model, loader, device, stochastic: bool = False):
    """Gather soft concept predictions and ground-truth concept labels.

    Returns ``(c_soft, c_true)`` as numpy arrays of shape
    ``(n_samples, n_concepts)`` -- ``c_soft`` are sigmoid concept probabilities
    and ``c_true`` the binary concept labels. Used by the concept-purity
    metrics (OIS / NIS).

    With ``stochastic=True`` and a model that exposes ``force_stochastic`` (the
    variational CIBM), concepts are *sampled* from ``q(c | x)`` while the
    backbone still runs deterministically -- i.e. purity is measured on the
    noisy channel rather than the deterministic mean. No effect on models
    without that attribute.
    """
    model.eval()
    had_force = hasattr(model, "force_stochastic")
    prev = getattr(model, "force_stochastic", False)
    if had_force:
        model.force_stochastic = bool(stochastic)
    try:
        soft_chunks, true_chunks = [], []
        for images, concepts, _labels in loader:
            images = images.to(device, non_blocking=True)
            out = model(images)
            probs = torch.sigmoid(out.concept_logits)
            soft_chunks.append(probs.cpu().numpy())
            true_chunks.append(concepts.cpu().numpy())
    finally:
        if had_force:
            model.force_stochastic = prev
    c_soft = np.concatenate(soft_chunks, axis=0)
    c_true = np.concatenate(true_chunks, axis=0)
    return c_soft, c_true
