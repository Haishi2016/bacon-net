"""
BACON-as-concept-layer Concept Bottleneck Model (CBM).

    image --CNN--> concept logits --sigmoid--> concept probs (the bottleneck)
          --fixed human BACON trees--> per-class truths --logit--> class logits

Only the CNN encoder (and a single global temperature) is trainable.  The 10
BACON trees are frozen human structure.  No concept-level supervision is used:
the sole training signal is the digit label, back-propagated through the fixed
symbolic logic, which forces the concept layer to become human-aligned.
"""

from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn as nn

from bacon_logic import BaconLogicBank


class ConceptCNN(nn.Module):
    """Small CNN mapping a 1x28x28 MNIST image to concept logits."""

    def __init__(self, n_concepts: int):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),                       # 14x14
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),                       # 7x7
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, n_concepts),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x))


class ConceptBaconCBM(nn.Module):
    def __init__(
        self,
        concept_names: List[str],
        formulas: Dict[int, str],
        and_andness: float = 1.0,
        or_andness: float = 0.0,
        logit_temperature: float = 4.0,
    ):
        super().__init__()
        self.concept_names = list(concept_names)
        self.encoder = ConceptCNN(len(concept_names))
        self.logic = BaconLogicBank(
            concept_names, formulas, and_andness=and_andness, or_andness=or_andness
        )
        # Single global temperature that sharpens the truth->logit map.
        # Monotonic, shared across classes -> preserves interpretability.
        self.log_temp = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(logit_temperature)))))

    def concept_probs(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.encoder(x))

    def forward(self, x: torch.Tensor):
        probs = self.concept_probs(x)                    # (B, C)
        truths = self.logic(probs)                        # (B, K) in (0,1)
        eps = 1e-6
        truths = truths.clamp(eps, 1.0 - eps)
        logits = self.log_temp.exp() * (torch.log(truths) - torch.log1p(-truths))
        return logits, probs, truths


def binarization_penalty(probs: torch.Tensor) -> torch.Tensor:
    """Encourage crisp (near 0/1) concepts: mean of p*(1-p)."""
    return (probs * (1.0 - probs)).mean()
