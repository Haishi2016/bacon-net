# mnist-addition-v4/backbone.py

import torch
import torch.nn as nn
import torch.nn.functional as F


# ------------------------------------------------------------
# Small CNN tower (same as v1)
# ------------------------------------------------------------
class SmallCnn(nn.Module):
    def __init__(self, out_features=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),  # 7x7
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, out_features), nn.ReLU(),
        )

    def forward(self, x):
        return self.fc(self.conv(x))


# ------------------------------------------------------------
# 10-way concept head per image (Gumbel-Softmax)
# ------------------------------------------------------------
class ImageConceptHead(nn.Module):
    def __init__(self, feat_dim: int, n_concepts=10):
        super().__init__()
        self.logit_head = nn.Linear(feat_dim, n_concepts)

    def forward(self, features, tau: float, hard: bool = False) -> torch.Tensor:
        logits = self.logit_head(features)          # [B,10]
        probs = F.gumbel_softmax(
            logits, tau=tau, hard=hard, dim=1
        )                                           # [B,10], sums to 1
        return probs


# ------------------------------------------------------------
# Shared backbone: two towers + two concept heads
# ------------------------------------------------------------
class AdditionConceptBackbone(nn.Module):
    """
    Shared concept backbone for BACON / DCR / etc.

    Given two MNIST images x1, x2 and a temperature tau, produces:
      - p1, p2: [B,10] concept probabilities (one-hot-like when hard=True)
    """

    def __init__(self, feat_dim: int = 128, n_concepts: int = 10):
        super().__init__()
        self.tower1 = SmallCnn(feat_dim)
        self.tower2 = SmallCnn(feat_dim)
        self.head1 = ImageConceptHead(feat_dim, n_concepts)
        self.head2 = ImageConceptHead(feat_dim, n_concepts)

    def forward(self, x1, x2, tau: float, hard: bool = False):
        z1 = self.tower1(x1)
        z2 = self.tower2(x2)
        p1 = self.head1(z1, tau=tau, hard=hard)    # [B,10]
        p2 = self.head2(z2, tau=tau, hard=hard)    # [B,10]
        return p1, p2
