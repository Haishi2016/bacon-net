# mnist-addition-v4/bacon_model_v4.py

import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure local bacon package is used instead of any installed site-package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from bacon.binaryTreeLogicNet import binaryTreeLogicNet
from bacon.aggregators.bool import MinMaxAggregator


# ------------------------------------------------------------
# Small CNN tower (same as v1)
# ------------------------------------------------------------
class SmallCnn(nn.Module):
    def __init__(self, out_features=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
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
        logits = self.logit_head(features)
        probs = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=1)  # [B,10], sum=1
        return probs


# ------------------------------------------------------------
# Shared entropy utility (same as v1)
# ------------------------------------------------------------
def group_entropy_loss(c_prob):
    eps = 1e-8
    g1 = c_prob[:, :10]
    g2 = c_prob[:, 10:]
    ent = -(g1 * (g1 + eps).log()).sum(dim=1) - (g2 * (g2 + eps).log()).sum(dim=1)
    return (ent / math.log(10)).mean()


# ------------------------------------------------------------
# BACON model (almost identical to v1 BaconMultiHead)
# ------------------------------------------------------------
class BaconModelV4(nn.Module):
    """
    This is effectively the original BaconMultiHead from v1:

      x1,x2 -> SmallCnn towers -> Gumbel concept heads -> p1,p2
           -> BACON perm trees + MinMaxAggregator -> pair scores
           -> smooth OR over pairs (i+j = k) -> y_prob over sums

    forward(x1,x2,tau,hard) returns:
      y_prob: [B,19]  (sum probabilities)
      c_prob: [B,20]  (concept probabilities [p1,p2] for entropy regularization)

    Exposes attributes:
      tower1, tower2, head1, head2, perm1, perm2, agg
    so that permutation search code works exactly as in v1.
    """

    def __init__(self):
        super().__init__()
        self.tower1 = SmallCnn(128)
        self.tower2 = SmallCnn(128)
        self.head1 = ImageConceptHead(128, 10)
        self.head2 = ImageConceptHead(128, 10)

        # Two small BACON trees (10 leaves each) to learn permutations for p1 and p2
        self.perm1 = binaryTreeLogicNet(
            input_size=10,
            is_frozen=False,
            weight_mode="fixed",
            weight_normalization="minmax",
            aggregator=MinMaxAggregator(),
            normalize_andness=False,
            tree_layout="paired",
            loss_amplifier=1.0,
        )
        self.perm2 = binaryTreeLogicNet(
            input_size=10,
            is_frozen=False,
            weight_mode="fixed",
            weight_normalization="minmax",
            aggregator=MinMaxAggregator(),
            normalize_andness=False,
            tree_layout="paired",
            loss_amplifier=1.0,
        )
        self.agg = MinMaxAggregator()

        # Precompute mask[k, i, j] = 1 if i + j == k
        mask = torch.zeros(19, 10, 10)
        for k in range(19):
            for i in range(10):
                j = k - i
                if 0 <= j <= 9:
                    mask[k, i, j] = 1.0
        self.register_buffer("mask", mask)  # [19,10,10]

    def forward(self, x1, x2, tau: float, hard: bool = False):
        """
        Returns:
          y_prob: [B,19]
          c_prob: [B,20] = [p1,p2]
        """
        # CNN towers + Gumbel concept heads
        z1 = self.tower1(x1)
        z2 = self.tower2(x2)
        p1 = self.head1(z1, tau=tau, hard=hard)  # [B,10]
        p2 = self.head2(z2, tau=tau, hard=hard)  # [B,10]

        # Learn soft/hard permutations via BACON trees' input_to_leaf
        p1p = self.perm1.input_to_leaf(p1)  # [B,10]
        p2p = self.perm2.input_to_leaf(p2)  # [B,10]

        B = p1p.size(0)
        x = p1p.unsqueeze(2).expand(B, 10, 10)
        y = p2p.unsqueeze(1).expand(B, 10, 10)

        # Pairwise AND via MinMaxAggregator
        a_and = torch.tensor(1.0, device=x.device)
        s = self.agg.aggregate_tensor(x, y, a_and, w0=0.5, w1=0.5)  # [B,10,10]

        # Smooth OR per class k: 1 - prod(1 - s_ij) for i+j=k
        one_minus = (1.0 - s.clamp(1e-6, 1 - 1e-6))
        log_one_minus = (one_minus + 1e-12).log()

        y_list = []
        for k in range(19):
            mk = self.mask[k].unsqueeze(0).expand(B, 10, 10)
            log_prod_k = (log_one_minus * mk).sum(dim=(1, 2))
            yk = 1.0 - torch.exp(log_prod_k)
            y_list.append(yk.unsqueeze(1))
        y_prob = torch.cat(y_list, dim=1)  # [B,19]

        # Final normalization (for stability)
        y_prob = y_prob / (y_prob.sum(dim=1, keepdim=True) + 1e-9)

        c_prob = torch.cat([p1, p2], dim=1)  # [B,20]
        return y_prob, c_prob
