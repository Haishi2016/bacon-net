# mnist-addition-v3/reasoners/bacon_reasoner.py

import os
import sys
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Make sure local bacon package is importable (same trick as v1)
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)

from bacon.binaryTreeLogicNet import binaryTreeLogicNet
from bacon.aggregators.bool import MinMaxAggregator


def _group_entropy_loss(c_prob: torch.Tensor) -> torch.Tensor:
    """
    Entropy regularization over the two 10-way groups (first digit, second digit).

    c_prob: [B,20] = [p1(0..9), p2(0..9)]
    """
    eps = 1e-8
    g1 = c_prob[:, :10]
    g2 = c_prob[:, 10:]
    ent = -(g1 * (g1 + eps).log()).sum(dim=1) - (g2 * (g2 + eps).log()).sum(dim=1)
    return (ent / math.log(10.0)).mean()


class BaconAdditionReasoner(nn.Module):
    """
    BACON-style graded logic head for MNIST addition, operating on digit probabilities.

    Inputs:
      p1, p2: [B,10] digit probabilities (from the shared CNN).

    Internals:
      - Two binaryTreeLogicNet modules (perm1, perm2) learn permutations/transformations
        of the 10-digit concept probabilities.
      - MinMaxAggregator implements graded AND (a=1.0) over all digit pairs.
      - For each sum k in [0..18], we compute a smooth OR over the subset of pairs (i,j)
        with i + j = k:

          y_k = 1 - ∏_{i+j=k} (1 - AND(p1'[i], p2'[j]))

      producing y_prob: [B,19] approximate probabilities for each sum.

    Loss:
      - Cross-entropy on logits derived from y_prob.
      - Plus a small entropy regularizer over [p1, p2] to avoid brittle collapse.
    """

    def __init__(self, entropy_weight: float = 5e-4):
        super().__init__()
        self.entropy_weight = entropy_weight

        # Two BACON trees to transform/permute the 10-digit distributions
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

        # Precompute mask[k, i, j] = 1 if i + j == k, else 0
        mask = torch.zeros(19, 10, 10)
        for k in range(19):
            for i in range(10):
                j = k - i
                if 0 <= j <= 9:
                    mask[k, i, j] = 1.0
        self.register_buffer("mask", mask)  # [19,10,10]

    def _pair_scores(self, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        """
        Apply BACON permutation trees and graded AND over all digit pairs.

        Args:
          p1, p2: [B,10] digit probabilities

        Returns:
          s: [B,10,10] graded AND scores for all pairs (i,j).
        """
        B = p1.size(0)

        # Learn soft permutations / transformations
        p1p = self.perm1.input_to_leaf(p1)  # [B,10]
        p2p = self.perm2.input_to_leaf(p2)  # [B,10]

        # Broadcast to [B,10,10]
        x = p1p.unsqueeze(2).expand(B, 10, 10)
        y = p2p.unsqueeze(1).expand(B, 10, 10)

        # BACON graded AND via MinMaxAggregator
        a_and = x.new_tensor(1.0)
        s = self.agg.aggregate_tensor(x, y, a_and, w0=0.5, w1=0.5)  # [B,10,10]
        return s

    def _sum_distribution(self, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        """
        Compute BACON-style probabilities over sums 0..18.

        Args:
          p1, p2: [B,10]

        Returns:
          y_prob: [B,19] in [0,1]
        """
        s = self._pair_scores(p1, p2)  # [B,10,10]
        B = s.size(0)

        # Smooth OR per k: y_k = 1 - ∏_{i+j=k} (1 - s_ij)
        one_minus = (1.0 - s.clamp(1e-6, 1 - 1e-6))        # [B,10,10]
        log_one_minus = (one_minus + 1e-12).log()          # [B,10,10]

        y_list = []
        for k in range(19):
            mk = self.mask[k].unsqueeze(0).expand(B, 10, 10)        # [B,10,10]
            log_prod_k = (log_one_minus * mk).sum(dim=(1, 2))       # [B]
            yk = 1.0 - torch.exp(log_prod_k)                        # [B]
            y_list.append(yk.unsqueeze(1))
        y_prob = torch.cat(y_list, dim=1)                           # [B,19]

        # Safety renormalization (optional)
        y_prob = y_prob / (y_prob.sum(dim=1, keepdim=True) + 1e-9)
        return y_prob

    def forward(self, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        """
        Forward: return y_prob over sums [B,19].
        """
        return self._sum_distribution(p1, p2)

    def loss(self, p1: torch.Tensor, p2: torch.Tensor, y_sum: torch.Tensor) -> torch.Tensor:
        """
        Loss = NLL on y_prob + entropy_weight * group_entropy([p1, p2])

        We already interpret y_prob as probabilities, so we use:
            L = - E_b [ log y_prob[b, y_b] ].
        """
        y_prob = self._sum_distribution(p1, p2)               # [B,19]
        # Negative log-likelihood
        log_prob = torch.log(y_prob + 1e-9)                   # [B,19]
        nll = F.nll_loss(log_prob, y_sum)

        # Entropy regularizer over the concept probabilities (digits here)
        c_prob = torch.cat([p1, p2], dim=1)                   # [B,20]
        ent = _group_entropy_loss(c_prob)

        return nll + self.entropy_weight * ent


    def predict(self, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        """
        Predict sum label via argmax over y_prob.
        """
        y_prob = self._sum_distribution(p1, p2)
        return y_prob.argmax(dim=1)
