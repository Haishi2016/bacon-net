# mnist-addition-v4/reasoners/bacon_reasoner.py

import os
import sys
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure local bacon package is used
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
)

from bacon.binaryTreeLogicNet import binaryTreeLogicNet
from bacon.aggregators.bool import MinMaxAggregator


def group_entropy_loss(c_prob: torch.Tensor) -> torch.Tensor:
    """
    Same as v1:
      c_prob: [B,20] = [p1(0..9), p2(0..9)]
    """
    eps = 1e-8
    g1 = c_prob[:, :10]
    g2 = c_prob[:, 10:]
    ent = -(g1 * (g1 + eps).log()).sum(dim=1) - (g2 * (g2 + eps).log()).sum(dim=1)
    return (ent / math.log(10.0)).mean()


class BaconAdditionReasoner(nn.Module):
    """
    Multiclass BACON head (19 classes) operating on concept probabilities.

    Inputs:
      p1, p2: [B,10] concept probabilities (from AdditionConceptBackbone).

    Internals:
      - Two binaryTreeLogicNet modules (perm1, perm2) learn transformations /
        permutations of the 10 concepts per image.
      - MinMaxAggregator implements graded AND over all digit pairs.
      - Smooth OR per sum k: 1 - prod_{i+j=k} (1 - AND_ij).
    """

    def __init__(self):
        super().__init__()
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

    # ---------------- internal helpers ----------------

    def _pair_scores(self, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        """
        Apply BACON permutation trees and graded AND over all digit pairs.

        Args:
          p1, p2: [B,10] concept probabilities

        Returns:
          s: [B,10,10] graded AND scores for all pairs (i,j).
        """
        B = p1.size(0)
        p1p = self.perm1.input_to_leaf(p1)  # [B,10]
        p2p = self.perm2.input_to_leaf(p2)  # [B,10]

        x = p1p.unsqueeze(2).expand(B, 10, 10)
        y = p2p.unsqueeze(1).expand(B, 10, 10)

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
        s = self._pair_scores(p1, p2)                 # [B,10,10]
        B = s.size(0)

        one_minus = (1.0 - s.clamp(1e-6, 1 - 1e-6))   # [B,10,10]
        log_one_minus = (one_minus + 1e-12).log()     # [B,10,10]

        y_list = []
        for k in range(19):
            mk = self.mask[k].unsqueeze(0).expand(B, 10, 10)  # [B,10,10]
            log_prod_k = (log_one_minus * mk).sum(dim=(1, 2)) # [B]
            yk = 1.0 - torch.exp(log_prod_k)                  # [B]
            y_list.append(yk.unsqueeze(1))

        y_prob = torch.cat(y_list, dim=1)                     # [B,19]
        # Renormalize (not strictly required but stabilizes)
        y_prob = y_prob / (y_prob.sum(dim=1, keepdim=True) + 1e-9)
        return y_prob

    # ---------------- public interface for v4 ----------------

    def forward(self, p1: torch.Tensor, p2: torch.Tensor):
        """
        Forward: returns y_prob over sums [B,19].
        """
        return self._sum_distribution(p1, p2)

    def loss(self, p1: torch.Tensor, p2: torch.Tensor, y_sum: torch.Tensor,
             entropy_weight: float) -> torch.Tensor:
        """
        Same loss structure as v1:
          - main CE on logits derived from y_prob,
          - plus entropy regularizer on concept probabilities.
        """
        y_prob = self._sum_distribution(p1, p2)                 # [B,19]
        logits = torch.logit(y_prob.clamp(1e-6, 1 - 1e-6))      # [B,19]
        ce = nn.CrossEntropyLoss()(logits, y_sum)

        c_prob = torch.cat([p1, p2], dim=1)                     # [B,20]
        ent = group_entropy_loss(c_prob) * entropy_weight

        return ce + ent

    def predict(self, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        """
        Predict sum label via argmax over y_prob.
        """
        y_prob = self._sum_distribution(p1, p2)
        return y_prob.argmax(dim=1)
