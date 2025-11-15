# mnist-addition-v3/reasoners/deepproblog_reasoner.py

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepProbLogAdditionReasoner(nn.Module):
    """
    DeepProbLog-style reasoning over digit probabilities.

    Given:
      p1: [B,10] digit probabilities for first image
      p2: [B,10] digit probabilities for second image

    We compute:
      P(sum = k) = sum_{i+j=k} p1[i] * p2[j], for k in [0..18]

    and use cross-entropy loss on this distribution.
    """

    def __init__(self):
        super().__init__()
        # Precompute sum indices: sum_idx[i, j] = i + j
        sum_idx = torch.arange(10).view(10, 1) + torch.arange(10).view(1, 10)
        self.register_buffer("sum_idx", sum_idx)  # [10,10]

    def sum_distribution(self, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            p1, p2: [B,10] (probabilities) – must be non-negative and sum to 1 per row.

        Returns:
            probs: [B,19] where probs[b, k] = P(sum = k | p1[b], p2[b]).
        """
        # Outer product: T[b, i, j] = p1[b, i] * p2[b, j]
        T = p1.unsqueeze(2) * p2.unsqueeze(1)  # [B,10,10]

        probs_k = []
        for k in range(19):
            mask_k = (self.sum_idx == k)          # [10,10] bool
            # Select entries where i + j == k and sum them
            p_k = T[:, mask_k].sum(dim=1)        # [B]
            probs_k.append(p_k)

        probs = torch.stack(probs_k, dim=1)      # [B,19]
        # Safety renormalization
        probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-9)
        return probs

    def forward(self, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: returns [B,19] probabilities over sums.
        """
        return self.sum_distribution(p1, p2)

    def loss(self, p1: torch.Tensor, p2: torch.Tensor, y_sum: torch.Tensor) -> torch.Tensor:
        """
        DeepProbLog-like loss: negative log-likelihood of the correct sum.
        """
        probs = self.sum_distribution(p1, p2)      # [B,19]
        log_probs = torch.log(probs + 1e-9)        # [B,19]
        loss = F.nll_loss(log_probs, y_sum)
        return loss

    def predict(self, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        """
        Predict sum label via argmax of P(sum=k).
        """
        probs = self.sum_distribution(p1, p2)
        return probs.argmax(dim=1)
