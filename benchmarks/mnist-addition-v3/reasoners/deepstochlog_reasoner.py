# mnist-addition-v3/reasoners/deepstochlog_reasoner.py

from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepStochLogAdditionReasoner(nn.Module):
    """
    DeepStochLog-style reasoning for MNIST addition, operating on digit probabilities.

    For this particular task, the semantics of the grammar still boil down to:

        P(sum = k) = sum_{i+j=k} p1[i] * p2[j]

    where p1, p2 are the digit distributions for the two images.

    We then use cross-entropy on this distribution with the true sum label.
    """

    def __init__(self):
        super().__init__()
        # Precompute sum indices: sum_idx[i, j] = i + j
        sum_idx = torch.arange(10).view(10, 1) + torch.arange(10).view(1, 10)
        self.register_buffer("sum_idx", sum_idx)  # [10,10]

    def sum_distribution(self, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            p1, p2: [B,10] digit probabilities

        Returns:
            probs: [B,19] where probs[b, k] = P(sum = k | p1[b], p2[b]).
        """
        # Outer product over digit probabilities
        T = p1.unsqueeze(2) * p2.unsqueeze(1)  # [B,10,10]

        probs_k = []
        for k in range(19):
            mask_k = (self.sum_idx == k)       # [10,10] bool
            p_k = T[:, mask_k].sum(dim=1)      # [B]
            probs_k.append(p_k)

        probs = torch.stack(probs_k, dim=1)    # [B,19]
        # Safety renormalization
        probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-9)
        return probs

    def forward(self, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        """
        Forward: return distribution over sums [B,19].
        """
        return self.sum_distribution(p1, p2)

    def loss(self, p1: torch.Tensor, p2: torch.Tensor, y_sum: torch.Tensor) -> torch.Tensor:
        """
        DeepStochLog-style loss: negative log-likelihood of the correct sum.
        """
        probs = self.sum_distribution(p1, p2)    # [B,19]
        log_probs = torch.log(probs + 1e-9)      # [B,19]
        return F.nll_loss(log_probs, y_sum)

    def predict(self, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        """
        Predict sum label via argmax over P(sum = k).
        """
        probs = self.sum_distribution(p1, p2)
        return probs.argmax(dim=1)
