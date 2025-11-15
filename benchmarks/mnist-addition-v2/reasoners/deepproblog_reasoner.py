# mnist-addition-v2/reasoners/deepproblog_reasoner.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class DeepProbLogAdditionReasoner(nn.Module):
    """
    DeepProbLog-style reasoning for MNIST addition, operating on digit probabilities.

    For each sample:
      p1: [B,10] digit probs for first image
      p2: [B,10] digit probs for second image

    We compute:
      P(sum = k) = sum_{i+j=k} p1[i] * p2[j]

    Loss: cross-entropy between this distribution and the true sum label.
    """

    def __init__(self):
        super().__init__()
        # Precompute sum_idx[i,j] = i + j
        sum_idx = torch.arange(10).view(10, 1) + torch.arange(10).view(1, 10)
        self.register_buffer("sum_idx", sum_idx)  # [10,10]

    def sum_distribution(self, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        """
        Compute P(sum=k) for k in [0..18].

        Args:
          p1, p2: [B,10] digit probabilities

        Returns:
          probs: [B,19], each row sums to 1
        """
        # Product t-norm per digit pair
        T = p1.unsqueeze(2) * p2.unsqueeze(1)   # [B,10,10]

        probs_k = []
        for k in range(19):
            mask_k = (self.sum_idx == k)       # [10,10] bool
            p_k = (T[:, mask_k]).sum(dim=1)    # [B]
            probs_k.append(p_k)
        probs = torch.stack(probs_k, dim=1)    # [B,19]

        # This already sums to 1 over k, but we can be safe and renorm
        probs = probs / (probs.sum(dim=1, keepdim=True) + 1e-9)
        return probs

    def forward(self, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        """
        Forward = return sum distribution [B,19].
        """
        return self.sum_distribution(p1, p2)

    def loss(self, p1: torch.Tensor, p2: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """
        DeepProbLog-style loss: Cross-entropy of P(sum | images) vs true sum label.
        """
        probs = self.sum_distribution(p1, p2)        # [B,19]
        # Cross-entropy needs logits; use log-probs here.
        log_probs = torch.log(probs + 1e-9)          # [B,19]
        loss = F.nll_loss(log_probs, y)
        return loss

    def predict(self, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        """
        Predict sum label by argmax over P(sum=k).
        """
        probs = self.sum_distribution(p1, p2)        # [B,19]
        return probs.argmax(dim=1)
