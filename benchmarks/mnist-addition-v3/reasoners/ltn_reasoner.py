# mnist-addition-v3/reasoners/ltn_reasoner.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class LTNAdditionReasoner(nn.Module):
    """
    LTN-style reasoning for MNIST addition, operating on digit probabilities.

    Given:
      p1: [B,10] digit probabilities for the first image
      p2: [B,10] digit probabilities for the second image

    We define satisfaction for each possible sum k in {0,..,18} as:

        sat_k = sum_{i+j = k} p1[i] * p2[j]

    and use a log-satisfaction loss on the true sum label:

        L = - mean_b log( sat_{y_b} )

    This is a parameter-free reasoner; all learning happens in the CNN
    (DeepProbLogMNISTNet) so that its outputs satisfy the logical rule.
    """

    def __init__(self):
        super().__init__()
        # Precompute sum indices: sum_idx[i, j] = i + j
        sum_idx = torch.arange(10).view(10, 1) + torch.arange(10).view(1, 10)
        self.register_buffer("sum_idx", sum_idx)  # [10,10]

    def satisfaction_per_sum(self, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        """
        Compute satisfaction degrees for each possible sum k.

        Args:
            p1, p2: [B,10] digit probabilities

        Returns:
            sat: [B,19], where sat[b, k] = degree of satisfaction that sum = k.
        """
        # Outer product: T[b, i, j] = p1[b, i] * p2[b, j]
        T = p1.unsqueeze(2) * p2.unsqueeze(1)  # [B,10,10]

        sats = []
        for k in range(19):
            mask_k = (self.sum_idx == k)       # [10,10] bool
            s_k = T[:, mask_k].sum(dim=1)      # [B]
            sats.append(s_k)

        sat = torch.stack(sats, dim=1)         # [B,19]
        # Optionally renormalize so they sum to ~1 (not strictly needed, but stable)
        sat = sat / (sat.sum(dim=1, keepdim=True) + 1e-9)
        return sat

    def forward(self, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: return satisfaction matrix [B,19].
        """
        return self.satisfaction_per_sum(p1, p2)

    def loss(self, p1: torch.Tensor, p2: torch.Tensor, y_sum: torch.Tensor) -> torch.Tensor:
        """
        LTN-style log-satisfaction loss:

            L = - E_b [ log sat[b, y_b] ]
        """
        sat = self.satisfaction_per_sum(p1, p2)     # [B,19]
        sat_y = sat.gather(1, y_sum.view(-1, 1)).squeeze(1)  # [B]
        loss = -torch.log(sat_y + 1e-9).mean()
        return loss

    def predict(self, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        """
        Predict sum by argmax over satisfaction degrees.
        """
        sat = self.satisfaction_per_sum(p1, p2)     # [B,19]
        return sat.argmax(dim=1)
