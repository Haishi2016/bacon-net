# mnist-addition-v3/reasoners/dcr_reasoner.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class DCRAdditionReasoner(nn.Module):
    """
    DCR-style addition reasoner.

    Interpretation:
      - The 10-dimensional probability vectors p1, p2 are activations
        over 10 "digit concepts" (0..9) for the first and second image.
      - We maintain separate learnable embeddings for these concepts:
          E1 ∈ R^{10×d}, E2 ∈ R^{10×d}.
      - For each input:
          c1 = p1 @ E1  ∈ R^d   (soft concept embedding for image 1)
          c2 = p2 @ E2  ∈ R^d   (soft concept embedding for image 2)
        and then reason over [c1, c2] with an MLP to predict the sum.

    This matches the DCR spirit: learned concept embeddings + a
    differentiable reasoning layer on top of them.
    """

    def __init__(self, emb_dim: int = 16, hidden_dim: int = 64):
        super().__init__()
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim

        # Learnable concept embeddings for the 10 digit concepts
        # E1: concepts for first digit position, shape [10, emb_dim]
        # E2: concepts for second digit position, shape [10, emb_dim]
        self.E1 = nn.Parameter(torch.randn(10, emb_dim) * 0.1)
        self.E2 = nn.Parameter(torch.randn(10, emb_dim) * 0.1)

        # Reasoning MLP over concatenated concept embeddings [c1, c2]
        self.reasoner = nn.Sequential(
            nn.Linear(2 * emb_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 19),  # sums 0..18
        )

    def forward(self, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        """
        Forward pass:
          p1, p2: [B,10] digit probabilities (concept activations)
          returns: logits over sums, shape [B,19]
        """
        # Soft concept embeddings: weighted averages of concept vectors
        # p1: [B,10], E1: [10,emb] → c1: [B,emb]
        c1 = p1 @ self.E1
        c2 = p2 @ self.E2

        # Concatenate and reason
        z = torch.cat([c1, c2], dim=1)        # [B, 2*emb_dim]
        logits = self.reasoner(z)            # [B,19]
        return logits

    def loss(self, p1: torch.Tensor, p2: torch.Tensor, y_sum: torch.Tensor) -> torch.Tensor:
        """
        Cross-entropy loss on logits vs. true sum labels.
        """
        logits = self.forward(p1, p2)        # [B,19]
        return F.cross_entropy(logits, y_sum)

    def predict(self, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        """
        Predict sum label via argmax of logits.
        """
        logits = self.forward(p1, p2)
        return logits.argmax(dim=1)
