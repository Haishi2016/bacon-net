# mnist-addition-v2/reasoners/mlp_reasoner.py

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLPAdditionReasoner(nn.Module):
    """
    Simple neural baseline: MLP over concatenated digit probs [p1, p2].

    Inputs:
      p1: [B,10]
      p2: [B,10]

    Outputs:
      logits: [B,19] (for sums 0..18)
    """
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 19),
        )

    def forward(self, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        x = torch.cat([p1, p2], dim=1)   # [B,20]
        logits = self.net(x)             # [B,19]
        return logits

    def loss(self, p1: torch.Tensor, p2: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logits = self.forward(p1, p2)
        return F.cross_entropy(logits, y)

    def predict(self, p1: torch.Tensor, p2: torch.Tensor) -> torch.Tensor:
        logits = self.forward(p1, p2)
        return logits.argmax(dim=1)
