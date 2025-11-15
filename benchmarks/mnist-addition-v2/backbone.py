# mnist-addition-2/backbone.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import BACKBONE_CFG


class DigitBackbone(nn.Module):
    """
    Shared CNN backbone for MNIST digits.

    Produces a feature vector of size `feat_dim` from a 1x28x28 input.
    """
    def __init__(self, feat_dim: int = BACKBONE_CFG.feat_dim,
                 use_batchnorm: bool = BACKBONE_CFG.use_batchnorm):
        super().__init__()

        def conv_block(in_ch, out_ch, bn=True):
            layers = [nn.Conv2d(in_ch, out_ch, 3, padding=1)]
            if bn and use_batchnorm:
                layers.append(nn.BatchNorm2d(out_ch))
            layers.append(nn.ReLU(inplace=True))
            return nn.Sequential(*layers)

        self.features = nn.Sequential(
            conv_block(1, 32),
            nn.MaxPool2d(2),         # 14x14
            conv_block(32, 64),
            nn.MaxPool2d(2),         # 7x7
            conv_block(64, 64),
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, feat_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.fc(x)
        return x


class DigitClassifier(nn.Module):
    """
    Backbone + linear head → digit logits (0..9).

    This is what you'll train once on single-digit MNIST,
    then reuse for MNIST-addition reasoning systems.
    """
    def __init__(self, feat_dim: int = BACKBONE_CFG.feat_dim):
        super().__init__()
        self.backbone = DigitBackbone(feat_dim=feat_dim)
        self.head = nn.Linear(feat_dim, 10)

    def forward(self, x: torch.Tensor):
        feat = self.backbone(x)      # [B, feat_dim]
        logits = self.head(feat)     # [B, 10]
        return logits, feat
