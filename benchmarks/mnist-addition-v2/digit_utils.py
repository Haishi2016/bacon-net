# mnist-addition-2/digit_utils.py

"""
Helpers for loading the shared DigitClassifier backbone and
computing digit probabilities for single images or (x1, x2) pairs.
"""

import os
from typing import Tuple

import torch
import torch.nn.functional as F

from backbone import DigitClassifier
from config import BACKBONE_CFG


def load_digit_classifier(
    checkpoint_path: str = "checkpoints/digit_classifier_best.pt",
    device: torch.device | None = None,
    eval_mode: bool = True,
    freeze: bool = True,
) -> DigitClassifier:
    """
    Load DigitClassifier from a checkpoint.

    Args:
      checkpoint_path: path to .pt checkpoint file
      device: torch.device (if None, infer from CUDA availability)
      eval_mode: if True, set model.eval()
      freeze: if True, disable gradients on all parameters

    Returns:
      model: DigitClassifier instance on the given device
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"Digit checkpoint not found: {checkpoint_path}")

    ckpt = torch.load(checkpoint_path, map_location=device)
    feat_dim = ckpt.get("feat_dim", BACKBONE_CFG.feat_dim)

    model = DigitClassifier(feat_dim=feat_dim).to(device)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)

    if freeze:
        for p in model.parameters():
            p.requires_grad = False

    if eval_mode:
        model.eval()

    return model


# digit_utils.py (relevant parts)

def digit_probs(
    model: DigitClassifier,
    x: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    logits, feat = model(x)
    probs = F.softmax(logits, dim=1)
    return probs, feat


def pair_digit_probs(
    model: DigitClassifier,
    x1: torch.Tensor,
    x2: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    logits1, feat1 = model(x1)
    logits2, feat2 = model(x2)

    p1 = F.softmax(logits1, dim=1)
    p2 = F.softmax(logits2, dim=1)

    return p1, feat1, p2, feat2
