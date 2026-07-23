"""
Turn real-life color photos into MNIST-style stroke images.

The concept encoder was trained on 28x28 white-stroke-on-black digits, so a raw
photo is completely out of distribution.  The bridge is an edge transform: a
photo of a round object (plate, clock, apple) becomes a white circular outline
on a black background -- i.e. it lands back in the "hand-drawn 0" domain the
encoder understands.

Pipeline (all torch, no OpenCV dependency):
    color -> grayscale -> (optional blur) -> Sobel gradient magnitude
          -> per-image normalize -> soft threshold -> resize 28x28
          -> MNIST normalize
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

_MNIST_MEAN = 0.1307
_MNIST_STD = 0.3081

_GRAY = torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1)
_SOBEL_X = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
_SOBEL_Y = _SOBEL_X.transpose(-1, -2).clone()
_BLUR = (torch.tensor([[1., 2., 1.], [2., 4., 2.], [1., 2., 1.]]) / 16.0).view(1, 1, 3, 3)


def to_sketch(imgs: torch.Tensor, out: int = 28, blur: int = 2,
              keep_frac: float = 0.12, sharpness: float = 20.0) -> torch.Tensor:
    """imgs: (N,3,H,W) float in [0,1] -> (N,1,28,28) MNIST-normalized edge sketch.

    ``keep_frac`` is the target fraction of white (stroke) pixels per image; a
    per-image percentile threshold enforces MNIST-like sparsity (~0.12) so the
    encoder is not swamped by dense texture edges.
    """
    dev = imgs.device
    gray = (imgs * _GRAY.to(dev)).sum(1, keepdim=True)          # (N,1,H,W)
    for _ in range(max(0, blur)):                               # suppress texture
        gray = F.conv2d(gray, _BLUR.to(dev), padding=1)
    gx = F.conv2d(gray, _SOBEL_X.to(dev), padding=1)
    gy = F.conv2d(gray, _SOBEL_Y.to(dev), padding=1)
    mag = torch.sqrt(gx * gx + gy * gy + 1e-6)                  # gradient magnitude
    # resize BEFORE thresholding -> thin strokes at the final resolution
    mag = F.interpolate(mag, size=(out, out), mode="bilinear", align_corners=False)
    # per-image percentile threshold -> fixed stroke density (MNIST-like sparsity)
    flat = mag.flatten(1)
    thr = torch.quantile(flat, 1.0 - keep_frac, dim=1).view(-1, 1, 1, 1)
    scale = flat.amax(1).view(-1, 1, 1, 1).clamp(min=1e-6)
    sketch = torch.sigmoid((mag - thr) / scale * sharpness)
    return (sketch - _MNIST_MEAN) / _MNIST_STD
