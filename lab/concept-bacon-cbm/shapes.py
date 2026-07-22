"""
Synthetic geometry-shapes dataset, rendered MNIST-style.

There is no off-the-shelf geometry-shape dataset in this repo, so we generate
one procedurally.  Shapes are drawn as *white outlines on a black background*
(like MNIST digit strokes), anti-aliased, jittered in size / position /
rotation / stroke width, then downsampled to 28x28 and normalized with the
MNIST statistics.  This lets the digit-trained concept encoder be applied
directly (zero-shot).

A circle outline is essentially a hand-drawn "0", so the learned "0" BACON tree
(loop_upper AND loop_lower AND NOT horizontal_middle AND NOT vertical_line)
should fire on circles without any retraining.

Shapes: circle, ellipse, square, rectangle, triangle, line, cross.
"""

from __future__ import annotations

import math
import random
from typing import List, Tuple

import torch
from PIL import Image, ImageDraw

SHAPES = ["circle", "ellipse", "square", "rectangle", "triangle", "line", "cross"]
ROUND_SHAPES = {"circle", "ellipse"}   # "positive" class for the circle detector

_MNIST_MEAN = 0.1307
_MNIST_STD = 0.3081
_SS = 4                                 # supersampling factor for anti-aliasing
_CANVAS = 28 * _SS


def _rot(points, cx, cy, ang):
    c, s = math.cos(ang), math.sin(ang)
    return [((x - cx) * c - (y - cy) * s + cx,
             (x - cx) * s + (y - cy) * c + cy) for x, y in points]


def _draw_shape(shape: str, rng: random.Random) -> Image.Image:
    img = Image.new("L", (_CANVAS, _CANVAS), 0)
    d = ImageDraw.Draw(img)
    width = rng.randint(2, 4) * _SS                 # stroke width (supersampled)
    # bounding box: keep the shape roughly centered, ~18-24 px at 28-scale
    size = rng.randint(15, 22) * _SS
    margin_x = rng.randint(2, max(2, 28 * _SS - size - 2 * _SS))
    margin_y = rng.randint(2, max(2, 28 * _SS - size - 2 * _SS))
    x0, y0 = margin_x, margin_y
    x1, y1 = x0 + size, y0 + size
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    ang = rng.uniform(-math.pi, math.pi)

    if shape == "circle":
        d.ellipse([x0, y0, x1, y1], outline=255, width=width)
    elif shape == "ellipse":
        squash = rng.uniform(0.55, 0.8)
        h = size * squash
        d.ellipse([x0, cy - h / 2, x1, cy + h / 2], outline=255, width=width)
    elif shape == "square":
        pts = _rot([(x0, y0), (x1, y0), (x1, y1), (x0, y1)], cx, cy, ang)
        d.polygon(pts, outline=255, width=width)
    elif shape == "rectangle":
        h = size * rng.uniform(0.5, 0.75)
        pts = _rot([(x0, cy - h / 2), (x1, cy - h / 2),
                    (x1, cy + h / 2), (x0, cy + h / 2)], cx, cy, ang)
        d.polygon(pts, outline=255, width=width)
    elif shape == "triangle":
        pts = _rot([(cx, y0), (x1, y1), (x0, y1)], cx, cy, ang)
        d.polygon(pts, outline=255, width=width)
    elif shape == "line":
        pts = _rot([(cx, y0), (cx, y1)], cx, cy, ang)
        d.line(pts, fill=255, width=width)
    elif shape == "cross":
        v = _rot([(cx, y0), (cx, y1)], cx, cy, ang)
        h = _rot([(x0, cy), (x1, cy)], cx, cy, ang)
        d.line(v, fill=255, width=width)
        d.line(h, fill=255, width=width)
    else:
        raise ValueError(f"unknown shape {shape}")

    return img.resize((28, 28), Image.BILINEAR)


def make_shape_tensor(shape: str, rng: random.Random) -> torch.Tensor:
    img = _draw_shape(shape, rng)
    # bytearray -> writable buffer (avoids torch.frombuffer non-writable warning)
    t = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8).float().reshape(1, 28, 28)
    t = t / 255.0
    return (t - _MNIST_MEAN) / _MNIST_STD


def generate(n_per_shape: int, seed: int = 0,
             shapes: List[str] = None) -> Tuple[torch.Tensor, List[str]]:
    """Return (images (N,1,28,28), shape_names list of length N)."""
    shapes = shapes or SHAPES
    rng = random.Random(seed)
    imgs, names = [], []
    for shape in shapes:
        for _ in range(n_per_shape):
            imgs.append(make_shape_tensor(shape, rng))
            names.append(shape)
    return torch.stack(imgs, 0), names


def make_circles_tensor(circles, rng: random.Random, jitter: float = 1.0) -> torch.Tensor:
    """Render several circle outlines in one 28x28 MNIST-style frame.

    Args:
        circles: list of (cx, cy, r) in 28-pixel coordinates.
        jitter: max global translation (px) applied to the whole scene.
    """
    img = Image.new("L", (_CANVAS, _CANVAS), 0)
    d = ImageDraw.Draw(img)
    width = rng.randint(2, 3) * _SS
    dx = rng.uniform(-jitter, jitter)
    dy = rng.uniform(-jitter, jitter)
    for (cx, cy, r) in circles:
        rr = r * rng.uniform(0.92, 1.08)
        d.ellipse([(cx + dx - rr) * _SS, (cy + dy - rr) * _SS,
                   (cx + dx + rr) * _SS, (cy + dy + rr) * _SS],
                  outline=255, width=width)
    img = img.resize((28, 28), Image.BILINEAR)
    t = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8).float().reshape(1, 28, 28)
    t = t / 255.0
    return (t - _MNIST_MEAN) / _MNIST_STD


def save_preview(path: str, seed: int = 0) -> None:
    """Save a small grid PNG (one column per shape) for eyeballing."""
    rng = random.Random(seed)
    cols, rows = len(SHAPES), 6
    grid = Image.new("L", (cols * 28, rows * 28), 0)
    for c, shape in enumerate(SHAPES):
        for r in range(rows):
            grid.paste(_draw_shape(shape, rng), (c * 28, r * 28))
    grid.save(path)


if __name__ == "__main__":
    import os
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "shapes_preview.png")
    save_preview(out)
    print(f"saved preview -> {out}")
