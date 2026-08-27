"""
Spectrum probes for concept c1 (closedness/roundness), on a saved emergent model.

Two sweeps that vary CLOSURE from opposite directions (user design):

  1. GAPPED CIRCLE : a full circle with a growing gap (arc spanning 360deg down
                     to a short arc).  Tests closed -> open.
  2. BENDING LINE  : a straight polyline that bends progressively until it closes
                     into a regular polygon.  Tests open -> closed.

If c1 = "closedness", it should rise monotonically with the closed fraction in
BOTH sweeps (and fall as the circle's gap grows).  We print c0/c1/c2 so the
other axes (c0 sharp-turns, c2 total-turning) can be watched too.

    python probe_c1_sweep.py --load saved/k3_trainable.pt
"""

from __future__ import annotations

import argparse
import math
import os
import random
import sys

import torch
from PIL import Image, ImageDraw

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, _REPO_ROOT)

from shapes import _SS, _MNIST_MEAN, _MNIST_STD                 # noqa: E402
from train_emergent_concepts import MultiTreeBaconCBM           # noqa: E402

_CANVAS = 28 * _SS


def _to_tensor(img):
    img = img.resize((28, 28), Image.BILINEAR)
    t = torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8)
    t = t.float().reshape(1, 28, 28) / 255.0
    return (t - _MNIST_MEAN) / _MNIST_STD


def arc_img(frac_closed, rng):
    """Circular arc spanning frac_closed*360 deg (1.0 = full circle)."""
    size = rng.randint(16, 20) * _SS
    m = (_CANVAS - size) // 2
    bbox = [m, m, m + size, m + size]
    w = rng.randint(2, 3) * _SS
    s0 = rng.uniform(0, 360)
    img = Image.new("L", (_CANVAS, _CANVAS), 0)
    ImageDraw.Draw(img).arc(bbox, s0, s0 + frac_closed * 360.0, fill=255, width=w)
    return _to_tensor(img)


def bend_img(bend, rng, n_seg=6):
    """Polyline of n_seg equal segments; joint turn = bend*(360/n_seg).
    bend=0 -> straight line; bend=1 -> closed regular n_seg-gon."""
    theta = bend * (2 * math.pi / n_seg)
    L = 6.0 * _SS
    ang = rng.uniform(0, 2 * math.pi)
    x = y = 0.0
    pts = [(0.0, 0.0)]
    for _ in range(n_seg):
        x += L * math.cos(ang); y += L * math.sin(ang)
        pts.append((x, y)); ang += theta
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    bw, bh = max(xs) - min(xs), max(ys) - min(ys)
    s = (0.7 * _CANVAS) / max(bw, bh, 1e-6)
    cx, cy = (min(xs) + max(xs)) / 2, (min(ys) + max(ys)) / 2
    pts = [((px - cx) * s + _CANVAS / 2, (py - cy) * s + _CANVAS / 2) for px, py in pts]
    w = rng.randint(2, 3) * _SS
    img = Image.new("L", (_CANVAS, _CANVAS), 0)
    ImageDraw.Draw(img).line(pts, fill=255, width=w, joint="curve")
    return _to_tensor(img)


def batch(fn, param, n, seed):
    rng = random.Random(seed)
    return torch.stack([fn(param, rng) for _ in range(n)], 0)


@torch.no_grad()
def sweep(model, device, fn, params, label, n=300):
    print(f"\n=== {label} ===")
    print(f"  {'param':>7s}   c0     c1     c2")
    for p in params:
        c = model.concept_probs(batch(fn, p, n, 123).to(device)).mean(0).cpu()
        print(f"  {p:7.2f}   {c[0]:.2f}   {c[1]:.2f}   {c[2]:.2f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", required=True)
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.load, map_location=device, weights_only=False)
    model = MultiTreeBaconCBM(ckpt["K"], weight_mode=ckpt.get("weight_mode", "trainable"),
                              device=device).to(device)
    if ckpt.get("frozen"):
        model.prepare_frozen_structure(); model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt["state_dict"]); model.anneal(1.0)
    model.eval()
    print(f"loaded {args.load}  K={ckpt['K']}  saved-acc={ckpt.get('acc', float('nan')) * 100:.2f}%")

    sweep(model, device, arc_img,
          [1.0, 0.9, 0.8, 0.65, 0.5, 0.35, 0.2], "GAPPED CIRCLE (frac closed: 1.0=full circle)", args.n)
    sweep(model, device, bend_img,
          [0.0, 0.2, 0.4, 0.6, 0.8, 1.0], "BENDING LINE (bend: 0=straight line, 1=closed hexagon)", args.n)


if __name__ == "__main__":
    main()
