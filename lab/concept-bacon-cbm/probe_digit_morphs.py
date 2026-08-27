"""
Faithful digit morphs on a saved emergent model.

  * 0 -> 8 PINCH : start from a single circle and separate it into two stacked
    loops (a crossing/pinch appears in the middle).  Adds a real sharp crossing,
    so c0 (sharpness) should rise and the argmax flip 0 -> 8.
  * 9 -> 6 TAIL  : a fixed loop with a tail whose curvature grows OUTWARD (the
    tail arcs away, adding total turning) from straight (9) to curly (6), so c2
    (curvature) should rise and the argmax flip 9 -> 6 -- the human direction.

    python probe_digit_morphs.py --load saved/k3_trainable.pt
"""

from __future__ import annotations

import argparse
import collections
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


def pinch_0to8_img(t, rng):
    """t=0 single circle (0); t=1 two tangent stacked loops (8).  Two circle
    outlines whose vertical separation grows with t (they overlap -> a crossing
    in the middle -> a figure-8)."""
    R = 5.0 * _SS
    sep = t * R                                        # 0 -> R
    cx = _CANVAS / 2 + rng.uniform(-1, 1) * _SS
    cy = _CANVAS / 2 + rng.uniform(-1, 1) * _SS
    w = rng.randint(2, 3) * _SS
    img = Image.new("L", (_CANVAS, _CANVAS), 0)
    d = ImageDraw.Draw(img)
    for oy in (-sep, sep):
        d.ellipse([cx - R, cy + oy - R, cx + R, cy + oy + R], outline=255, width=w)
    return _to_tensor(img)


def tail_9to6_img(t, rng):
    """Fixed loop + a tail arcing OUTWARD; curvature grows with t.
    t=0 straight tail (9-like), t=1 curly tail (6-like)."""
    R = 5.5 * _SS
    cx = _CANVAS * 0.5 + rng.uniform(-1, 1) * _SS
    cy = _CANVAS * 0.58 + rng.uniform(-1, 1) * _SS
    w = rng.randint(2, 3) * _SS
    img = Image.new("L", (_CANVAS, _CANVAS), 0)
    d = ImageDraw.Draw(img)
    d.ellipse([cx - R, cy - R, cx + R, cy + R], outline=255, width=w)   # fixed loop
    # tail leaves the top of the loop and curves to the LEFT (outward), more with t
    x, y = cx, cy - R
    ang = -math.pi / 2                                 # start upward
    n, ds = 11, 1.5 * _SS
    kappa = t * (1.5 * math.pi) / (n * ds)             # total outward turn ~ t*1.5pi
    pts = [(x, y)]
    for _ in range(n):
        x += ds * math.cos(ang); y += ds * math.sin(ang)
        ang += kappa * ds                              # + = curl to the left/outward
        pts.append((x, y))
    d.line(pts, fill=255, width=w, joint="curve")
    return _to_tensor(img)


def batch(fn, t, n, seed):
    rng = random.Random(seed)
    return torch.stack([fn(t, rng) for _ in range(n)], 0)


@torch.no_grad()
def run(model, device, fn, ts, label, da, db, n):
    print(f"\n=== {label} ===")
    print(f"  {'t':>5s}   c0     c1     c2    tree{da}  tree{db}   pred")
    for t in ts:
        x = batch(fn, t, n, 123).to(device)
        logits, probs, truths = model(x)
        c = probs.mean(0).cpu()
        ta, tb = truths[:, da].mean().item(), truths[:, db].mean().item()
        pred = logits.argmax(1)
        mode, cnt = collections.Counter(pred.tolist()).most_common(1)[0]
        print(f"  {t:5.2f}   {c[0]:.2f}   {c[1]:.2f}   {c[2]:.2f}   "
              f"{ta:.2f}   {tb:.2f}   {mode} ({100 * cnt / len(pred):.0f}%)")


@torch.no_grad()
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

    run(model, device, pinch_0to8_img, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "0 -> 8 PINCH (t=0 circle, t=1 two stacked loops)", 0, 8, args.n)
    run(model, device, tail_9to6_img, [0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        "9 -> 6 TAIL (t=0 straight tail, t=1 curly tail)", 9, 6, args.n)


if __name__ == "__main__":
    main()
