"""
6-vs-9 curvature morph, on a saved emergent K=3 model.

6 and 9 share the same closedness (a loop + a tail); people distinguish them by
the TAIL: a "6" has a curly tail, a "9" a straight one.  This holds the loop
FIXED (so c1 = closedness is ~constant) and sweeps only the tail's curvature from
straight (9-like) to curly (6-like), then reads:

  * c0 / c1 / c2  (sharp-turns / closedness / total-curvature),
  * the digit-6 and digit-9 TREE truth scores,
  * the model's most-common predicted digit.

Expectation: c1 ~flat, c2 rises with curl, and the argmax slides 9 -> 6 as the
tail curls -- i.e. the 6/9 boundary is essentially a c2 (curvature) threshold.

    python probe_6v9_morph.py --load saved/k3_trainable.pt
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


def loop_tail_img(curl, rng):
    """Fixed loop + a tail whose curvature = curl (0 straight -> 1 curly)."""
    img = Image.new("L", (_CANVAS, _CANVAS), 0)
    d = ImageDraw.Draw(img)
    w = rng.randint(2, 3) * _SS
    R = 6 * _SS
    cx = _CANVAS * 0.5 + rng.uniform(-1, 1) * _SS
    cy = _CANVAS * 0.62 + rng.uniform(-1, 1) * _SS
    d.ellipse([cx - R, cy - R, cx + R, cy + R], outline=255, width=w)   # fixed loop
    # tail from top of the loop, curving by `curl`
    x, y = cx, cy - R
    ang = -math.pi / 2                                      # start upward
    n, ds = 10, 1.6 * _SS
    kappa = curl * math.pi / (n * ds)                      # total turn ~ curl*pi
    pts = [(x, y)]
    for _ in range(n):
        x += ds * math.cos(ang); y += ds * math.sin(ang)
        ang -= kappa * ds
        pts.append((x, y))
    d.line(pts, fill=255, width=w, joint="curve")
    return _to_tensor(img)


def batch(curl, n, seed):
    rng = random.Random(seed)
    return torch.stack([loop_tail_img(curl, rng) for _ in range(n)], 0)


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

    print("\n=== 6-vs-9 tail-curvature morph (loop FIXED, tail straight->curly) ===")
    print(f"  {'curl':>5s}   c0     c1     c2    tree6  tree9   pred")
    for curl in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]:
        x = batch(curl, args.n, 123).to(device)
        logits, probs, truths = model(x)
        c = probs.mean(0).cpu()
        t6, t9 = truths[:, 6].mean().item(), truths[:, 9].mean().item()
        pred = logits.argmax(1)
        mode, cnt = collections.Counter(pred.tolist()).most_common(1)[0]
        print(f"  {curl:5.2f}   {c[0]:.2f}   {c[1]:.2f}   {c[2]:.2f}   "
              f"{t6:.2f}   {t9:.2f}   {mode} ({100 * cnt / len(pred):.0f}%)")


if __name__ == "__main__":
    main()
