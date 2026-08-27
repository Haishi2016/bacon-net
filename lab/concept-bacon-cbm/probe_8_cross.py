"""Targeted probe: does the emergent tree-8 detect 'closed curves WITH crosses'?

Draws a closed curve alone, a closed curve with a horizontal mid-bar (8-junction),
a closed curve with an internal X cross, two stacked circles (real 8 shape), and
plain cross/line -- all MNIST-style -- and reads the 3 emergent concepts
(c0~closed / c1~curved / c2~cross) plus the tree-0 and tree-8 scores.

    python probe_8_cross.py --load saved/k3_harden.pt
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from PIL import Image, ImageDraw

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

from train_emergent_concepts import MultiTreeBaconCBM           # noqa: E402

_MEAN, _STD, _SS, _C = 0.1307, 0.3081, 4, 28 * 4
_W = 2 * _SS                                                    # stroke width


def _canvas():
    return Image.new("L", (_C, _C), 0)


def _finish(img):
    small = img.resize((28, 28), Image.BILINEAR)
    t = torch.from_numpy(
        __import__("numpy").frombuffer(bytearray(small.tobytes()), dtype="uint8")
        .reshape(28, 28).copy()).float() / 255.0
    return ((t - _MEAN) / _STD).view(1, 1, 28, 28)


def circle():
    im = _canvas(); d = ImageDraw.Draw(im)
    d.ellipse([22, 22, _C - 22, _C - 22], outline=255, width=_W)
    return im


def circle_midbar():
    im = circle(); d = ImageDraw.Draw(im)
    d.line([24, _C // 2, _C - 24, _C // 2], fill=255, width=_W)      # theta / 8 junction
    return im


def circle_xcross():
    im = circle(); d = ImageDraw.Draw(im)
    d.line([34, 34, _C - 34, _C - 34], fill=255, width=_W)          # closed curve
    d.line([_C - 34, 34, 34, _C - 34], fill=255, width=_W)          #   WITH an X
    return im


def two_circles():
    im = _canvas(); d = ImageDraw.Draw(im)
    d.ellipse([30, 12, _C - 30, _C // 2 + 4], outline=255, width=_W)
    d.ellipse([30, _C // 2 - 4, _C - 30, _C - 12], outline=255, width=_W)
    return im


def plain_cross():
    im = _canvas(); d = ImageDraw.Draw(im)
    d.line([28, 28, _C - 28, _C - 28], fill=255, width=_W)
    d.line([_C - 28, 28, 28, _C - 28], fill=255, width=_W)
    return im


def line():
    im = _canvas(); d = ImageDraw.Draw(im)
    d.line([_C // 2, 20, _C // 2, _C - 20], fill=255, width=_W)
    return im


SHAPES = [("circle (closed curve)", circle),
          ("circle + midbar", circle_midbar),
          ("circle + X cross", circle_xcross),
          ("two stacked circles (8)", two_circles),
          ("plain cross", plain_cross),
          ("line", line)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", type=str, default=os.path.join(_HERE, "saved", "k3_harden.pt"))
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(args.load, map_location=device, weights_only=False)
    model = MultiTreeBaconCBM(ckpt["K"], weight_mode=ckpt.get("weight_mode", "trainable"),
                              device=device).to(device)
    if ckpt.get("frozen"):
        model.prepare_frozen_structure(); model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt["state_dict"]); model.anneal(1.0)
    model.eval()

    print(f"\nEMERGENT {os.path.basename(args.load)}  (c0~closed c1~curved c2~cross)")
    print(f"  {'shape':26s} {'c0':>5s} {'c1':>5s} {'c2':>5s}   {'tree0':>6s} {'tree8':>6s}")
    with torch.no_grad():
        for name, fn in SHAPES:
            x = _finish(fn()).to(device)
            _, probs, truths = model(x)
            p = probs[0].tolist()
            print(f"  {name:26s} {p[0]:5.2f} {p[1]:5.2f} {p[2]:5.2f}   "
                  f"{truths[0, 0].item():6.3f} {truths[0, 8].item():6.3f}")
    print("\n  hypothesis: if tree-8 = 'closed curve WITH cross', then "
          "'circle + X cross' should\n  score higher on tree-8 than a plain circle.")


if __name__ == "__main__":
    main()
