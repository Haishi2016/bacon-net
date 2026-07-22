"""
Probe the frozen "8" BACON tree on multi-circle scenes (zero-shot, no retraining).

The "8" tree is
    loop_upper AND loop_lower AND horizontal_middle
i.e. a loop in the upper half AND a loop in the lower half AND a bar/junction
across the middle.  A real "8" is exactly two vertically-stacked *touching*
circles: the two loops plus the pinch where they meet (the middle junction).

We render circle scenes that vary along three axes and read the "8" tree's truth
value (truths[:, 8]) as an "is-this-an-8?" score:

  1) number of circles         : 1 / 2 / 3 vertically stacked & touching
  2) relative position         : vertical / horizontal / diagonal (2 touching)
  3) touching vs not touching  : 2 vertical circles, touch / small gap / far gap

Everything transfers from the MNIST-trained encoder; nothing is trained on these
scenes.
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, _REPO_ROOT)

import shapes as shapes_mod                    # noqa: E402
from eval_shapes import load_model             # noqa: E402


# Scene layouts: name -> (list of (cx, cy, r), axis-group).  28x28 frame, center 14.
SCENES = [
    # --- axis 1: number of circles (vertical, touching) ---------------- #
    ("1_circle",              [(14, 14, 6)],                              "count"),
    ("2_vstack_touch (8)",    [(14, 8, 6), (14, 20, 6)],                  "count"),
    ("3_vstack_touch",        [(14, 6, 4), (14, 14, 4), (14, 22, 4)],     "count"),
    # --- axis 2: relative position (2 circles, touching) --------------- #
    ("2_vertical_touch",      [(14, 8, 6), (14, 20, 6)],                  "position"),
    ("2_horizontal_touch",    [(8, 14, 6), (20, 14, 6)],                  "position"),
    ("2_diagonal_touch",      [(10, 10, 6), (18, 18, 6)],                 "position"),
    # --- axis 3: touching vs not (2 vertical circles, r held constant) - #
    ("2_vertical_touch2",     [(14, 8, 6), (14, 20, 6)],                  "contact"),
    ("2_vertical_gap",        [(14, 7, 6), (14, 21, 6)],                  "contact"),
    ("2_vertical_fargap",     [(14, 6, 6), (14, 22, 6)],                  "contact"),
]


def generate_scene(circles, n, seed):
    rng = random.Random(seed)
    return torch.stack([shapes_mod.make_circles_tensor(circles, rng) for _ in range(n)], 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=os.path.join(_HERE, "checkpoint.pt"))
    ap.add_argument("--n", type=int, default=400, help="samples per scene")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--preview", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.exists(args.ckpt):
        raise SystemExit(f"checkpoint not found: {args.ckpt}\nrun train.py first.")
    model = load_model(args.ckpt, device)
    ci = model.logic.concept_index
    lu, ll, hm = ci["loop_upper"], ci["loop_lower"], ci["horizontal_middle"]

    if args.preview:
        from PIL import Image
        grid = Image.new("L", (len(SCENES) * 28, 28), 0)
        rng = random.Random(args.seed)
        for c, (name, circles, _) in enumerate(SCENES):
            t = shapes_mod.make_circles_tensor(circles, rng)
            im = ((t * shapes_mod._MNIST_STD + shapes_mod._MNIST_MEAN)
                  .clamp(0, 1).mul(255).byte().squeeze().numpy())
            grid.paste(Image.fromarray(im, "L"), (c * 28, 0))
        p = os.path.join(_HERE, "eight_scenes_preview.png")
        grid.save(p)
        print(f"saved scene preview -> {p}")

    print(f"\ncheckpoint: {args.ckpt}   {args.n} samples/scene   device: {device}")
    print("\n'8'-tree = loop_upper AND loop_lower AND horizontal_middle")
    print("(also showing '0'-tree for contrast, and the 3 literal concepts)\n")
    header = (f"{'scene':22s} {'8-score':>8s} {'0-score':>8s} "
              f"{'loopU':>6s} {'loopL':>6s} {'midBar':>6s}")

    last_group = None
    for name, circles, group in SCENES:
        if group != last_group:
            print(f"-- axis: {group} " + "-" * (58 - len(group)))
            last_group = group
        x = generate_scene(circles, args.n, args.seed).to(device)
        with torch.no_grad():
            _, probs, truths = model(x)
        s8 = truths[:, 8].mean().item()
        s0 = truths[:, 0].mean().item()
        print(f"{name:22s} {s8:8.3f} {s0:8.3f} "
              f"{probs[:, lu].mean():6.2f} {probs[:, ll].mean():6.2f} "
              f"{probs[:, hm].mean():6.2f}")

    print("\nExpectation: the '8'-tree score peaks for 2-3 vertically stacked "
          "TOUCHING circles,\nand drops for a single circle, side-by-side "
          "circles, or non-touching circles\n(loss of the middle-junction concept).")


if __name__ == "__main__":
    main()
