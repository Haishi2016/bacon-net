"""Show how the negative-only digits (1, 3, 6) are distinguished from each other.

They lack a dedicated positive concept, so they are recognised by WHICH concepts
they require to be ABSENT. Prints the per-digit mean concept vector and the
pruned negative rule, so one can see 1/3/6 occupy different corners of concept
space.

    python show_neg_digits.py --load saved/k5_harden.pt
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

from concept_receptive_fields import load_model, collect        # noqa: E402
from train_emergent_concepts import make_loaders                # noqa: E402

RULES = {1: "not c1 & not c0", 3: "not c1 & not c2 (& not c0)",
         6: "not c2 & not c3"}
NAMES = ["c0 junction", "c1 loop", "c2 straight", "c3 top-arc", "c4 edge"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", required=True)
    ap.add_argument("--data", default="./data")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ck = load_model(args.load, device)
    _, ld = make_loaders(args.data, 512)
    X, C, Y = collect(model, ld, device)

    print("\nper-digit MEAN concept activation  (" + ", ".join(NAMES) + "):")
    print("   d    c0    c1    c2    c3    c4     rule (neg-only digits)")
    for d in range(10):
        row = "  ".join(f"{C[Y == d, i].mean():.2f}" for i in range(5))
        tag = f"   <- {RULES[d]}" if d in RULES else ""
        mark = "*" if d in RULES else " "
        print(f"  {mark}{d}  {row}{tag}")

    print("\nhow 1 / 3 / 6 separate (each LOW on a DIFFERENT pair):")
    for d in (1, 3, 6):
        v = C[Y == d].mean(0)
        lows = sorted(range(5), key=lambda i: v[i])[:2]
        highs = sorted(range(5), key=lambda i: -v[i])[:2]
        print(f"  digit {d}: lowest = {[NAMES[i] for i in lows]}"
              f"  ({', '.join(f'{v[i]:.2f}' for i in lows)});"
              f"  highest = {[NAMES[i] for i in highs]}"
              f"  ({', '.join(f'{v[i]:.2f}' for i in highs)})")


if __name__ == "__main__":
    main()
