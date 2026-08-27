"""OCBM v3 (GL trees) on CelebA (Smiling) -> ACC_Y/ACC_C. Same GL smile-tree
structure kept fixed, andness + per-input weights data-fine-tuned, end-to-end
ResNet-18."""

from __future__ import annotations

import argparse
import json
import os

import _bench
import _celeba

TRAINABLE, COL, LABEL = True, "CelebA", "OCBM v3 (GL)"
_TREES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trees", "celeba_v1.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--freeze", action="store_true")
    ap.add_argument("--trees", type=str, default=_TREES)
    args = ap.parse_args()

    with open(args.trees, "r", encoding="utf-8") as f:
        trees = json.load(f)
    acc_y, acc_c = _celeba.run_gl_cell(trees, args.iters, trainable=TRAINABLE,
                                       epochs=args.epochs, batch_size=args.batch_size,
                                       lam=args.lam, seed=args.seed,
                                       freeze_backbone=args.freeze)
    _bench.report(LABEL, COL, acc_y, acc_c)


if __name__ == "__main__":
    main()
