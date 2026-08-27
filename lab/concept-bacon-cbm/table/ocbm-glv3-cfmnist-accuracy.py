"""OCBM v3 (GL trees) on cFMNIST -> ACC_Y/ACC_C. The complete cFMNIST GL tree
structure kept fixed, with each node's andness + per-input weights fine-tuned."""

from __future__ import annotations

import argparse
import json
import os

import _bench
import _fmnist

DATASET, TRAINABLE, COL, LABEL = "cfmnist", True, "cFMNIST", "OCBM v3 (GL)"
_TREES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trees", "cfmnist_v1.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trees", type=str, default=_TREES)
    args = ap.parse_args()

    with open(args.trees, "r", encoding="utf-8") as f:
        trees = json.load(f)
    acc_y, acc_c = _fmnist.run_gl_cell(trees, DATASET, args.iters, trainable=TRAINABLE,
                                       epochs=args.epochs, batch_size=args.batch_size,
                                       lam=args.lam, seed=args.seed)
    _bench.report(LABEL, COL, acc_y, acc_c)


if __name__ == "__main__":
    main()
