"""OCBM v2 (GL trees) on iFMNIST -> ACC_Y/ACC_C. Updated GL trees on the complete
sFMNIST concept set (adds long_sleeve/front_opening/open_toe/ankle_high + NOT/OR),
fixing the incompleteness leak-free. See OCBM_V1_TREE_PROMPT.md."""

from __future__ import annotations

import argparse
import json
import os

import _bench
import _fmnist

DATASET, TRAINABLE, COL, LABEL = "sfmnist", False, "iFMNIST", "OCBM v2 (GL)"
_TREES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trees", "sfmnist_v2.json")


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
