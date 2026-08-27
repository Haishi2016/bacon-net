"""
C_true -> Y on CUB -> (C_true->Y, CUB) ACC_Y cell.

Linear probe on the 112 true concept labels -> 200 classes (no backbone).  CUB
concepts are complete, so this ceiling is ~100%.  ACC_C blank.

    python ctruey-cub-accuracy.py --iters 3
"""

from __future__ import annotations

import argparse

import _bench
import _cub

MODEL, COL = "ctruey", "CUB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    acc_y, acc_c = _cub.run_cell(MODEL, args.iters, epochs=args.epochs,
                                 batch_size=args.batch_size, lam=args.lam,
                                 seed=args.seed)
    _bench.report(_cub.LABELS[MODEL], COL, acc_y, acc_c)


if __name__ == "__main__":
    main()
