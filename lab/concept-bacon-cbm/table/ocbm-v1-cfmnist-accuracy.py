"""
OCBM v1 on cFMNIST -> (OCBM v1, cFMNIST) ACC_Y/ACC_C.

cFMNIST's concept set is already COMPLETE (the season mutex group disambiguates
the ambiguous apparel triples, C_true->Y = 100), so v1 already reaches ~92 and
v2 collapses to v1 (no incompleteness to fix).  Fixed BACON trees, leak-free.

    python ocbm-v1-cfmnist-accuracy.py --iters 10
"""

from __future__ import annotations

import argparse

import _bench
import _fmnist

MODEL, DATASET, COL, LABEL = "ocbm", "cfmnist", "cFMNIST", "OCBM v1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    acc_y, acc_c = _fmnist.run_cell(MODEL, DATASET, args.iters, epochs=args.epochs,
                                    batch_size=args.batch_size, lam=args.lam,
                                    seed=args.seed)
    _bench.report(LABEL, COL, acc_y, acc_c)


if __name__ == "__main__":
    main()
