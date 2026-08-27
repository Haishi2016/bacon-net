"""
OCBM v2 on iFMNIST -> (OCBM v2, iFMNIST) ACC_Y/ACC_C.

v2 = the UPDATED BACON trees that fix the incompleteness revealed by v1: the
sFMNIST concept set adds four meaningful binary attributes (long_sleeve,
front_opening, open_toe, ankle_high) and disambiguates with richer logic
(NOT / nested OR), making the concept set complete (C_true->Y = 100) -- so the
fixed trees now reach ~92, still leak-free.  Same FashionMNIST task.

    python ocbm-v2-ifmnist-accuracy.py --iters 10
"""

from __future__ import annotations

import argparse

import _bench
import _fmnist

MODEL, DATASET, COL, LABEL = "ocbm", "sfmnist", "iFMNIST", "OCBM v2"


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
