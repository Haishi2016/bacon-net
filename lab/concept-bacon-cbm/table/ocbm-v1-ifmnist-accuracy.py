"""
OCBM v1 on iFMNIST -> (OCBM v1, iFMNIST) ACC_Y/ACC_C.

OCBM = our BACON-CBM: fixed per-class BACON logic trees (AND/OR/NOT) over the
concepts, no learned task head.  v1 uses the base (LLM/human-authored) trees on
the *incomplete* iFMNIST concept set -- which reveals the incompleteness (task
caps near the C_true->Y ceiling of ~60).

    python ocbm-v1-ifmnist-accuracy.py --iters 10
"""

from __future__ import annotations

import argparse

import _bench
import _fmnist

MODEL, DATASET, COL, LABEL = "ocbm", "ifmnist", "iFMNIST", "OCBM v1"


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
