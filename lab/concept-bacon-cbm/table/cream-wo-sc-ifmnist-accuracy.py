"""
CREAM w/o SC on iFMNIST -> fills the (CREAM w/o SC, iFMNIST) ACC_Y/ACC_C cells.

The CREAM model with its black-box side-channel turned OFF at evaluation, i.e.
the concept-only pathway.  Reuses the verified reproduction via _fmnist.run_cell.

    python cream-wo-sc-ifmnist-accuracy.py --iters 10
"""

from __future__ import annotations

import argparse

import _bench
import _fmnist

MODEL, DATASET, COL = "cream-wo-sc", "ifmnist", "iFMNIST"


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
    _bench.report(_fmnist.LABELS[MODEL], COL, acc_y, acc_c)


if __name__ == "__main__":
    main()
