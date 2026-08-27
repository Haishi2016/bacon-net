"""
C_true -> Y on cFMNIST -> fills the (C_true->Y, cFMNIST) ACC_Y cell.

Linear probe on ground-truth concepts.  cFMNIST concepts are COMPLETE (the
season mutex group disambiguates the apparel triples), so this ceiling is ~100%.
ACC_C is blank.

    python ctruey-cfmnist-accuracy.py --iters 10
"""

from __future__ import annotations

import argparse

import _bench
import _fmnist

MODEL, DATASET, COL = "ctruey", "cfmnist", "cFMNIST"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lam", type=float, default=0.3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    acc_y, acc_c = _fmnist.run_cell(MODEL, DATASET, args.iters, epochs=args.epochs,
                                    batch_size=args.batch_size, lam=args.lam,
                                    seed=args.seed)
    _bench.report(_fmnist.LABELS[MODEL], COL, acc_y, acc_c)


if __name__ == "__main__":
    main()
