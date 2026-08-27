"""
CBM+SC on iFMNIST -> fills the (CBM+SC, iFMNIST) ACC_Y/ACC_C cells.

CBM plus a CREAM-style dropout-regularized black-box side-channel added to the
logits (SoftCBMSC).  Reuses the verified reproduction via _fmnist.run_cell.

    python cbm-sc-ifmnist-accuracy.py --iters 10
"""

from __future__ import annotations

import argparse

import _bench
import _fmnist

MODEL, DATASET, COL = "cbm+sc", "ifmnist", "iFMNIST"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lam", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    acc_y, acc_c = _fmnist.run_cell(MODEL, DATASET, args.iters, epochs=args.epochs,
                                    batch_size=args.batch_size, lam=args.lam,
                                    seed=args.seed)
    _bench.report(_fmnist.LABELS[MODEL], COL, acc_y, acc_c)


if __name__ == "__main__":
    main()
