"""
OCBM v3 on cFMNIST -> (OCBM v3, cFMNIST) ACC_Y/ACC_C.

v3 = the (already complete) cFMNIST tree structure kept FIXED, aggregators
data-fine-tuned (FixedGLTree, gl.generic): andness + per-input weights trained.

    python ocbm-v3-cfmnist-accuracy.py --iters 10
"""

from __future__ import annotations

import argparse

import _bench
import _fmnist

MODEL, DATASET, COL, LABEL = "ocbm-ft", "cfmnist", "cFMNIST", "OCBM v3"


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
