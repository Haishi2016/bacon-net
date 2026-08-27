"""
OCBM v1 on CUB (end-to-end ResNet-18) -> (OCBM v1, CUB) ACC_Y/ACC_C.

OCBM = our BACON-CBM: per-class signed-AND BACON tree built from the class
concept prototype (positive literal where an attribute is ON, NOT where OFF),
no learned task head.  CUB's 112 concepts are complete (ceiling 100), so v2
collapses to v1.

    python ocbm-v1-cub-accuracy.py --iters 3 --epochs 40
"""

from __future__ import annotations

import argparse

import _bench
import _cub

MODEL, COL, LABEL = "ocbm", "CUB", "OCBM v1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    acc_y, acc_c = _cub.run_cell(MODEL, args.iters, epochs=args.epochs,
                                 batch_size=args.batch_size, lam=args.lam,
                                 seed=args.seed)
    _bench.report(LABEL, COL, acc_y, acc_c)


if __name__ == "__main__":
    main()
