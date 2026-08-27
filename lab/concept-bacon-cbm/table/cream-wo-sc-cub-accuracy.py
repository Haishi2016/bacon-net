"""
CREAM w/o SC on CUB (end-to-end fine-tuned ResNet-18) -> ACC_Y/ACC_C.

CREAM with the side-channel OFF at evaluation (concept-only pathway).

    python cream-wo-sc-cub-accuracy.py --iters 3 --epochs 30
"""

from __future__ import annotations

import argparse

import _bench
import _cub

MODEL, COL = "cream-wo-sc", "CUB"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--ay-mode", type=str, default="full",
                    choices=["positive", "signed", "full"])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    acc_y, acc_c = _cub.run_cell(MODEL, args.iters, epochs=args.epochs,
                                 batch_size=args.batch_size, lam=args.lam,
                                 seed=args.seed, ay_mode=args.ay_mode)
    _bench.report(_cub.LABELS[MODEL], COL, acc_y, acc_c)


if __name__ == "__main__":
    main()
