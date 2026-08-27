"""CREAM on CelebA (Smiling) -> (CREAM, CelebA) ACC_Y/ACC_C. See _celeba.py."""

from __future__ import annotations

import argparse

import _bench
import _celeba

MODEL, COL = "cream", "CelebA"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--freeze", action="store_true", help="freeze backbone (frozen features)")
    args = ap.parse_args()

    acc_y, acc_c = _celeba.run_cell(MODEL, args.iters, epochs=args.epochs,
                                    batch_size=args.batch_size, lam=args.lam,
                                    seed=args.seed, freeze_backbone=args.freeze)
    _bench.report(_celeba.LABELS[MODEL], COL, acc_y, acc_c)


if __name__ == "__main__":
    main()
