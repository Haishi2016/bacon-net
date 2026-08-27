"""
OCBM v3 on CelebA (Smiling, end-to-end ResNet-18) -> ACC_Y/ACC_C.

v3 = the same fixed smile-tree structure, aggregators data-fine-tuned
(FixedGLTree, gl.generic): andness + per-input weights trained while the
topology stays frozen.

    python ocbm-v3-celeba-accuracy.py --iters 5 --epochs 40
"""

from __future__ import annotations

import argparse

import _bench
import _celeba

MODEL, COL, LABEL = "ocbm-ft", "CelebA", "OCBM v3"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--freeze", action="store_true")
    args = ap.parse_args()

    acc_y, acc_c = _celeba.run_cell(MODEL, args.iters, epochs=args.epochs,
                                    batch_size=args.batch_size, lam=args.lam,
                                    seed=args.seed, freeze_backbone=args.freeze)
    _bench.report(LABEL, COL, acc_y, acc_c)


if __name__ == "__main__":
    main()
