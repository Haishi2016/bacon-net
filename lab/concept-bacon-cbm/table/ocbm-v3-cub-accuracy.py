"""
OCBM v3 on CUB (end-to-end ResNet-18) -> (OCBM v3, CUB) ACC_Y/ACC_C.

v3 = the same fixed per-class signed-AND tree structure, aggregators
data-fine-tuned (FixedGLTree, gl.generic): each AND node's andness + per-input
weights trained while the topology stays frozen -- relaxes the strict 112-way
conjunction that makes v1 fragile.

    python ocbm-v3-cub-accuracy.py --iters 3 --epochs 40
"""

from __future__ import annotations

import argparse

import _bench
import _cub

MODEL, COL, LABEL = "ocbm-ft", "CUB", "OCBM v3"


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
