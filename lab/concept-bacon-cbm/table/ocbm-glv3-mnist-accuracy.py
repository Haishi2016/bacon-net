"""
OCBM v3 (GL trees) on MNIST -> ACC_Y.

Same hierarchical GL trees as ocbm-glv1-mnist (trees/mnist_v1.json), but the tree
STRUCTURE is frozen while each node's andness and per-input weights are
data-fine-tuned (GLTreeCBM trainable=True).

    python ocbm-glv3-mnist-accuracy.py --iters 10
"""

from __future__ import annotations

import argparse
import json
import os

import _bench
import _gltree

TRAINABLE, COL, LABEL = True, "MNIST", "OCBM v3 (GL)"
_TREES = os.path.join(os.path.dirname(os.path.abspath(__file__)), "trees", "mnist_v1.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--trees", type=str, default=_TREES)
    args = ap.parse_args()

    device = _bench.get_device()
    spec = _bench.mnist_concept_spec(device)
    with open(args.trees, "r", encoding="utf-8") as f:
        trees = json.load(f)
    print(f"{LABEL} / {COL}   iters={args.iters}  epochs={args.epochs}  "
          f"trainable={TRAINABLE}  trees={os.path.basename(args.trees)}  device={device}")

    acc_y = []
    for it in range(args.iters):
        _bench.set_seed(args.seed + it)
        tl, vl = _bench.mnist_loaders(args.batch_size)
        model = _gltree.GLTreeCBM(spec.concept_names, trees, trainable=TRAINABLE)
        _bench.train_with_concepts(model, tl, spec.target_fn, device,
                                   epochs=args.epochs, lr=args.lr, lam=args.lam)
        a = _bench.task_accuracy(model, vl, device)
        acc_y.append(a)
        print(f"  [iter {it + 1}/{args.iters}]  ACC_Y = {a * 100:.2f}")

    _bench.report(LABEL, COL, acc_y)


if __name__ == "__main__":
    main()
