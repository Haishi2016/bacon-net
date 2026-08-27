"""
Blackbox on MNIST  ->  fills the (Blackbox, MNIST) ACC_Y cell.

A plain CNN -> 10 classes, trained on digit labels only.  The Blackbox has no
concept layer, so ACC_C is undefined for this row and is left blank ("--").

    python blackbox-mnist-accuracy.py --iters 5 --epochs 6

Runs the model for --iters random seeds and prints ACC_Y as mean +/- std plus a
LaTeX-ready table cell.
"""

from __future__ import annotations

import argparse

import _bench


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=5, help="number of seeds")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0, help="base seed")
    args = ap.parse_args()

    from models import BlackBox                       # CNN backbone -> 10 classes

    device = _bench.get_device()
    print(f"Blackbox / MNIST   iters={args.iters}  epochs={args.epochs}  device={device}")

    acc_y = []
    for it in range(args.iters):
        _bench.set_seed(args.seed + it)
        tl, vl = _bench.mnist_loaders(args.batch_size)
        model = BlackBox()
        _bench.train_task_only(model, tl, device, epochs=args.epochs, lr=args.lr)
        a = _bench.task_accuracy(model, vl, device)
        acc_y.append(a)
        print(f"  [iter {it + 1}/{args.iters}]  ACC_Y = {a * 100:.2f}")

    _bench.report("Blackbox", "MNIST", acc_y)


if __name__ == "__main__":
    main()
