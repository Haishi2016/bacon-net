"""
C_true -> Y on MNIST  ->  fills the (C_true->Y, MNIST) ACC_Y cell.

The C_true->Y reference is a linear probe trained on GROUND-TRUTH concepts (no
image encoder), i.e. it measures how well the label can be recovered from
perfect concepts -- the task-accuracy ceiling implied by the concept set.

MNIST has no per-image concept annotations, so the ground-truth concept vector
of an image is its digit's idealized rule signature from config.DIGIT_RULES
(1.0 positive literal, 0.0 negated literal, 0.5 don't-care).  Since the 10
signatures are distinct, this ceiling is ~100%.

C_true->Y consumes concepts rather than predicting them, so ACC_C is left blank.

    python ctruey-mnist-accuracy.py --iters 5
"""

from __future__ import annotations

import argparse

import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

import _bench


class LinearProbe(nn.Module):
    """Linear classifier on ground-truth concepts; forward -> (logits, None)."""

    def __init__(self, k: int, n_classes: int = 10):
        super().__init__()
        self.lin = nn.Linear(k, n_classes)

    def forward(self, x):
        return self.lin(x), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=5, help="number of seeds")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--seed", type=int, default=0, help="base seed")
    args = ap.parse_args()

    spec = _bench.mnist_concept_spec()              # shared MNIST concept set
    M = spec.M                                      # (10, K) in {0, 0.5, 1}
    K = spec.K
    n_distinct = len({tuple(r) for r in M.tolist()})

    y_train, y_test = _bench.mnist_labels()
    Xtr, Xte = M[y_train], M[y_test]                # true concept vector per image
    device = _bench.get_device()
    print(f"C_true->Y / MNIST   iters={args.iters}  K={K}  "
          f"distinct-signatures={n_distinct}/{M.shape[0]}  device={device}")

    acc_y = []
    for it in range(args.iters):
        _bench.set_seed(args.seed + it)
        tl = DataLoader(TensorDataset(Xtr, y_train),
                        batch_size=args.batch_size, shuffle=True)
        vl = DataLoader(TensorDataset(Xte, y_test), batch_size=512)
        model = LinearProbe(K)
        _bench.train_task_only(model, tl, device, epochs=args.epochs, lr=args.lr)
        a = _bench.task_accuracy(model, vl, device)
        acc_y.append(a)
        print(f"  [iter {it + 1}/{args.iters}]  ACC_Y = {a * 100:.2f}")

    _bench.report("C_true->Y", "MNIST", acc_y)


if __name__ == "__main__":
    main()
