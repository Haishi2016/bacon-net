"""
CBM (joint Concept Bottleneck Model) on MNIST  ->  (CBM, MNIST) ACC_Y cell.

Architecture = the canonical Koh et al. 2020 joint CBM: CNN encoder -> sigmoid
concept layer (the bottleneck) -> linear task head.  We reuse the project's
``SoftCBM`` (cream/models.py) so the CBM is identical to the one used in the
iFMNIST / cFMNIST / CUB columns.

MNIST has no per-image concept labels, so -- as a CBM requires concept
supervision -- the bottleneck is supervised against each digit's idealized rule
signature (config.DIGIT_RULES), masking the don't-care (0.5) literals.  Per the
project decision, ACC_C is NOT reported for MNIST (left blank); only ACC_Y.

    python cbm-mnist-accuracy.py --iters 10
"""

from __future__ import annotations

import argparse
import types

import _bench


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=5, help="number of seeds")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam", type=float, default=1.0, help="concept-loss weight")
    ap.add_argument("--seed", type=int, default=0, help="base seed")
    args = ap.parse_args()

    from models import SoftCBM                        # canonical joint CBM

    device = _bench.get_device()

    spec = _bench.mnist_concept_spec(device)          # shared MNIST concept set
    K = spec.K
    target_fn = spec.target_fn                         # y -> (ctgt[y], cmask[y])
    net_spec = types.SimpleNamespace(K=K)

    print(f"CBM (joint) / MNIST   iters={args.iters}  epochs={args.epochs}  "
          f"K={K}  lam={args.lam}  device={device}")
    print(f"  concept supervision: idealized rule signatures, "
          f"{K} concepts, ~{spec.cmask.sum(1).mean():.1f} constrained/digit")

    acc_y = []
    for it in range(args.iters):
        _bench.set_seed(args.seed + it)
        tl, vl = _bench.mnist_loaders(args.batch_size)
        model = SoftCBM(net_spec)
        _bench.train_with_concepts(model, tl, target_fn, device,
                                   epochs=args.epochs, lr=args.lr, lam=args.lam)
        a = _bench.task_accuracy(model, vl, device)
        acc_y.append(a)
        print(f"  [iter {it + 1}/{args.iters}]  ACC_Y = {a * 100:.2f}")

    _bench.report("CBM", "MNIST", acc_y)


if __name__ == "__main__":
    main()
