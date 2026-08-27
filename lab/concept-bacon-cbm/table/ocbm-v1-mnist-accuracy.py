"""
OCBM v1 on MNIST -> (OCBM v1, MNIST) ACC_Y.

OCBM = our BACON-CBM: fixed per-digit BACON logic trees (config.DIGIT_RULES)
over the 9 stroke concepts, no learned task head.  MNIST's concept set is already
complete (C_true->Y = 100), so v1 reaches high accuracy and v2 collapses to v1.
ACC_C is not reported for MNIST (no per-image concept labels).

    python ocbm-v1-mnist-accuracy.py --iters 10
"""

from __future__ import annotations

import argparse

import _bench

MODEL_FT, COL, LABEL = False, "MNIST", "OCBM v1"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from models import BaconCBM

    device = _bench.get_device()
    spec = _bench.mnist_concept_spec(device)
    print(f"{LABEL} / {COL}   iters={args.iters}  epochs={args.epochs}  "
          f"K={spec.K}  finetune={MODEL_FT}  device={device}")

    acc_y = []
    for it in range(args.iters):
        _bench.set_seed(args.seed + it)
        tl, vl = _bench.mnist_loaders(args.batch_size)
        model = BaconCBM(spec, finetune_logic=MODEL_FT)
        _bench.train_with_concepts(model, tl, spec.target_fn, device,
                                   epochs=args.epochs, lr=args.lr, lam=args.lam)
        a = _bench.task_accuracy(model, vl, device)
        acc_y.append(a)
        print(f"  [iter {it + 1}/{args.iters}]  ACC_Y = {a * 100:.2f}")

    _bench.report(LABEL, COL, acc_y)


if __name__ == "__main__":
    main()
