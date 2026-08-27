"""Driver: measure information leakage across the table models on one dataset.

  py -3 leakage.py --dataset ifmnist --iters 10
  py -3 leakage.py --dataset cub     --iters 10 --epochs 40

Prints, per model, full / cpath / cpath_hard accuracies and the three leakage
measures (leak_ceiling, leak_side, leak_soft) as mean +/- std over seeds.
"""

import argparse

import _leakage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True,
                    choices=["ifmnist", "cfmnist", "cub", "celeba"])
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.dataset in ("ifmnist", "cfmnist"):
        ep = args.epochs or 20
        res = _leakage.run_fmnist(args.dataset, args.iters, epochs=ep, seed=args.seed)
    elif args.dataset == "cub":
        ep = args.epochs or 40
        res = _leakage.run_cub(args.iters, epochs=ep, seed=args.seed)
    else:
        ep = args.epochs or 40
        res = _leakage.run_celeba(args.iters, epochs=ep, seed=args.seed)

    _leakage.summarize(args.dataset, res)


if __name__ == "__main__":
    main()
