r"""Cross-selection probe for the "binary-tree feature selection -> full tree"
hybrid idea. Question: are the features the BINARY tree selects (prune-until-
accuracy-drops) good enough to drive the FULL tree?

For each per-class budget k we prune the FULL tree three ways and compare:
  * FULL @ its OWN top-k features        (upper bound: full picks its favourites)
  * FULL @ BINARY's top-k features       (the hybrid: binary selects, full uses)
  * BINARY @ its OWN top-k                (reference: what binary gets)

Concept i is the same named CUB attribute in both models (supervised concept-lam),
so a per-class keep-mask of attribute indices transfers across the two backbones.
Both heads are frozen; pruning is faithful (forward_pruned).

    py -3 cross_select_prune.py --ks 8,12,16,24,32,48
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from prune_and_infer_cub import (                              # noqa: E402
    DEFAULT_CKPT, LEFT_CKPT, load_model, load_left, head_negation,
    knockout_importance, collect_concepts, head_truths)
import _cub                                                    # noqa: E402
from verbalize_bird_rules import class_mean_concepts           # noqa: E402


def ranking(model, head, device, workers):
    """Per-class knockout-importance ranking [H,K] of the K concepts."""
    H, K = head.num_heads, head.input_size
    baselines = class_mean_concepts(model, device, workers, list(range(H)))
    negated = head_negation(head)
    imp = torch.zeros(H, K)
    for h in range(H):
        imp[h] = knockout_importance(head, h, baselines[h], negated[h], device).clamp(min=0.0)
    return imp.argsort(dim=1, descending=True)                 # [H,K]


def acc_at(head, C, y, order, k, K, device):
    km = torch.zeros(head.num_heads, K)
    km.scatter_(1, order[:, :k], 1.0)
    T = head_truths(head, C, device, keep_mask=km)
    return (T.argmax(1) == y).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", default=DEFAULT_CKPT)
    ap.add_argument("--left", default=LEFT_CKPT)
    ap.add_argument("--ks", default="8,12,16,24,32,48")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    ks = [int(k) for k in args.ks.split(",")]

    from torch.utils.data import DataLoader
    loader = DataLoader(_cub._CUBImages("test", False), batch_size=128,
                        shuffle=False, num_workers=args.workers, pin_memory=True)

    print("loading full tree ...", flush=True)
    mfull, K, *_ = load_model(args.full, device)
    hfull = mfull.head
    print("loading binary tree ...", flush=True)
    mleft, _ = load_left(args.left, device)
    hleft = mleft.head

    print("collecting concepts (full backbone) ...", flush=True)
    Cf, y = collect_concepts(mfull, loader, device)
    print("collecting concepts (binary backbone) ...", flush=True)
    Cl, yl = collect_concepts(mleft, loader, device)
    assert torch.equal(y, yl)

    accf = (head_truths(hfull, Cf, device).argmax(1) == y).float().mean().item()
    accl = (head_truths(hleft, Cl, device).argmax(1) == y).float().mean().item()
    print(f"\nfull tree   (all {K}): {accf*100:.2f}%")
    print(f"binary tree (all {K}): {accl*100:.2f}%")

    print("\nranking features (knockout) for each tree ...", flush=True)
    order_full = ranking(mfull, hfull, device, args.workers)
    order_bin = ranking(mleft, hleft, device, args.workers)

    print("\n===== accuracy vs per-class budget k =====")
    print(f"  {'k':>4} {'FULL@own':>9} {'FULL@binsel':>12} {'BINARY@own':>11}")
    for k in ks:
        a_fo = acc_at(hfull, Cf, y, order_full, k, K, device)
        a_fb = acc_at(hfull, Cf, y, order_bin, k, K, device)
        a_bo = acc_at(hleft, Cl, y, order_bin, k, K, device)
        print(f"  {k:>4} {a_fo*100:8.2f}% {a_fb*100:11.2f}% {a_bo*100:10.2f}%")


if __name__ == "__main__":
    main()
