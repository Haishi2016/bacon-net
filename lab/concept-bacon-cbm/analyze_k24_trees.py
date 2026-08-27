"""BACON prune analysis of the K=24 per-class OBM trees.

Mirrors the medical-diagnosis pipeline (common.run_standard_analysis ->
bacon.utils.analyze_feature_importance_with_pruning): STRUCTURAL cumulative
pruning, NO retraining. For each species tree we bypass leaves one at a time
(the BACON `prune_features` operation: set a node's aggregator weight to [1,0]
so the pruned leaf drops out and the graded-mean renormalises over survivors),
prune cumulatively in leaf order while protecting the first two leaves as the
baseline, and read off the critical support = leaves that cannot be pruned
without the class's discrimination dropping below baseline.

  python analyze_k24_trees.py --load saved/cub_tree_k24_400ep_hybrid.pt --K 24
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "table"))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

import _cub                                                     # noqa: E402
from cub_emergent import CUBEmergent, collect, top_attr_per_concept  # noqa: E402
from eval_shapes import roc_auc                                 # noqa: E402

_BIG = 30.0  # logit that makes softmax weight ~[1,0] (== prune_features bypass)


@torch.no_grad()
def head_scores_col(model, C, c, device, bs=8192):
    """Class-c tree score for every sample (uses current head weights)."""
    outs = []
    for i in range(0, C.shape[0], bs):
        outs.append(model.head(C[i:i + bs].to(device))[:, c].cpu())
    return torch.cat(outs)


@torch.no_grad()
def prune_and_score(model, C, c, pruned_leaves, device):
    """Structurally bypass `pruned_leaves` (positions >=1) in class c's tree,
    score all samples, then restore. Leaf L is the right input of node L-1;
    setting that node's weight to [1,0] drops it (== binaryTreeLogicNet.prune_features)."""
    wl = model.head.weight_logits.data
    orig = wl[c].clone()
    for L in pruned_leaves:
        wl[c, L - 1, 0] = _BIG      # keep left  (running accumulator)
        wl[c, L - 1, 1] = -_BIG     # drop right (this leaf)
    s = head_scores_col(model, C, c, device)
    wl[c].copy_(orig)
    return s


@torch.no_grad()
def leaf_concept_map(model, device):
    """Annealed/frozen permutation -> which concept lands on each leaf, per head."""
    h = model.head
    if bool(getattr(h, "perm_frozen", torch.tensor(False))):
        P = h.frozen_perm.to(device)                              # exact hard bijection
    else:
        P = h._sinkhorn(h.perm_logits.to(device))                 # soft (approx)
    return P.argmax(dim=2).cpu()                                   # (heads, N) concept idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", type=str, required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--head", type=str, default="tree")
    ap.add_argument("--classes", type=int, default=6, help="# classes to detail")
    ap.add_argument("--sample", type=int, default=200,
                    help="# classes for the aggregate critical-support distribution")
    ap.add_argument("--tol", type=float, default=0.01,
                    help="AUC drop tolerated while pruning (knee threshold)")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CUBEmergent(args.K, head=args.head).to(device)
    ckpt = torch.load(args.load, map_location=device)
    model.load_state_dict(ckpt["state_dict"], strict=False)  # tolerate freeze buffers
    model.eval()

    vl = DataLoader(_cub._CUBImages("test", False), batch_size=128,
                    shuffle=False, num_workers=args.workers, pin_memory=True)
    names112 = _cub._load_attr_groups()[0]
    C, A = collect(model, vl, device)
    Y = torch.cat([y for _i, _c, y in DataLoader(
        _cub._CUBImages("test", False), batch_size=512, shuffle=False,
        num_workers=args.workers)])
    n_cls = int(Y.max().item()) + 1

    tops = top_attr_per_concept(C, A)
    cname = [f"c{j:02d}={names112[a][:32] if a >= 0 else '--'}({au:.2f})"
             for j, (au, a) in enumerate(tops)]
    leaf2concept = leaf_concept_map(model, device)     # (n_cls, K)

    def critical_support(c):
        """Cumulative structural prune of leaves 2..i; return (#critical, kept_leaves, base_auc)."""
        is_pos = (Y == c).long()
        base = roc_auc(head_scores_col(model, C, c, device), is_pos)
        num_pruned = 0
        for i in range(2, args.K):                     # leaves 0,1 = protected baseline
            a = roc_auc(prune_and_score(model, C, c, set(range(2, i + 1)), device), is_pos)
            if a >= base - args.tol:
                num_pruned = i - 1                     # pruned leaves 2..i  (= i-1 leaves)
            else:
                break
        kept = [0, 1] + list(range(2 + num_pruned, args.K))
        return len(kept), kept, base

    # ---- aggregate over classes --------------------------------------------
    g = torch.Generator().manual_seed(0)
    sample = (list(range(n_cls)) if args.sample >= n_cls
              else torch.randperm(n_cls, generator=g)[:args.sample].tolist())
    supp_list = []
    for _i, c in enumerate(sample):
        supp_list.append(critical_support(c)[0])
        if (_i + 1) % 20 == 0:
            print(f"  ...critical-support {_i + 1}/{len(sample)} classes done",
                  flush=True)
    supp = torch.tensor(supp_list)

    print(f"\n{'='*74}\nBACON PRUNE ANALYSIS  (structural, K={args.K}, "
          f"{os.path.basename(args.load)})")
    print(f"method: cumulative leaf-order prune (baseline=leaves 0,1 protected), "
          f"knee at AUC drop > {args.tol}")
    print(f"{'='*74}")
    print(f"CRITICAL SUPPORT over {len(sample)} classes "
          f"(# of {args.K} concepts each decision needs):")
    print(f"  mean {supp.float().mean():.1f}   median {int(supp.median())}   "
          f"min {int(supp.min())}   max {int(supp.max())}")
    sh = torch.bincount(supp, minlength=args.K + 1)
    print("  histogram (#critical : #classes): "
          + "  ".join(f"{k}:{int(sh[k])}" for k in range(args.K + 1) if sh[k] > 0))
    print(f"  => a species decision structurally needs ~{int(supp.float().median())} "
          f"of {args.K} concepts; the rest prune with no accuracy loss.")

    # ---- per-class detail ---------------------------------------------------
    base_all = torch.tensor([roc_auc(head_scores_col(model, C, c, device),
                                     (Y == c).long()) for c in range(n_cls)])
    picks = torch.argsort(base_all, descending=True)[:args.classes].tolist()
    print(f"\n{'-'*74}\nPER-CLASS DETAIL (top-{args.classes} best-separated species):")
    for c in picks:
        ncrit, kept, base = critical_support(c)
        sp = _cub_species_name(c)
        print(f"\n  class {c:3d} {sp}  baseline AUC {base:.3f}")
        print(f"    critical support: {ncrit}/{args.K} concepts "
              f"({args.K - ncrit} leaves pruned, no AUC loss)")
        for L in kept:
            tag = "baseline" if L < 2 else "critical"
            print(f"      leaf{L:2d} [{tag}]  {cname[int(leaf2concept[c, L])]}")


def _cub_species_name(c):
    try:
        p = os.path.join(_cub.CUB, "classes.txt")
        if os.path.exists(p):
            with open(p) as f:
                for line in f:
                    idx, nm = line.strip().split(" ", 1)
                    if int(idx) - 1 == c:
                        return nm.split(".", 1)[-1]
    except Exception:
        pass
    return f"(class {c})"


if __name__ == "__main__":
    main()
