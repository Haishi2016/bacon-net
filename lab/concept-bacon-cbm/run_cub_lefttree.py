r"""Refresh the LEFT/BINARY tree (VectorTreeLogicHead) with the SAME recipe as the
best full tree, for a FAITHFUL ensemble between the two hardened heads.

Matches the full tree's proven soft-train + long-frozen-finetune schedule
(anneal_cap=0.75, freeze_frac=0.65, softer routing), supervised on all 112 CUB
attributes (concept-lam=1.0), 800ep, hardened (Hungarian-frozen permutation ->
faithful per-concept trees). Both heads use the same full_weight aggregator.

    py -3 -u run_cub_lefttree.py --epochs 800 --K 112 --concept-lam 1.0 --workers 4
    -> saved/cub_ocbm_k{K}_tree_sup{lam}_matched_{epochs}ep.pt
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
from cub_emergent import train_one                             # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--K", type=int, default=112)
    ap.add_argument("--concept-lam", type=float, default=1.0,
                    help=">0 pins the K concepts to the K CUB attributes (supervised)")
    # soft-train + long-finetune schedule, matched to the best full tree
    ap.add_argument("--anneal-cap", type=float, default=0.75)
    ap.add_argument("--freeze-frac", type=float, default=0.65)
    ap.add_argument("--perm-sparsity", type=float, default=2.0)
    ap.add_argument("--sinkhorn-iters", type=int, default=100)
    ap.add_argument("--freeze-conf", type=float, default=0.90)
    ap.add_argument("--save-dir", default=os.path.join(_HERE, "saved"))
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sup = args.concept_lam > 0
    ct = torch.arange(args.K) if sup else None
    suptag = f"_sup{args.concept_lam:g}".replace(".", "p") if sup else "_emergent"
    path = os.path.join(args.save_dir,
                        f"cub_ocbm_k{args.K}_tree{suptag}_matched_{args.epochs}ep.pt")
    if os.path.exists(path):
        print(f"checkpoint exists, SKIPPING ({path})", flush=True)
        return

    tl = DataLoader(_cub._CUBImages("train", True), batch_size=args.batch_size,
                    shuffle=True, num_workers=args.workers, pin_memory=True)
    vl = DataLoader(_cub._CUBImages("test", False), batch_size=128,
                    shuffle=False, num_workers=args.workers, pin_memory=True)
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"\n===== k{args.K}_tree (LEFT/binary) matched recipe: sup_lam={args.concept_lam} "
          f"anneal_cap={args.anneal_cap} freeze_frac={args.freeze_frac} "
          f"perm_sparsity={args.perm_sparsity} ({args.epochs}ep, hardened) =====",
          flush=True)
    acc, model = train_one(
        args.K, tl, vl, device, epochs=args.epochs, seed=args.seed,
        head="tree", loss_mode="hybrid", ovr_weight=0.3,
        harden=True, sinkhorn_iters=args.sinkhorn_iters,
        perm_sparsity=args.perm_sparsity, freeze_conf=args.freeze_conf,
        freeze_frac=args.freeze_frac, anneal_cap=args.anneal_cap,
        concept_lam=args.concept_lam, concept_targets=ct,
        ckpt_path=path + ".partial", ckpt_every=50)
    torch.save({"state_dict": model.state_dict(), "K": args.K, "head": "tree",
                "supervised": sup,
                "concept_targets": (ct.tolist() if ct is not None else None)}, path)
    print(f"  DONE k{args.K}_tree matched: acc {acc * 100:.2f}%  saved {path}", flush=True)


if __name__ == "__main__":
    main()
