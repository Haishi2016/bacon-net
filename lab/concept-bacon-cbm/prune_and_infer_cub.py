"""Faithful tree pruning on the frozen CUB full-tree head: keep the top-k most
important leaves PER CLASS (by GL global importance -- knockout or Shapley),
run the PRUNED head for inference through the real graded-logic tree, and report
how test accuracy changes vs the unpruned 71.25% head.

Pruning is structural and faithful: each removed leaf-edge is dropped from its
parent power mean and the surviving siblings are renormalized (VectorFullTreeHead
.forward_pruned); keep_mask all-ones reproduces forward() exactly, so any drop is
attributable purely to the removed literals -- this is a real simplified model
used for inference, not a post-hoc explanation.

    py -3 prune_and_infer_cub.py --importance shapley --n-perm 128 --ks 4,6,8,12,16,24
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "table"))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

import _cub                                                    # noqa: E402
from interpret_fulltree_cub import DEFAULT_CKPT, load_model    # noqa: E402
from verbalize_bird_rules import class_mean_concepts           # noqa: E402
from cub_emergent import CUBEmergent                           # noqa: E402

LEFT_CKPT = os.path.join(_HERE, "saved", "cub_ocbm_k112_tree_sup1_matched_800ep.pt")


def load_left(ckpt, device):
    """Load a binary/left-tree CUB head (VectorTreeLogicHead)."""
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    K = int(ck.get("K", 112))
    model = CUBEmergent(K, n_species=200, head="tree")
    missing, unexpected = model.load_state_dict(ck["state_dict"], strict=False)
    missing = [m for m in missing if "num_batches_tracked" not in m]
    if missing or unexpected:
        print(f"  [left load] missing={missing[:3]} unexpected={unexpected[:3]}")
    return model.to(device).eval(), K


@torch.no_grad()
def head_negation(head):
    """Per-head per-concept negation mask [H,K], works for both head types."""
    H, K = head.num_heads, head.input_size
    if getattr(head, "use_negation", False) and bool(getattr(head, "transform_frozen", False)):
        return (head.frozen_transform < 0.5).cpu()            # full tree (frozen 0/1)
    if getattr(head, "transform_logits", None) is not None:
        tw = torch.softmax(head.transform_logits.float(), dim=-1)   # [H,K,2]
        return (tw[..., 1] > 0.5).cpu()                       # left tree (identity/neg)
    return torch.zeros(H, K, dtype=torch.bool)


@torch.no_grad()
def knockout_importance(head, h, baseline, negated_row, device):
    """Drop in head h's root truth when each concept is knocked to its non-
    supporting state (0 for identity, 1 for negated). Works for any head with
    .forward / .input_size."""
    K = head.input_size
    base = baseline.float().cpu()
    X = base.unsqueeze(0).repeat(K + 1, 1)
    for k in range(K):
        X[k + 1, k] = 1.0 if bool(negated_row[k]) else 0.0
    out = head(X.to(device))[:, h].cpu()
    return (out[0] - out[1:]).clamp(min=0.0)


@torch.no_grad()
def shapley_importance(head, h, baseline, negated_row, device, n_perm=128, seed=0):
    """Monte-Carlo permutation Shapley through the frozen head (either type)."""
    K = head.input_size
    present = baseline.float().cpu()
    absent = negated_row.float()                              # non-supporting state
    g = torch.Generator().manual_seed(seed)
    phi = torch.zeros(K)
    steps = torch.arange(K + 1).unsqueeze(1)
    for _ in range(n_perm):
        perm = torch.randperm(K, generator=g)
        rank = torch.empty(K, dtype=torch.long)
        rank[perm] = torch.arange(K)
        mask = rank.unsqueeze(0) < steps                      # [K+1,K]
        X = torch.where(mask, present.unsqueeze(0), absent.unsqueeze(0))
        out = head(X.to(device))[:, h].cpu()
        phi[perm] += out[1:] - out[:-1]
    return (phi / n_perm)


def ig_importance(head, h, baseline, negated_row, device, steps=32):
    """GL-native integrated-gradients importance through the frozen tree.

    Integrates ``d(root truth)/d(leaf)`` -- the andness-weighted path sensitivity
    ``w_i * prod_path (x/m)^(p-1)`` -- along the straight path from each leaf's
    NON-supporting reference (0 identity / 1 negated) to its actual value. This
    is the graded-logic influence the user describes: a leaf on a conjunctive
    path contributes only while it is the binding constraint, and the integral
    over the path weights that by how much it actually binds at this operating
    point. Satisfies completeness: ``sum_i IG_i == f(present) - f(reference)``.
    Returned importance is the positive contribution ``IG_i.clamp(min=0)``.
    """
    K = head.input_size
    present = baseline.float().to(device)
    ref = negated_row.float().to(device)                     # non-supporting state
    diff = present - ref                                     # (K,)
    grad_sum = torch.zeros(K, device=device)
    for s in range(1, steps + 1):
        alpha = float(s) / steps
        x = (ref + alpha * diff).unsqueeze(0).clone().requires_grad_(True)
        out = head(x)[0, h]
        (gx,) = torch.autograd.grad(out, x)
        grad_sum += gx[0].detach()
    ig = diff * (grad_sum / steps)                           # (K,)
    return ig.detach().cpu().clamp(min=0.0)



@torch.no_grad()
def collect_concepts(model, loader, device):
    """Return concept truths C [N,K], labels y [N] over the test set."""
    model.eval()
    cs, ys = [], []
    for img, c, y in loader:
        _, concepts, _ = model(img.to(device))               # (logits, concepts, truths)
        cs.append(concepts.cpu())
        ys.append(y)
    return torch.cat(cs), torch.cat(ys)


@torch.no_grad()
def head_truths(head, C, device, keep_mask=None, bs=512):
    """Run the head over concept matrix C [N,K] -> truths [N,H]."""
    outs = []
    for i in range(0, C.size(0), bs):
        xb = C[i:i + bs].to(device)
        if keep_mask is None:
            t = head(xb)
        else:
            t = head.forward_pruned(xb, keep_mask)
        outs.append(t.cpu())
    return torch.cat(outs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--head", choices=("fulltree", "tree"), default="fulltree",
                    help="which frozen CUB head to prune")
    ap.add_argument("--ckpt", default=None,
                    help="checkpoint (defaults per --head)")
    ap.add_argument("--importance", choices=("knockout", "shapley", "ig"), default="shapley")
    ap.add_argument("--n-perm", type=int, default=128)
    ap.add_argument("--ks", default="4,6,8,12,16,24",
                    help="comma list of leaves-kept-per-class to sweep")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    ks = [int(k) for k in args.ks.split(",")]

    if args.head == "tree":
        ckpt = args.ckpt or LEFT_CKPT
        model, K = load_left(ckpt, device)
        head = model.head
        frozen = bool(getattr(head, "perm_frozen", torch.tensor(True)))
        print(f"loaded LEFT/binary tree K={K} heads={head.num_heads} "
              f"perm_frozen={frozen}")
    else:
        ckpt = args.ckpt or DEFAULT_CKPT
        model, K, branching, negation, coeff, max_egress = load_model(ckpt, device)
        head = model.head
        print(f"loaded FULL tree K={K} heads={head.num_heads} egress={max_egress} "
              f"frozen={bool(head.egress_frozen)}")
    H = head.num_heads

    from torch.utils.data import DataLoader
    loader = DataLoader(_cub._CUBImages("test", False), batch_size=128,
                        shuffle=False, num_workers=args.workers, pin_memory=True)

    print("collecting test concepts ...", flush=True)
    C, y = collect_concepts(model, loader, device)
    print(f"  C={tuple(C.shape)}  y={tuple(y.shape)}")

    # baseline (unpruned) accuracy + equivalence check
    T_full = head_truths(head, C, device)
    acc_full = (T_full.argmax(1) == y).float().mean().item()
    km_all = torch.ones(H, K)
    T_all = head_truths(head, C, device, keep_mask=km_all)
    max_dev = (T_full - T_all).abs().max().item()
    acc_all = (T_all.argmax(1) == y).float().mean().item()
    print(f"\nunpruned head accuracy         : {acc_full*100:.2f}%")
    print(f"forward_pruned(all-ones) acc   : {acc_all*100:.2f}%  "
          f"(max|d-truth| vs forward = {max_dev:.2e}  <- faithfulness check)")

    # per-class importance -> ranking of the K leaves
    print(f"\nranking leaves by {args.importance} importance (per class) ...", flush=True)
    baselines = class_mean_concepts(model, device, args.workers, list(range(H)))
    negated = head_negation(head)                             # [H,K]
    imp = torch.zeros(H, K)
    for h in range(H):
        base = baselines[h]
        if args.importance == "shapley":
            phi = shapley_importance(head, h, base, negated[h], device, n_perm=args.n_perm)
        elif args.importance == "ig":
            phi = ig_importance(head, h, base, negated[h], device)
        else:
            phi = knockout_importance(head, h, base, negated[h], device)
        imp[h] = phi.clamp(min=0.0)
        if (h + 1) % 40 == 0:
            print(f"    {h+1}/{H}", flush=True)
    order = imp.argsort(dim=1, descending=True)                # [H,K]

    print("\n===== accuracy vs leaves-kept-per-class =====")
    print(f"  {'keep/class':>10} {'acc':>7} {'d vs full':>10} {'mean lits kept':>14}")
    print(f"  {'ALL('+str(K)+')':>10} {acc_full*100:6.2f}% {'+0.00':>10} {float(K):>14.1f}")
    for k in ks:
        km = torch.zeros(H, K)
        topk = order[:, :k]
        km.scatter_(1, topk, 1.0)
        T = head_truths(head, C, device, keep_mask=km)
        acc = (T.argmax(1) == y).float().mean().item()
        print(f"  {k:>10} {acc*100:6.2f}% {(acc-acc_full)*100:>+9.2f} {float(k):>14.1f}")


if __name__ == "__main__":
    main()

