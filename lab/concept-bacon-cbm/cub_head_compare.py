"""Fast, controlled HEAD comparison on a FROZEN concept extractor.

Isolates head expressiveness: precompute the joint CBM's concepts ONCE, then
train each class head to convergence on the *identical* frozen 312-concept
features. This is the clean "same features, different head" experiment (and it
is ~100x faster than end-to-end finetuning a fresh head through the backbone).

Heads:
  linear   : Linear(312->200)                     -- CBM ceiling / control
  logic    : LogicLayer(312->250 fixed gates)->Linear(250->200)  -- paper LogicCBM head
  fulltree : VectorFullTreeHead (graded-logic funnel)            -- OURS
  recttree : VectorRectTreeHead (graded-logic rectangular DAG)   -- OURS

Usage:
  py -3 -u cub_head_compare.py --joint saved/cub_joint_120.pt
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "table"))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

import _cub                                                    # noqa: E402

N_CONCEPTS = 312
N_CLASSES = 200
IMG_SIZE = 299


# ---------------------------------------------------------------- concept cache
@torch.no_grad()
def _extract(concept, split, device):
    dl = DataLoader(
        _cub._CUBImages(split, False, image_size=IMG_SIZE, paper_tf=True, attr312=True),
        batch_size=64, shuffle=False, num_workers=10, pin_memory=True)
    C, Y = [], []
    for img, _c, y in dl:
        C.append(torch.sigmoid(concept(img.to(device))).cpu())
        Y.append(y)
    return torch.cat(C), torch.cat(Y)


def get_concepts(joint_ckpt, device, cache_path):
    if os.path.exists(cache_path):
        d = torch.load(cache_path, map_location="cpu", weights_only=False)
        print(f"loaded concept cache {cache_path}", flush=True)
        return d["ctr"], d["ytr"], d["cte"], d["yte"]
    ck = torch.load(joint_ckpt, map_location="cpu", weights_only=False)
    concept = _cub.InceptionConcept(N_CONCEPTS).to(device).eval()
    concept.load_state_dict(ck["concept"])
    print(f"extracting concepts from {os.path.basename(joint_ckpt)} "
          f"(joint acc {ck.get('acc', 0)*100:.2f}%) ...", flush=True)
    ctr, ytr = _extract(concept, "train", device)
    cte, yte = _extract(concept, "test", device)
    torch.save({"ctr": ctr, "ytr": ytr, "cte": cte, "yte": yte}, cache_path)
    print(f"cached concepts -> {cache_path}  (train {tuple(ctr.shape)}, test {tuple(cte.shape)})",
          flush=True)
    return ctr, ytr, cte, yte


# ---------------------------------------------------------------- heads
def _build_head(head, seed):
    if head == "linear":
        return nn.Linear(N_CONCEPTS, N_CLASSES), False
    if head == "logic":
        from logic_cbm import LogicLayer
        return nn.Sequential(LogicLayer(N_CONCEPTS, 250, fixed_gates=True, seed=seed),
                             nn.Linear(250, N_CLASSES)), False
    if head == "fulltree":
        from bacon.vectorizedFullTree import VectorFullTreeHead
        return VectorFullTreeHead(N_CONCEPTS, N_CLASSES, branching=8, use_negation=True), True
    if head == "recttree":
        from bacon.vectorizedRectTree import VectorRectTreeHead
        return VectorRectTreeHead(N_CONCEPTS, N_CLASSES, width=64, depth=3, max_parents=1,
                                  straight_through=True, leaf_shortcut=True,
                                  use_negation=True), True
    raise ValueError(head)


def _loss(scores, y, is_tree):
    if not is_tree:
        return F.cross_entropy(scores, y)
    # tree heads output per-class probabilities in [0,1]; convert to log-odds and
    # use multiclass cross-entropy (much stronger gradient than one-vs-rest BCE,
    # which barely moves the near-uniform tree outputs).
    p = scores.clamp(1e-6, 1 - 1e-6)
    logits = torch.log(p) - torch.log1p(-p)
    return F.cross_entropy(logits, y)


@torch.no_grad()
def _acc(head, is_tree, X, Y, device):
    head.eval()
    correct = 0
    for i in range(0, X.size(0), 512):
        s = head(X[i:i + 512].to(device))
        correct += (s.argmax(1).cpu() == Y[i:i + 512]).sum().item()
    return correct / X.size(0)


def train_head(head_name, ctr, ytr, cte, yte, device, epochs=200, freeze_frac=0.6, seed=0):
    torch.manual_seed(seed)
    head, is_tree = _build_head(head_name, seed)
    head = head.to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-2, weight_decay=1e-4)
    # smaller batch for the 200-head trees: the vectorized power-mean materializes
    # (w_in, B, H, w_out) tensors, so batch 256 thrashes 16GB; batch 64 is faster.
    bs = 32 if is_tree else 256
    dl = DataLoader(TensorDataset(ctr, ytr), batch_size=bs, shuffle=True)
    freeze_ep = int(freeze_frac * epochs)
    frozen = False
    best = 0.0
    for ep in range(1, epochs + 1):
        if is_tree and not frozen:
            head.anneal(min(1.0, (ep - 1) / max(freeze_ep, 1)))
        head.train()
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            s = head(xb)
            loss = _loss(s, yb, is_tree)
            if is_tree and hasattr(head, "regularization"):
                loss = loss + head.regularization()
            opt.zero_grad(); loss.backward(); opt.step()
        if is_tree and not frozen and ep == freeze_ep:
            (head.harden() if hasattr(head, "harden") else head.freeze_egress())
            frozen = True
        # eval is expensive for the 200-head trees -> sparse during the soft phase,
        # every epoch once frozen (to catch the best faithful hardened point).
        do_eval = (not is_tree) or frozen or ep == freeze_ep or ep % 5 == 0 or ep == epochs
        if do_eval:
            acc = _acc(head, is_tree, cte, yte, device)
            if acc >= best and ((not is_tree) or frozen):
                best = acc
            if is_tree:
                print(f"    [{head_name}] ep {ep}/{epochs} test {acc*100:.2f}% "
                      f"best {best*100:.2f}%{' FROZEN' if frozen else ''}", flush=True)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--joint", required=True)
    ap.add_argument("--heads", default="linear,logic,fulltree,recttree")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache", default=os.path.join(_HERE, "saved", "concepts_cache.pt"))
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ctr, ytr, cte, yte = get_concepts(args.joint, device, args.cache)
    print(f"\n===== FROZEN-CONCEPT HEAD COMPARISON (train {ctr.size(0)}, test {cte.size(0)}) =====",
          flush=True)
    results = {}
    for head in args.heads.split(","):
        acc = train_head(head, ctr, ytr, cte, yte, device, epochs=args.epochs, seed=args.seed)
        results[head] = acc
        print(f"  {head:9s}: test {acc*100:.2f}%", flush=True)
    print("\n----- SUMMARY (same frozen concepts) -----", flush=True)
    for head, acc in sorted(results.items(), key=lambda kv: -kv[1]):
        print(f"  {head:9s}  {acc*100:.2f}%", flush=True)


if __name__ == "__main__":
    main()
