"""Zero-shot the EMERGENT hardened model's own tree-0 / tree-8 detectors.

Unlike eval_shapes/eval_eight/eval_quickdraw (which use the fixed human-authored
tree OCBM), this applies the EMERGENT MultiTreeBaconCBM (k3_harden.pt): a shared
CNN -> 3 unnamed emergent concepts -> 10 hard graded-logic trees. The digit-0
tree (truths[:,0]) is its self-discovered circle detector and the digit-8 tree
(truths[:,8]) its double-loop detector -- neither was trained on shapes/doodles.
We also show the 3 emergent concepts c0/c1/c2 (~closed / curved / cross).

    python eval_emergent_zeroshot.py --load saved/k3_harden.pt
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, _REPO_ROOT)

import shapes as shapes_mod                                     # noqa: E402
import quickdraw as qd                                          # noqa: E402
from train_emergent_concepts import MultiTreeBaconCBM           # noqa: E402
from eval_shapes import roc_auc, best_threshold_accuracy        # noqa: E402
from eval_eight import SCENES, generate_scene                   # noqa: E402

ROUND = ["circle", "donut", "clock", "wheel", "cookie", "basketball", "pizza"]
NON_ROUND = ["ladder", "envelope", "line", "pants", "table", "fork", "pencil", "zigzag"]


def load_emergent(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = MultiTreeBaconCBM(ckpt["K"], weight_mode=ckpt.get("weight_mode", "trainable"),
                              device=device).to(device)
    if ckpt.get("frozen"):
        model.prepare_frozen_structure(); model.load_state_dict(ckpt["state_dict"])
    else:
        model.load_state_dict(ckpt["state_dict"]); model.anneal(1.0)
    model.eval()
    return model, ckpt


@torch.no_grad()
def run(model, imgs, device, bs=1024):
    probs_all, truth_all = [], []
    for i in range(0, len(imgs), bs):
        _, probs, truths = model(imgs[i:i + bs].to(device))
        probs_all.append(probs.cpu()); truth_all.append(truths.cpu())
    return torch.cat(probs_all), torch.cat(truth_all)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", type=str, default=os.path.join(_HERE, "saved", "k3_harden.pt"))
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_emergent(args.load, device)
    K = ckpt["K"]
    cc = " ".join(f"c{j}" for j in range(K))
    print(f"\nEMERGENT model {os.path.basename(args.load)}  K={K}  "
          f"{'HARD' if ckpt.get('frozen') else 'soft'}  acc={ckpt.get('acc', 0) * 100:.2f}%")

    # ---- (1) tree-0 on synthetic shapes ------------------------------------
    imgs, names = shapes_mod.generate(args.n, seed=args.seed)
    probs, truths = run(model, imgs, device)
    s0 = truths[:, 0]
    print("\n=== tree-0 (emergent circle detector) by shape ===")
    print(f"  {'shape':10s} {'0-score':>8s}   {cc}")
    for shape in shapes_mod.SHAPES:
        idx = [i for i, n in enumerate(names) if n == shape]
        tag = " round" if shape in shapes_mod.ROUND_SHAPES else ""
        cmean = " ".join(f"{probs[idx, j].mean():.2f}" for j in range(K))
        print(f"  {shape:10s} {s0[idx].mean():8.3f}   {cmean}{tag}")
    lab = torch.tensor([1 if n in shapes_mod.ROUND_SHAPES else 0 for n in names])
    auc = roc_auc(s0, lab); acc, thr = best_threshold_accuracy(s0, lab)
    print(f"  round vs rest -- ROC-AUC {auc:.3f} | best acc {acc:.3f} @ thr {thr:.3f}")

    # ---- (2) tree-8 on multi-circle scenes ---------------------------------
    print("\n=== tree-8 (emergent double-loop detector) on circle scenes ===")
    print(f"  {'scene':22s} {'8-score':>8s} {'0-score':>8s}   {cc}")
    last = None
    for name, circles, group in SCENES:
        if group != last:
            print(f"  -- axis: {group} " + "-" * (40 - len(group)))
            last = group
        x = generate_scene(circles, args.n, args.seed)
        probs, truths = run(model, x, device)
        cmean = " ".join(f"{probs[:, j].mean():.2f}" for j in range(K))
        print(f"  {name:22s} {truths[:, 8].mean():8.3f} {truths[:, 0].mean():8.3f}   {cmean}")

    # ---- (3) tree-0 on QuickDraw -------------------------------------------
    print("\n=== tree-0 (emergent circle detector) on QuickDraw doodles ===")
    cats = [(c, 1) for c in ROUND] + [(c, 0) for c in NON_ROUND]
    rows, sc, lb = [], [], []
    for cat, is_round in cats:
        try:
            imgs = qd.load_category(cat, n=args.n)
        except Exception as e:
            print(f"  {cat:12s} skip ({e})"); continue
        _, truths = run(model, imgs, device)
        s = truths[:, 0]
        sc.append(s); lb.append(torch.full((len(s),), is_round))
        rows.append((cat, is_round, s.mean().item()))
    scores = torch.cat(sc); labels = torch.cat(lb)
    auc = roc_auc(scores, labels); acc, thr = best_threshold_accuracy(scores, labels)
    print(f"  round vs non-round -- ROC-AUC {auc:.3f} | best acc {acc:.3f} @ thr {thr:.3f}")
    rows.sort(key=lambda r: -r[2])
    print("  ranked by tree-0 score:")
    for cat, is_round, s in rows:
        print(f"    {s:.3f}  {'round' if is_round else '     '}  {cat}")


if __name__ == "__main__":
    main()
