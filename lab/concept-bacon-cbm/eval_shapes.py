"""
Zero-shot circle detection: apply the MNIST-trained concept encoder + the frozen
"0" BACON tree directly to synthetic geometry shapes, with NO retraining.

    python eval_shapes.py                      # uses checkpoint.pt from train.py

The "0" tree is
    loop_upper AND loop_lower AND NOT horizontal_middle AND NOT vertical_line
so its truth value IS a circle score.  We report that score per shape type and
the circle-vs-rest ROC-AUC / accuracy — all transfer, none of it trained on
shapes.
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

import shapes as shapes_mod                    # noqa: E402
from model import ConceptBaconCBM               # noqa: E402


def load_model(ckpt_path: str, device: str) -> ConceptBaconCBM:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = ConceptBaconCBM(
        ckpt["concepts"], {int(k): v for k, v in ckpt["rules"].items()},
        and_andness=ckpt["and_andness"], or_andness=ckpt["or_andness"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def roc_auc(scores: torch.Tensor, labels: torch.Tensor) -> float:
    """Rank-based ROC-AUC (positive label == 1). No sklearn dependency."""
    order = torch.argsort(scores)
    ranks = torch.empty_like(order, dtype=torch.float)
    ranks[order] = torch.arange(1, len(scores) + 1, dtype=torch.float)
    pos = labels == 1
    n_pos = pos.sum().item()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return (ranks[pos].sum().item() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def best_threshold_accuracy(scores: torch.Tensor, labels: torch.Tensor):
    thr_candidates = torch.unique(scores)
    best_acc, best_t = 0.0, 0.5
    for t in thr_candidates:
        acc = ((scores >= t).long() == labels).float().mean().item()
        if acc > best_acc:
            best_acc, best_t = acc, t.item()
    return best_acc, best_t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, default=os.path.join(_HERE, "checkpoint.pt"))
    ap.add_argument("--n", type=int, default=500, help="samples per shape")
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--preview", action="store_true", help="also save a shape grid PNG")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not os.path.exists(args.ckpt):
        raise SystemExit(f"checkpoint not found: {args.ckpt}\nrun train.py first.")
    model = load_model(args.ckpt, device)

    if args.preview:
        p = os.path.join(_HERE, "shapes_preview.png")
        shapes_mod.save_preview(p, seed=args.seed)
        print(f"saved shape preview -> {p}")

    imgs, names = shapes_mod.generate(args.n, seed=args.seed)
    imgs = imgs.to(device)

    with torch.no_grad():
        probs_all, truth_all = [], []
        for i in range(0, len(imgs), 1024):
            _, probs, truths = model(imgs[i:i + 1024])
            probs_all.append(probs.cpu())
            truth_all.append(truths.cpu())
        probs = torch.cat(probs_all)            # (N, n_concepts)
        truths = torch.cat(truth_all)           # (N, 10)

    circle_score = truths[:, 0]                 # the "0" tree = circle detector
    name_t = names

    print(f"\ncheckpoint: {args.ckpt}   shapes: {args.n}/type   device: {device}")
    print("\n'0'-tree (circle) score by shape:")
    for shape in shapes_mod.SHAPES:
        idx = [i for i, n in enumerate(name_t) if n == shape]
        s = circle_score[idx]
        tag = "  <- round" if shape in shapes_mod.ROUND_SHAPES else ""
        print(f"  {shape:10s} mean {s.mean():.3f}  median {s.median():.3f}{tag}")

    labels = torch.tensor([1 if n in shapes_mod.ROUND_SHAPES else 0 for n in name_t])
    auc = roc_auc(circle_score, labels)
    acc, thr = best_threshold_accuracy(circle_score, labels)
    print(f"\nround (circle/ellipse) vs rest  -- ROC-AUC {auc:.3f} | "
          f"best acc {acc:.3f} @ thr {thr:.3f}")

    # strict circle-only vs polygon/line negatives (drop ellipse from both sides)
    strict_idx = [i for i, n in enumerate(name_t)
                  if n == "circle" or n not in shapes_mod.ROUND_SHAPES]
    s_scores = circle_score[strict_idx]
    s_labels = torch.tensor([1 if name_t[i] == "circle" else 0 for i in strict_idx])
    print(f"circle-only vs polygons/lines   -- ROC-AUC {roc_auc(s_scores, s_labels):.3f}")

    print("\nMean concept activation by shape (cols = concepts):")
    header = "shape      " + " ".join(f"{n[:6]:>6s}" for n in model.concept_names)
    print(header)
    for shape in shapes_mod.SHAPES:
        idx = [i for i, n in enumerate(name_t) if n == shape]
        row = probs[idx].mean(0)
        print(f"  {shape:9s} " + " ".join(f"{v:6.2f}" for v in row.tolist()))


if __name__ == "__main__":
    main()
