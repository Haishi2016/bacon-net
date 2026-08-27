"""
Zero-shot transfer comparison: OCBM vs CBM vs CREAM, "0" and "8" detectors.

All three models are trained on MNIST DIGIT LABELS ONLY (no stroke-concept
supervision) with the SAME ConceptCNN backbone and the SAME 9 human stroke
concepts, then applied ZERO-SHOT (no retraining) to:

  * synthetic shapes   -> "0"/circle detection  (round vs non-round AUC)
  * Google QuickDraw   -> "0"/circle detection  (round doodles vs non-round AUC)
  * synthetic scenes   -> "8" detection         (2-3 stacked touching circles vs
                                                  single / side-by-side / gapped)

Models (identical supervision, differ only in the concept->task head):
  * OCBM  : fixed AND/OR/NOT BACON logic trees over the concepts (checkpoint.pt).
  * CBM   : sigmoid concepts -> DENSE unmasked linear -> digits (leakage-prone).
  * CREAM : masked C->Y (per-digit reasoning graph) + regularized side-channel.

The "0"/"8" score is the model's belief in that digit:
  OCBM -> the "0"/"8" tree truth (truths[:, d]);  CBM/CREAM -> softmax P(digit d).

    python eval_zeroshot_compare.py
    python eval_zeroshot_compare.py --domains shapes eight     # skip quickdraw
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, _REPO_ROOT)

import config as cfg                                            # noqa: E402
import shapes as shapes_mod                                     # noqa: E402
from model import ConceptCNN                                    # noqa: E402
from eval_shapes import load_model, roc_auc                     # noqa: E402
from eval_cream_zeroshot import CREAMDigit, train_cream, mnist_acc  # noqa: E402
from eval_eight import SCENES, generate_scene                   # noqa: E402
from train import make_loaders                                  # noqa: E402


# --------------------------------------------------------------------------- #
# Vanilla CBM digit model (the leakage-prone baseline): dense unmasked head.
# --------------------------------------------------------------------------- #
class CBMDigit(nn.Module):
    """sigmoid concepts -> DENSE linear -> digits (no reasoning graph, no side)."""

    def __init__(self, concept_names, n_classes=10):
        super().__init__()
        self.concept_names = list(concept_names)
        K = len(concept_names)
        cnn = ConceptCNN(K)
        self.features = cnn.features
        self.fc = nn.Sequential(nn.Flatten(),
                                nn.Linear(64 * 7 * 7, 128), nn.ReLU(), nn.Dropout(0.3))
        self.concept = nn.Linear(128, K)
        self.head = nn.Linear(K, n_classes)

    def forward(self, x):
        c = torch.sigmoid(self.concept(self.fc(self.features(x))))
        return self.head(c), c


# --------------------------------------------------------------------------- #
# model-agnostic digit score + concepts
# --------------------------------------------------------------------------- #
@torch.no_grad()
def digit_score(model, x, digit, kind):
    """Return (score for `digit` in [0,1], concept_probs) for any model."""
    out = model(x)
    if kind == "ocbm":
        _, probs, truths = out
        return truths[:, digit], probs
    logits, c = out
    return torch.softmax(logits, 1)[:, digit], c


def _auc_over_categories(model, kind, imgs_by_cat, pos_set, digit, ci):
    """AUC of the digit-score separating pos_set categories from the rest, plus
    the loop-concept alignment gap (mean loop | pos - mean loop | neg)."""
    lu, ll = ci["loop_upper"], ci["loop_lower"]
    scores, labels, loop_pos, loop_neg = [], [], [], []
    for cat, imgs in imgs_by_cat.items():
        s, probs = digit_score(model, imgs, digit, kind)
        is_pos = 1 if cat in pos_set else 0
        scores.append(s.cpu())
        labels.append(torch.full((len(s),), is_pos))
        loop = 0.5 * (probs[:, lu] + probs[:, ll]).mean().item()
        (loop_pos if is_pos else loop_neg).append(loop)
    auc = roc_auc(torch.cat(scores), torch.cat(labels))
    mp = sum(loop_pos) / max(len(loop_pos), 1)
    mn = sum(loop_neg) / max(len(loop_neg), 1)
    return auc, mp, mn


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--domains", nargs="*",
                    default=["shapes", "quickdraw", "eight"])
    ap.add_argument("--n", type=int, default=400, help="images per category/scene")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--retrain", action="store_true", help="ignore cached CBM/CREAM")
    args = ap.parse_args()
    domains = set(args.domains)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = os.path.join(_REPO_ROOT, "benchmarks", "mnist-addition", "data")
    tl, vl = make_loaders(data, 256)
    rules = {int(k): v for k, v in cfg.DIGIT_RULES.items()}
    ci = {n: i for i, n in enumerate(cfg.CONCEPTS)}

    # ---- OCBM (existing fixed-logic checkpoint) ---- #
    ocbm = load_model(os.path.join(_HERE, "checkpoint.pt"), device)
    ocbm.eval()

    # ---- CBM (dense) + CREAM (masked C->Y + side channel), task-only ---- #
    def get_model(cls_build, path, tag):
        m = cls_build()
        p = os.path.join(_HERE, path)
        if os.path.exists(p) and not args.retrain:
            m.load_state_dict(torch.load(p, map_location=device)); m.to(device)
        else:
            print(f"training {tag} (task-only, no concept supervision)...")
            train_cream(m, tl, device, epochs=args.epochs)
            torch.save(m.state_dict(), p)
        m.eval()
        return m

    cbm = get_model(lambda: CBMDigit(cfg.CONCEPTS), "cbm_digit_zs.pt", "CBM")
    cream = get_model(lambda: CREAMDigit(cfg.CONCEPTS, rules, d_y=20),
                      "cream_digit_zs.pt", "CREAM")

    models = [("OCBM", ocbm, "ocbm"), ("CBM", cbm, "cbm"), ("CREAM", cream, "cream")]

    print("\nMNIST test acc:  " + "   ".join(
        f"{tag} {mnist_acc(m, vl, device) * 100:.2f}" for tag, m, _ in models))

    # ---- "0" / circle detection: synthetic shapes ---- #
    if "shapes" in domains:
        imgs, names = shapes_mod.generate(args.n, seed=123)
        by_cat = {s: imgs[[i for i, n in enumerate(names) if n == s]].to(device)
                  for s in shapes_mod.SHAPES}
        pos = shapes_mod.ROUND_SHAPES
        print("\n=== Synthetic shapes: '0'/circle detection ===")
        print(f"{'model':8s} {'round-vs-rest AUC':>18s} {'loop(round)':>12s} "
              f"{'loop(non)':>10s} {'align-gap':>10s}")
        for tag, m, kind in models:
            auc, mp, mn = _auc_over_categories(m, kind, by_cat, pos, 0, ci)
            print(f"{tag:8s} {auc:18.3f} {mp:12.2f} {mn:10.2f} {mp - mn:10.2f}")

    # ---- "0" / circle detection: QuickDraw ---- #
    if "quickdraw" in domains:
        import quickdraw as qd
        ROUND = ["circle", "donut", "clock", "cookie", "basketball", "pizza"]
        NON_ROUND = ["ladder", "envelope", "line", "pants", "table", "fork", "zigzag"]
        by_cat = {}
        for cat in ROUND + NON_ROUND:
            try:
                by_cat[cat] = qd.load_category(cat, n=args.n).to(device)
            except Exception as e:
                print(f"  skip quickdraw:{cat} ({e})")
        print("\n=== QuickDraw doodles: '0'/circle detection (zero-shot) ===")
        print(f"{'model':8s} {'round-vs-rest AUC':>18s} {'loop(round)':>12s} "
              f"{'loop(non)':>10s} {'align-gap':>10s}")
        for tag, m, kind in models:
            auc, mp, mn = _auc_over_categories(m, kind, by_cat, set(ROUND), 0, ci)
            print(f"{tag:8s} {auc:18.3f} {mp:12.2f} {mn:10.2f} {mp - mn:10.2f}")

    # ---- "8" detection: stacked-circle scenes ---- #
    if "eight" in domains:
        POS = {"2_vstack_touch (8)", "3_vstack_touch",
               "2_vertical_touch", "2_vertical_touch2"}
        NEG = {"1_circle", "2_horizontal_touch", "2_diagonal_touch",
               "2_vertical_gap", "2_vertical_fargap"}
        by_scene = {}
        for name, circles, _grp in SCENES:
            if name in POS or name in NEG:
                by_scene[name] = generate_scene(circles, args.n, seed=7).to(device)
        print("\n=== Stacked-circle scenes: '8' detection (zero-shot) ===")
        print("  positives = 2-3 vertically stacked TOUCHING circles;  "
              "negatives = single / side-by-side / gapped")
        print(f"{'model':8s} {'8-vs-rest AUC':>14s}   per-scene mean 8-score "
              "(pos | neg)")
        for tag, m, kind in models:
            scores, labels = [], []
            pos_means, neg_means = {}, {}
            for name, imgs in by_scene.items():
                s, _ = digit_score(m, imgs, 8, kind)
                is_pos = 1 if name in POS else 0
                scores.append(s.cpu())
                labels.append(torch.full((len(s),), is_pos))
                (pos_means if is_pos else neg_means)[name] = s.mean().item()
            auc = roc_auc(torch.cat(scores), torch.cat(labels))
            pstr = "/".join(f"{v:.2f}" for v in pos_means.values())
            nstr = "/".join(f"{v:.2f}" for v in neg_means.values())
            print(f"{tag:8s} {auc:14.3f}   {pstr}  |  {nstr}")


if __name__ == "__main__":
    main()
