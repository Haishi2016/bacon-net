"""Which QuickDraw shapes do the emergent DIGIT detectors fire on?

The frozen K=5 MNIST OBM has ten graded-logic digit trees; truths[:,d] is the
digit-d detector. We already know tree-0 detects circles. Here we sweep many
QuickDraw categories through all ten detectors and, for each detector, find the
QuickDraw shape it most strongly (and unexpectedly) responds to -- i.e. the
everyday doodle that "looks like" that digit to the emergent model.

Outputs a table (top categories per detector) and a figure with, for each
detector, the highest-firing example doodles of its top category.

    python scan_quickdraw_detectors.py --load saved/k5_harden.pt
    -> results/quickdraw_digit_detectors.png
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402

from concept_receptive_fields import load_model, denorm         # noqa: E402
import quickdraw as qd                                          # noqa: E402

# a broad set of shape-like + object doodles (skips any that 404 / time out)
CATEGORIES = [
    "circle", "square", "triangle", "line", "zigzag", "hexagon", "octagon",
    "star", "clock", "donut", "cookie", "envelope", "ladder", "moon",
    "snowman", "eyeglasses", "hourglass", "key", "candle", "pencil",
    "lightning", "spiral", "wheel", "pizza", "basketball", "tornado",
    "fish", "banana", "hook", "tennis racquet", "stairs", "bowtie",
    "diamond", "cloud", "door", "flower", "leaf", "mushroom", "spoon", "sun",
]


@torch.no_grad()
def truths_of(model, imgs, device, bs=512):
    out = []
    for i in range(0, len(imgs), bs):
        _, _, t = model(imgs[i:i + bs].to(device))
        out.append(t.cpu())
    return torch.cat(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--examples", type=int, default=6)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_model(args.load, device)

    # gather mean detector scores per category
    cats, imgs_by_cat, T_by_cat, means = [], {}, {}, []
    all_imgs, all_cat_idx, all_T = [], [], []
    for cat in CATEGORIES:
        try:
            imgs = qd.load_category(cat, n=args.n)
        except Exception as e:
            print(f"  skip {cat}: {e}")
            continue
        t = truths_of(model, imgs, device)              # (m,10)
        cats.append(cat); imgs_by_cat[cat] = imgs; T_by_cat[cat] = t
        means.append(t.mean(0))
        all_imgs.append(imgs); all_T.append(t)
        all_cat_idx.append(torch.full((len(imgs),), len(cats) - 1))
    M = torch.stack(means)                              # (C,10)
    AI = torch.cat(all_imgs); AT = torch.cat(all_T)     # pooled images/truths
    ACI = torch.cat(all_cat_idx)
    print(f"\nQUICKDRAW x DIGIT-DETECTOR  ({len(cats)} categories, n={args.n})")

    # per-detector top categories
    print("\nper detector: top-3 QuickDraw categories by mean detector score:")
    top_cat = {}
    for d in range(10):
        order = M[:, d].argsort(descending=True)
        trio = [(cats[i], M[i, d].item()) for i in order[:3]]
        top_cat[d] = trio[0][0]
        print(f"  det {d}:  " + "   ".join(f"{c} {s:.2f}" for c, s in trio))

    # figure: for each detector, best category's highest-firing examples
    ncol = args.examples
    fig, axes = plt.subplots(10, ncol, figsize=(ncol + 1.5, 10 * 1.05))
    for d in range(10):
        cat = top_cat[d]
        t = T_by_cat[cat][:, d]
        idx = t.argsort(descending=True)[:ncol]
        ex = denorm(imgs_by_cat[cat][idx]).numpy()
        for j in range(ncol):
            ax = axes[d, j]; ax.set_xticks([]); ax.set_yticks([])
            if j < len(ex):
                ax.imshow(ex[j], cmap="gray", vmin=0, vmax=1)
            if j == 0:
                ax.set_ylabel(f"det {d}\n{cat}\n{M[cats.index(cat), d]:.2f}",
                              fontsize=7.5, rotation=0, ha="right", va="center",
                              labelpad=26)
    fig.suptitle("What each emergent DIGIT detector fires on in QuickDraw "
                 "(top category + strongest examples)", fontsize=10)
    fig.tight_layout(rect=(0.06, 0, 1, 0.98))
    out = args.out or os.path.join(_HERE, "results", "quickdraw_digit_detectors.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")

    # ---- pooled figure: highest-firing INDIVIDUAL doodles across ALL categories
    # (surfaces the single cleanest shapes per detector, not category averages)
    fig2, axes2 = plt.subplots(10, ncol, figsize=(ncol + 1.5, 10 * 1.05))
    print("\nper detector: top individual doodles (pooled across categories):")
    for d in range(10):
        idx = AT[:, d].argsort(descending=True)[:ncol]
        ex = denorm(AI[idx]).numpy()
        exc = [cats[ACI[i]] for i in idx]
        print(f"  det {d}:  " + "  ".join(f"{c}({AT[i,d]:.2f})"
              for c, i in zip(exc, idx)))
        for j in range(ncol):
            ax = axes2[d, j]; ax.set_xticks([]); ax.set_yticks([])
            ax.imshow(ex[j], cmap="gray", vmin=0, vmax=1)
            ax.set_xlabel(f"{exc[j]}\n{AT[idx[j], d]:.2f}", fontsize=6)
        axes2[d, 0].set_ylabel(f"det {d}", fontsize=8, rotation=0,
                               ha="right", va="center", labelpad=10)
    fig2.suptitle("Highest-firing individual QuickDraw doodles per digit detector"
                  " (pooled across all categories)", fontsize=10)
    fig2.tight_layout(rect=(0.04, 0, 1, 0.98))
    out2 = out.replace(".png", "_pooled.png")
    fig2.savefig(out2, dpi=150)
    print(f"\nwrote {out2}")

    # ---- table figure: each detector row = its THREE best-matched shape
    # categories (by mean), one representative (highest-firing) doodle each
    fig3, axes3 = plt.subplots(10, 3, figsize=(3 * 1.35 + 1.2, 10 * 1.15))
    for d in range(10):
        order = M[:, d].argsort(descending=True)[:3]
        for j, ci in enumerate(order):
            cat = cats[ci]
            best = T_by_cat[cat][:, d].argmax()
            img = denorm(imgs_by_cat[cat][best:best + 1])[0].numpy()
            ax = axes3[d, j]; ax.set_xticks([]); ax.set_yticks([])
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.set_xlabel(f"{cat}  {M[ci, d]:.2f}", fontsize=7.5)
        axes3[d, 0].set_ylabel(f"det {d}", fontsize=10, rotation=0,
                               ha="right", va="center", labelpad=10)
    fig3.suptitle("Three best-matched QuickDraw shapes per digit detector",
                  fontsize=11)
    fig3.tight_layout(rect=(0.05, 0, 1, 0.98))
    out3 = out.replace(".png", "_table.png")
    fig3.savefig(out3, dpi=150)
    print(f"wrote {out3}")

    # ---- individual-doodle table: each detector row = the three highest-firing
    # INDIVIDUAL doodles (distinct categories for variety), labelled per-image
    fig4, axes4 = plt.subplots(10, 3, figsize=(3 * 1.35 + 1.2, 10 * 1.15))
    print("\nper detector: three top INDIVIDUAL doodles (distinct shapes):")
    for d in range(10):
        order = AT[:, d].argsort(descending=True)
        picked, seen = [], set()
        for i in order.tolist():
            c = cats[ACI[i]]
            if c in seen:
                continue
            seen.add(c); picked.append(i)
            if len(picked) == 3:
                break
        print(f"  det {d}:  " + "   ".join(f"{cats[ACI[i]]} {AT[i, d]:.2f}"
              for i in picked))
        for j, i in enumerate(picked):
            img = denorm(AI[i:i + 1])[0].numpy()
            ax = axes4[d, j]; ax.set_xticks([]); ax.set_yticks([])
            ax.imshow(img, cmap="gray", vmin=0, vmax=1)
            ax.set_xlabel(f"{cats[ACI[i]]}  {AT[i, d]:.2f}", fontsize=7.5)
        axes4[d, 0].set_ylabel(f"det {d}", fontsize=10, rotation=0,
                               ha="right", va="center", labelpad=10)
    fig4.suptitle("Three best-matched individual QuickDraw doodles per digit "
                  "detector", fontsize=11)
    fig4.tight_layout(rect=(0.05, 0, 1, 0.98))
    out4 = out.replace(".png", "_table_individual.png")
    fig4.savefig(out4, dpi=150)
    print(f"wrote {out4}")


if __name__ == "__main__":
    main()
