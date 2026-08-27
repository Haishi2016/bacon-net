"""Build the single 'concept grounding' figure for the paper.

One figure, K rows (one per emergent concept), columns:
   [ n max-activating MNIST test digits | occlusion saliency heatmap ]
Row label shows the concept id, its top-digit family + purity, and the
occlusion-peak magnitude.  This is the visual half of the grounding section;
the numeric halves (correlation/VIF, transfer AUC) are tables in the LaTeX.

    python concept_grounding_figure.py --load saved/k5_harden.pt
    -> results/concept_grounding_k5.png
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

from concept_receptive_fields import (load_model, collect, denorm,   # noqa: E402
                                       occlusion_maps)

# human-facing family labels (from receptive-field analysis)
LABELS = {0: "5/9 family", 1: "closed-loop", 2: "'4' (diffuse)",
          3: "'2' family", 4: "7/4 family"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", required=True)
    ap.add_argument("--data", default="./data")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--ncols", type=int, default=5, help="# max-activating imgs shown")
    ap.add_argument("--top", type=int, default=48, help="# top imgs for occlusion avg")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_model(args.load, device)
    K = ckpt["K"]
    from train_emergent_concepts import make_loaders
    _, test_ld = make_loaders(args.data, args.batch_size)
    X, C, Y = collect(model, test_ld, device)

    ncols = args.ncols
    fig, axes = plt.subplots(K, ncols + 1, figsize=(ncols + 2.4, K * 1.15))
    for i in range(K):
        order = C[:, i].argsort(descending=True)
        top_idx = order[:args.top]
        top_digits = Y[top_idx].tolist()
        hist = sorted({d: top_digits.count(d) for d in set(top_digits)}.items(),
                      key=lambda kv: -kv[1])
        purity = 100.0 * sum(n for _, n in hist[:2]) / len(top_digits)

        imgs = denorm(X[top_idx[:ncols]]).numpy()
        for j in range(ncols):
            ax = axes[i, j]
            ax.imshow(imgs[j], cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                fam = LABELS.get(i, "")
                ax.set_ylabel(f"c{i}\n{fam}", fontsize=8, rotation=0,
                              ha="right", va="center", labelpad=22)

        heat = occlusion_maps(model, X[top_idx], i, device).mean(0).numpy()
        axh = axes[i, ncols]
        v = float(np.abs(heat).max()) or 1e-6
        axh.imshow(heat, cmap="bwr", vmin=-v, vmax=v)
        axh.set_xticks([]); axh.set_yticks([])
        top2 = " ".join(f"{d}:{n}" for d, n in hist[:2])
        axh.set_title(f"occ |{v:.2f}|", fontsize=7)
        axes[i, 0].text(-0.02, -0.35, f"top {top2}  ({purity:.0f}%)",
                        transform=axes[i, 0].transAxes, fontsize=6.5, color="k")

    for j, t in enumerate(["max-activating test digits"] + [""] * (ncols - 1) +
                          ["occlusion"]):
        if t:
            axes[0, j].set_title(t, fontsize=8) if j == 0 else None
    fig.suptitle(f"Concept grounding (K={K}): what each emergent concept fires on"
                 " (left) and where (right)", fontsize=10)
    fig.tight_layout(rect=(0.04, 0, 1, 0.97))
    out = args.out or os.path.join(_HERE, "results", f"concept_grounding_k{K}.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
