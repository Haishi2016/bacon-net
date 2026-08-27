"""Visualize how the K emergent concepts segment the digit population.

Loads a hardened MultiTreeBaconCBM, extracts the K concept activations on the
MNIST test set, and renders:
  1. a digit x concept mean-activation HEATMAP with a hierarchical-clustering
     dendrogram (which digits share a concept code), and
  2. a 2D PCA SCATTER of the per-image concept vectors coloured by digit
     (how the digit clouds separate in concept space).

    python visualize_concept_segmentation.py --load saved/k5_harden.pt --k 5
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                # noqa: E402
from scipy.cluster.hierarchy import linkage, dendrogram        # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, _REPO_ROOT)

from train_emergent_concepts import MultiTreeBaconCBM, make_loaders  # noqa: E402


def load_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    m = MultiTreeBaconCBM(ckpt["K"], weight_mode=ckpt.get("weight_mode", "trainable"),
                          device=device).to(device)
    if ckpt.get("frozen"):
        m.prepare_frozen_structure(); m.load_state_dict(ckpt["state_dict"])
    else:
        m.load_state_dict(ckpt["state_dict"]); m.anneal(1.0)
    m.eval()
    return m, ckpt


@torch.no_grad()
def collect(model, loader, device):
    C, Y = [], []
    for x, y in loader:
        C.append(model.concept_probs(x.to(device)).cpu()); Y.append(y)
    return torch.cat(C).numpy(), torch.cat(Y).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", type=str, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--out", type=str, default=os.path.join(_HERE, "results"))
    ap.add_argument("--data", type=str,
                    default=os.path.join(_REPO_ROOT, "benchmarks", "mnist-addition", "data"))
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_model(args.load, device)
    K = ckpt["K"]
    _, test_ld = make_loaders(args.data, 256)
    C, Y = collect(model, test_ld, device)                     # (N,K), (N,)

    per_digit = np.stack([C[Y == d].mean(0) for d in range(10)])   # (10, K)

    # ---- (1) heatmap + dendrogram ------------------------------------------
    Z = linkage(per_digit, method="ward")
    fig = plt.figure(figsize=(1.6 + 0.55 * K, 5.2))
    gsL = fig.add_axes([0.02, 0.12, 0.20, 0.78])
    dn = dendrogram(Z, orientation="left", no_labels=True,
                    color_threshold=0, above_threshold_color="#888", ax=gsL)
    gsL.set_xticks([]); gsL.set_title("digit clusters", fontsize=9)
    # scipy's left-dendrogram lists leaves bottom->top; imshow row 0 is on top,
    # so reverse the leaf order to align heatmap rows with the dendrogram.
    order = [int(t) for t in dn["ivl"]][::-1]
    ax = fig.add_axes([0.30, 0.12, 0.66, 0.78])
    M = per_digit[order]
    im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax.set_yticks(range(10)); ax.set_yticklabels(order)
    ax.set_xticks(range(K)); ax.set_xticklabels([f"c{j}" for j in range(K)])
    ax.set_ylabel("digit (clustered)"); ax.set_xlabel("emergent concept")
    for i in range(10):
        for j in range(K):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    color="w" if M[i, j] < 0.5 else "k", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    fig.suptitle(f"K={K} concept codes per digit  (acc {ckpt.get('acc', 0)*100:.2f}%)",
                 fontsize=11)
    p1 = os.path.join(args.out, f"concept_heatmap_k{K}.png")
    fig.savefig(p1, dpi=140); plt.close(fig)

    # ---- (2) PCA scatter in concept space ----------------------------------
    Xc = C - C.mean(0)
    if K >= 2:
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
        proj = Xc @ Vt[:2].T                                   # (N, 2)
        ev = (S[:2] ** 2) / (S ** 2).sum()
    else:
        proj = np.concatenate([Xc, np.zeros_like(Xc)], 1); ev = [1.0, 0.0]
    fig2, ax2 = plt.subplots(figsize=(6.2, 5.6))
    cmap = plt.get_cmap("tab10")
    idx = np.random.RandomState(0).permutation(len(Y))[:4000]
    ax2.scatter(proj[idx, 0], proj[idx, 1], c=[cmap(int(y)) for y in Y[idx]],
                s=4, alpha=0.35, linewidths=0)
    for d in range(10):
        cen = proj[Y == d].mean(0)
        ax2.text(cen[0], cen[1], str(d), fontsize=15, fontweight="bold",
                 ha="center", va="center",
                 bbox=dict(boxstyle="circle,pad=0.15", fc="white", ec=cmap(d), lw=2))
    ax2.set_xlabel(f"PC1 ({ev[0]*100:.0f}% var)")
    ax2.set_ylabel(f"PC2 ({ev[1]*100:.0f}% var)")
    ax2.set_title(f"K={K} concept space (PCA) — digit clouds")
    p2 = os.path.join(args.out, f"concept_scatter_k{K}.png")
    fig2.savefig(p2, dpi=140, bbox_inches="tight"); plt.close(fig2)

    # ---- text summary: binary code groups ----------------------------------
    codes = (per_digit > 0.5).astype(int)
    groups = {}
    for d in range(10):
        key = "".join(map(str, codes[d]))
        groups.setdefault(key, []).append(d)
    print(f"\nK={K}  saved:\n  {p1}\n  {p2}")
    print(f"\nbinary concept codes (thr 0.5), grouped -> digit clusters:")
    for key, ds in sorted(groups.items()):
        collide = "  <-- COLLISION" if len(ds) > 1 else ""
        print(f"  {key}  ->  digits {ds}{collide}")
    print(f"  distinct codes: {len(groups)}/10")


if __name__ == "__main__":
    main()
