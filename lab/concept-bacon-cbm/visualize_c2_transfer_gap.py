"""Visualize c2's zero-shot transfer FAILURES to expose an MNIST coverage gap.

c2 ("4" concept) drops from 0.91 (MNIST) to 0.73 (USPS) 4-vs-rest AUC, and its
top-selected USPS digit flips 4 -> 8. This script lays out, in one figure:

  A. MNIST 4s with HIGHEST c2   -- what MNIST *taught* c2 a "4" looks like.
  B. USPS 4s with LOWEST  c2   -- real 4s the concept MISSES (false negatives).
  C. USPS non-4s with HIGHEST c2 -- what fires INSTEAD (false positives; the
                                    true label is printed under each).
  D. USPS 4s with HIGHEST c2   -- the 4s that DID transfer (for contrast).

Comparing A vs B reveals the stylistic gap: if MNIST's high-c2 4s are one glyph
style (e.g. closed-top) and the missed USPS 4s are another (open-top), the
concept latched onto an MNIST-over-represented style -> a sampling gap.

    python visualize_c2_transfer_gap.py --load saved/k5_harden.pt
    -> results/c2_transfer_gap.png
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402

from concept_receptive_fields import load_model, denorm          # noqa: E402
from eval_concept_transfer import usps_loader, _MEAN, _STD       # noqa: E402

CONCEPT = 2      # the "4" concept
TARGET = 4       # its hypothesised digit family


def mnist_loader(root, bs):
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((_MEAN,), (_STD,))])
    ds = datasets.MNIST(root, train=False, download=True, transform=tf)
    return DataLoader(ds, batch_size=bs)


@torch.no_grad()
def collect_xy(model, loader, device, ci):
    xs, cs, ys = [], [], []
    for x, y in loader:
        xs.append(x)
        cs.append(model.concept_probs(x.to(device))[:, ci].cpu())
        ys.append(y)
    return torch.cat(xs), torch.cat(cs), torch.cat(ys)


def strip(ax_row, imgs, scores, labels, title):
    for k, ax in enumerate(ax_row):
        ax.set_xticks([]); ax.set_yticks([])
        if k < len(imgs):
            ax.imshow(imgs[k], cmap="gray", vmin=0, vmax=1)
            tag = f"c2={scores[k]:.2f}"
            if labels is not None:
                tag = f"'{labels[k]}' {tag}"
            ax.set_xlabel(tag, fontsize=6.5)
        else:
            ax.axis("off")
    ax_row[0].set_ylabel(title, fontsize=8, rotation=0, ha="right",
                         va="center", labelpad=48)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", required=True)
    ap.add_argument("--data", default="./data")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--n", type=int, default=10, help="images per row")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_model(args.load, device)
    n = args.n

    Xm, Cm, Ym = collect_xy(model, mnist_loader(args.data, args.batch_size),
                            device, CONCEPT)
    Xu, Cu, Yu = collect_xy(model, usps_loader(args.data, args.batch_size),
                            device, CONCEPT)

    is4m, is4u = Ym == TARGET, Yu == TARGET

    # A: MNIST 4s highest c2
    idx = torch.nonzero(is4m).squeeze(1)
    a = idx[Cm[idx].argsort(descending=True)][:n]
    # B: USPS 4s lowest c2 (missed)
    idx = torch.nonzero(is4u).squeeze(1)
    b = idx[Cu[idx].argsort()][:n]
    # C: USPS non-4s highest c2 (false positives)
    idx = torch.nonzero(~is4u).squeeze(1)
    c = idx[Cu[idx].argsort(descending=True)][:n]
    # D: USPS 4s highest c2 (caught)
    idx = torch.nonzero(is4u).squeeze(1)
    d = idx[Cu[idx].argsort(descending=True)][:n]

    fig, axes = plt.subplots(4, n, figsize=(n * 0.95, 5.2))
    strip(axes[0], denorm(Xm[a]).numpy(), Cm[a].tolist(), None,
          "A  MNIST 4s\nhighest c2\n(what it learned)")
    strip(axes[1], denorm(Xu[b]).numpy(), Cu[b].tolist(), Yu[b].tolist(),
          "B  USPS 4s\nMISSED\n(false neg)")
    strip(axes[2], denorm(Xu[c]).numpy(), Cu[c].tolist(), Yu[c].tolist(),
          "C  USPS non-4s\nfire c2\n(false pos)")
    strip(axes[3], denorm(Xu[d]).numpy(), Cu[d].tolist(), Yu[d].tolist(),
          "D  USPS 4s\nCAUGHT\n(contrast)")

    a4m = (Cm[is4m] > 0.5).float().mean().item()
    a4u = (Cu[is4u] > 0.5).float().mean().item()
    fig.suptitle(f"c2 (\"4\") transfer gap:  MNIST 4s mean c2={Cm[is4m].mean():.2f}"
                 f"  ->  USPS 4s mean c2={Cu[is4u].mean():.2f}"
                 f"   (frac>0.5: {a4m:.2f} -> {a4u:.2f})", fontsize=9)
    fig.tight_layout(rect=(0.10, 0, 1, 0.95))
    out = args.out or os.path.join(_HERE, "results", "c2_transfer_gap.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")
    # quick text summary of the false-positive label mix
    fp_labels = Yu[c].tolist()
    mix = sorted({l: fp_labels.count(l) for l in set(fp_labels)}.items(),
                 key=lambda kv: -kv[1])
    print("false-positive (high-c2 non-4) USPS labels:",
          " ".join(f"{l}:{n_}" for l, n_ in mix))


if __name__ == "__main__":
    main()
