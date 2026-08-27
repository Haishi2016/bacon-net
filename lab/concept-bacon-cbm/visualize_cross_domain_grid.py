"""Cross-domain grid: what each digit detector fires on across four domains.

10 columns (digit detectors 0-9), 8 rows in four pairs:
   rows 1-2  MNIST      (top-2 true-digit-d test images by tree-d score)
   rows 3-4  USPS       (top-2 true-digit-d USPS images by tree-d score)
   rows 5-6  shape probe(top-2 synthetic shapes by tree-d score)
   rows 7-8  QuickDraw  (top-2 doodles by tree-d score)

Shows, per detector, its in-domain digit, its domain-shifted (USPS) digit, and
the everyday shape / doodle it "sees" as that digit (shape pareidolia).

    python visualize_cross_domain_grid.py --load saved/k5_harden.pt
    -> results/cross_domain_grid.png
"""

from __future__ import annotations

import argparse
import os
import random
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402

from concept_receptive_fields import load_model, denorm         # noqa: E402
from train_emergent_concepts import make_loaders                # noqa: E402
from eval_concept_transfer import usps_loader                   # noqa: E402
import shapes as shapes_mod                                     # noqa: E402
import quickdraw as qd                                          # noqa: E402

QD_CATS = ["circle", "line", "zigzag", "sun", "ladder", "key", "hourglass",
           "snowman", "flower", "banana", "bowtie", "lightning", "wheel",
           "pencil", "moon", "mushroom", "tornado", "clock", "spoon", "star",
           "eyeglasses", "pizza", "triangle", "basketball"]
SHAPE_SET = ["circle", "ellipse", "square", "rectangle", "triangle", "line",
             "cross", "corner", "vee", "zigzag"]


@torch.no_grad()
def truths_of(model, imgs, device, bs=512):
    out = []
    for i in range(0, len(imgs), bs):
        out.append(model(imgs[i:i + bs].to(device))[2].cpu())
    return torch.cat(out)                                        # (N,10)


@torch.no_grad()
def collect_labeled(model, loader, device):
    xs, ts, ys = [], [], []
    for x, y in loader:
        xs.append(x); ts.append(model(x.to(device))[2].cpu()); ys.append(y)
    return torch.cat(xs), torch.cat(ts), torch.cat(ys)


def top2_labeled(X, T, Y, d):
    """Top-2 true-digit-d images by tree-d score."""
    idx = torch.nonzero(Y == d).squeeze(1)
    order = idx[T[idx, d].argsort(descending=True)][:2]
    return denorm(X[order]).numpy()


def top2_pooled(X, T, d):
    """Top-2 images (any label) by tree-d score."""
    order = T[:, d].argsort(descending=True)[:2]
    return denorm(X[order]).numpy()


def top2_category(X, T, snames, cat, d):
    """Top-2 images of a given shape category `cat` by tree-d score."""
    idx = torch.tensor([i for i, n in enumerate(snames) if n == cat])
    order = idx[T[idx, d].argsort(descending=True)][:2]
    return denorm(X[order]).numpy()


def make_multi_circles(n, k, seed=0):
    """n frames of k vertically-stacked touching circles (k=2 -> an '8')."""
    rng = random.Random(seed)
    imgs = []
    for _ in range(n):
        r = rng.uniform(4.2, 5.2)
        ys = [14 + (i - (k - 1) / 2) * 2 * r * 0.98 for i in range(k)]
        imgs.append(shapes_mod.make_circles_tensor(
            [(14.0, y, r) for y in ys], rng, jitter=1.0))
    return torch.stack(imgs)


def make_composite(n, top, bottom, seed=0):
    """n frames of `top` stacked over `bottom` (each: bar / arc / circle)."""
    rng = random.Random(seed)
    return torch.stack([shapes_mod.make_stack_tensor(top, bottom, rng)
                        for _ in range(n)])


# stacked-part probes (top, bottom, label) used as per-digit hypotheses
COMPOSITES = [("bar", "arc", "bar-over-arc"),        # ~ 5
              ("arc", "bar", "arc-over-bar"),        # ~ 2
              ("circle", "arc", "circle-over-arc"),  # ~ 9
              ("arc", "circle", "arc-over-circle"),  # ~ 6
              ("arcL", "arcL", "stacked-arcs"),       # ~ 3 (two ")" bumps)
              ("bar", "diag", "bar-over-diag")]        # ~ 7 (top + "/" descender)

# one hypothesis probe per digit detector: "the shape this rule should fire on"
HYPOTHESIS = {0: "circle", 1: "line", 2: "arc-over-bar", 3: "stacked-arcs",
              4: "cross", 5: "bar-over-arc", 6: "arc-over-circle",
              7: "bar-over-diag", 8: "two-circles", 9: "circle-over-arc"}

# broad QuickDraw category pool -- MUST match scan_quickdraw_detectors.CATEGORIES
# (same order + n) so pooled per-doodle ranks are reproducible.
QD_POOL_CATS = [
    "circle", "square", "triangle", "line", "zigzag", "hexagon", "octagon",
    "star", "clock", "donut", "cookie", "envelope", "ladder", "moon",
    "snowman", "eyeglasses", "hourglass", "key", "candle", "pencil",
    "lightning", "spiral", "wheel", "pizza", "basketball", "tornado",
    "fish", "banana", "hook", "tennis racquet", "stairs", "bowtie",
    "diamond", "cloud", "door", "flower", "leaf", "mushroom", "spoon", "sun",
]

# curated QuickDraw doodles per detector: (pooled rank [0-based], expected cat).
# Leading pooled scores are near-tied, so we hand-pick the most legible doodles.
QD_PICKS = {
    0: [(3, "clock"), (5, "clock")],
    1: [(0, "pencil"), (2, "line")],
    2: [(0, "eyeglasses"), (1, "lightning")],
    3: [(1, "zigzag"), (4, "zigzag")],
    4: [(1, "pencil"), (4, "star")],
    5: [(1, "spoon"), (0, "moon")],
    6: [(0, "pizza"), (2, "banana")],
    7: [(0, "tornado"), (1, "ladder")],
    8: [(0, "hourglass"), (5, "tennis racquet")],
    9: [(0, "key"), (3, None)],
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", required=True)
    ap.add_argument("--data", default="./data")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, _ = load_model(args.load, device)

    _, test_ld = make_loaders(args.data, 512)
    Xm, Tm, Ym = collect_labeled(model, test_ld, device)
    Xu, Tu, Yu = collect_labeled(model, usps_loader(args.data, 512), device)
    # shape probe (basic shapes + stacked-circle + stacked-part composites)
    Xs, snames = shapes_mod.generate(30, seed=0, shapes=SHAPE_SET)
    X2 = make_multi_circles(30, 2, seed=1)
    X3 = make_multi_circles(30, 3, seed=2)
    parts = [Xs, X2, X3]
    snames = snames + ["two-circles"] * 30 + ["three-circles"] * 30
    for si, (top, bot, label) in enumerate(COMPOSITES):
        parts.append(make_composite(30, top, bot, seed=11 + si))
        snames = snames + [label] * 30
    Xs = torch.cat(parts)
    Ts = truths_of(model, Xs, device)
    # report each detector's top shape category (mean score)
    uniq = list(dict.fromkeys(snames))
    Msh = torch.stack([Ts[[i for i, n in enumerate(snames) if n == c]].mean(0)
                       for c in uniq])                          # (n_cat,10)
    print("per-detector top shape (mean tree-d score):")
    for d in range(10):
        j = Msh[:, d].argmax().item()
        two = Msh[uniq.index("two-circles"), d].item()
        print(f"  det {d}: top={uniq[j]}({Msh[j, d]:.2f})  two-circles={two:.2f}")
    # hypothesis validation: does detector d respond to ITS probe, and is it
    # the detector that responds most to that probe (specificity = rank 1)?
    print("hypothesis validation (probe = per-digit shape hypothesis):")
    for d in range(10):
        cat = HYPOTHESIS[d]
        ci = uniq.index(cat)
        s = Msh[ci, d].item()
        rank = int((Msh[ci] > s).sum().item()) + 1   # d's rank on this probe
        argmax_d = Msh[ci].argmax().item()
        if rank == 1:
            tag = "CONFIRM" if s >= 0.30 else "CONFIRM(weak)"
        elif rank <= 3:
            tag = "PARTIAL"
        else:
            tag = "REFUTE"
        print(f"  det {d}: {cat:>16} score={s:.2f} "
              f"rank={rank}/10 (best=det{argmax_d})  [{tag}]")
    # quickdraw pool (broad 40-cat set; pooled per-doodle ranking per detector)
    qd_imgs, qnames = [], []
    for cat in QD_POOL_CATS:
        try:
            x = qd.load_category(cat, n=300)
            qd_imgs.append(x); qnames += [cat] * len(x)
        except Exception as e:
            print(f"  skip {cat}: {e}")
    Xq = torch.cat(qd_imgs)
    Tq = truths_of(model, Xq, device)
    # curated picks: for each detector, the two hand-chosen pooled-rank doodles
    qd_cells = {}
    print("curated QuickDraw picks (pooled rank -> category, score):")
    for d in range(10):
        order = Tq[:, d].argsort(descending=True)
        cells = []
        for rank, expcat in QD_PICKS[d]:
            gi = order[rank].item()
            cat, sc = qnames[gi], Tq[gi, d].item()
            flag = ""
            if expcat is not None and cat != expcat:
                flag = f"  <-- expected {expcat}!"
            print(f"  det {d} rank{rank + 1}: {cat} ({sc:.2f}){flag}")
            cells.append((denorm(Xq[gi:gi + 1])[0].numpy(), f"{cat} ({sc:.2f})"))
        qd_cells[d] = cells

    fig, axes = plt.subplots(8, 10, figsize=(10, 8.4))
    band = [("MNIST", Xm, Tm, Ym), ("USPS", Xu, Tu, Yu),
            ("shape", Xs, Ts, None), ("QuickDraw", Xq, Tq, None)]
    for b, (name, X, T, Y) in enumerate(band):
        for d in range(10):
            row_labels = [None, None]
            if Y is not None:
                imgs = top2_labeled(X, T, Y, d)
            elif name == "shape":
                cat = HYPOTHESIS[d]                       # per-digit hypothesis
                score = Msh[uniq.index(cat), d].item()
                imgs = top2_category(X, T, snames, cat, d)
                row_labels = [None, f"{cat} ({score:.2f})"]
            elif name == "QuickDraw":
                cells = qd_cells[d]                       # curated, both labeled
                imgs = [c[0] for c in cells]
                row_labels = [c[1] for c in cells]
            else:
                imgs = top2_pooled(X, T, d)
            for r in range(2):
                ax = axes[b * 2 + r, d]
                ax.set_xticks([]); ax.set_yticks([])
                if r < len(imgs):
                    ax.imshow(imgs[r], cmap="gray", vmin=0, vmax=1)
                if d == 0 and r == 0:
                    ax.set_ylabel(name, fontsize=9, rotation=90, labelpad=6)
                if row_labels[r] is not None:
                    ax.set_xlabel(row_labels[r], fontsize=6.5, labelpad=2)
    for d in range(10):
        axes[0, d].set_title(str(d), fontsize=11)
    fig.suptitle("What each digit detector fires on across domains "
                 "(top-2 by tree-$d$ score)", fontsize=11)
    fig.tight_layout(rect=(0.02, 0, 1, 0.97))
    out = args.out or os.path.join(_HERE, "results", "cross_domain_grid.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
