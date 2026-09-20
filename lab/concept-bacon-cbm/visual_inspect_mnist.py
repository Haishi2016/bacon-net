r"""Visual-inspection step for concept grounding (K=5 MNIST).

Closes the loop between the STRUCTURAL claim ("digit d uses concept c_i") and the
PIXELS: it pulls real test images -- either the top activators of a concept
overall, or the images of a specific class ranked by that concept -- and renders a
labelled contact sheet a human can eyeball to confirm/refute the concept's meaning.

  # what does each concept fire on? (judge the labels)
  py -3 visual_inspect_mnist.py --load saved/k5_harden.pt --all-concepts

  # do the 8s with high c2 really show a horizontal bar?
  py -3 visual_inspect_mnist.py --load saved/k5_harden.pt --concept 2 --digit 8
"""
from __future__ import annotations

import argparse
import math
import os
import sys

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "table"))

from auto_ground_mnist import load_model, make_loaders          # noqa: E402

# default (relabelled) concept names -- override with --names c0=.. etc.
NAMES = {0: "junction", 1: "upper arc/hook", 2: "U-turn (varied dir.)", 3: "arc", 4: "slant bar"}


@torch.no_grad()
def collect(model, loader, device):
    xs, cs, ys = [], [], []
    for x, y in loader:
        xs.append(x)
        cs.append(model.concept_probs(x.to(device)).cpu())
        ys.append(y)
    return torch.cat(xs), torch.cat(cs), torch.cat(ys)


def concept_heatmap(model, x, i):
    """Grad-CAM attribution 'where concept c_i lands': localize the concept to a
    coarse spatial region using the last conv feature map of the encoder. Grads
    are taken on the PRE-sigmoid logit (encoder.head output) so near-1.0 concepts
    still produce signal. Returns a smooth per-image map in [0,1] at input size."""
    enc = model.encoder
    with torch.enable_grad():
        A = enc.features(x).detach().requires_grad_(True)        # (N,64,7,7)
        logit = enc.head(A)[:, i].sum()
        grads, = torch.autograd.grad(logit, A)                   # (N,64,7,7)
    w = grads.mean(dim=(2, 3), keepdim=True)                     # channel weights
    cam = (w * A).sum(1).clamp_min(0)                            # (N,7,7)
    cam = cam - cam.flatten(1).amin(1)[:, None, None]
    cam = cam / cam.flatten(1).amax(1).clamp_min(1e-8)[:, None, None]
    cam = F.interpolate(cam[:, None], size=x.shape[-2:],
                        mode="bilinear", align_corners=False)[:, 0]
    return cam.detach().cpu()


def _overlay(ax, img, heat):
    """Grayscale digit with a warm Grad-CAM heatmap; alpha scales with heat so
    low-attribution pixels stay transparent and the stroke remains visible."""
    ax.imshow(img, cmap="gray")
    ax.imshow(heat, cmap="jet", vmin=0, vmax=1, alpha=(0.6 * heat).clamp(0, 0.7).numpy())


def usage_polarity(model, K):
    """P[i,d] = +1 if digit d's tree uses c_i as identity, -1 if via negation
    (¬c_i). Read from the frozen routing + transformation layer."""
    P = torch.zeros(K, 10)
    for d in range(10):
        t = model.trees[d]
        il = t.input_to_leaf
        if hasattr(il, "P_hard") and il.P_hard is not None:
            leaf_concept = il.P_hard.argmax(1).tolist()
        else:
            Pm = il.sinkhorn(il.logits.detach(), temperature=float(il.temperature),
                             n_iters=il.sinkhorn_iters)
            leaf_concept = Pm.argmax(1).tolist()
        trans = t.transformation_layer.logits.argmax(1).tolist()   # per leaf: 0=id,1=neg
        for leaf, ci in enumerate(leaf_concept):
            P[ci, d] = -1.0 if trans[leaf] == 1 else 1.0
    return P


def contact_sheet(images, acts, title, out, ncol=10):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    n = len(images)
    nrow = max(1, math.ceil(n / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol * 1.05, nrow * 1.2 + 0.4))
    axes = axes.reshape(nrow, ncol)
    for k in range(nrow * ncol):
        ax = axes[k // ncol, k % ncol]
        if k < n:
            ax.imshow(images[k].squeeze(0), cmap="gray")
            ax.set_title(f"{acts[k]:.2f}", fontsize=8)
        ax.axis("off")
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def concept_gallery(X, C, Y, i, name, out, P_col, model, per_digit=8, max_rows=10):
    """Polarity-aware gallery: one row per digit, showing the images that digit's
    RULE prefers -- highest c_i if the tree uses it as identity (+, 'present'),
    lowest c_i if via negation (¬, 'absent'). Rows are ordered + usage first.
    Each image carries a pixel-attribution heatmap overlay ('where c_i lands').
    A clean concept has high mean where used + and low mean where used ¬."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    means = {d: float(C[(Y == d), i].mean()) for d in range(10)}
    order = sorted(range(10), key=lambda d: (-P_col[d].item(), -means[d]))[:max_rows]
    nrow = len(order)
    # gather the selected images per row, then compute attribution in one batch
    sel_rows = []
    for d in order:
        mask = (Y == d).nonzero(as_tuple=True)[0]
        rank = torch.argsort(C[mask, i], descending=P_col[d].item() > 0)
        sel_rows.append(mask[rank][:per_digit])
    flat = torch.cat(sel_rows) if sel_rows else torch.empty(0, dtype=torch.long)
    heat = concept_heatmap(model, X[flat], i) if len(flat) else None
    heat_at = {int(idx): heat[k] for k, idx in enumerate(flat.tolist())}
    fig, axes = plt.subplots(nrow, per_digit,
                             figsize=(per_digit * 1.05 + 1.4, nrow * 1.1 + 0.5))
    axes = axes.reshape(nrow, per_digit)
    for r, d in enumerate(order):
        pos = P_col[d].item() > 0
        sel = sel_rows[r]
        for c in range(per_digit):
            ax = axes[r, c]
            if c < len(sel):
                j = int(sel[c])
                _overlay(ax, X[j].squeeze(0), heat_at[j] * float(C[j, i]))
                ax.set_title(f"{C[j, i]:.2f}", fontsize=7)
            ax.axis("off")
        pol = "present (+)" if pos else "absent (\u00ac)"
        axes[r, 0].text(-0.42, 0.5, f"digit {d}  {pol}\nmean {means[d]:.2f}",
                        transform=axes[r, 0].transAxes, fontsize=8,
                        va="center", ha="right",
                        color=("#0a7d3c" if pos else "#c0392b"))
    # polarity-aware coherence: mean act on +digits vs ¬digits (want a big gap)
    pos_m = [means[d] for d in range(10) if P_col[d] > 0]
    neg_m = [means[d] for d in range(10) if P_col[d] < 0]
    gap = (sum(pos_m) / len(pos_m) if pos_m else 0) - (sum(neg_m) / len(neg_m) if neg_m else 0)
    fig.suptitle(f"c{i} = '{name}'  |  rows: each digit's rule-preferred images; "
                 f"warm overlay = Grad-CAM (where c{i} lands, intensity \u221d activation).  "
                 f"coherence gap = {gap:+.2f}",
                 fontsize=11)
    fig.tight_layout(rect=(0.05, 0, 1, 0.96))
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", default=os.path.join(_HERE, "saved", "k5_harden.pt"))
    ap.add_argument("--data", default=os.path.join(
        os.path.dirname(_HERE), "..", "benchmarks", "mnist-addition", "data"))
    ap.add_argument("--concept", type=int, default=None)
    ap.add_argument("--digit", type=int, default=None)
    ap.add_argument("--all-concepts", action="store_true")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--names", nargs="*", default=[],
                    help="override names, e.g. c2='horizontal bar'")
    ap.add_argument("--outdir", default=os.path.join(_HERE, "results", "visual_inspect"))
    args = ap.parse_args()

    for kv in args.names:
        k, v = kv.split("=", 1)
        NAMES[int(k.lstrip("c"))] = v

    device = torch.device("cpu")
    model, K, acc = load_model(args.load, device)
    model.eval()
    _, test_ld = make_loaders(args.data, 256)
    X, C, Y = collect(model, test_ld, device)
    os.makedirs(args.outdir, exist_ok=True)
    print(f"loaded {os.path.basename(args.load)}  acc={acc*100:.2f}%  "
          f"test images={len(X)}")

    if args.all_concepts:
        P = usage_polarity(model, K)
        print("\n  polarity-aware coherence: mean activation on digits that use the "
              "concept + (identity) minus those that use it \u00ac (negation).")
        print("  a clean concept fires HIGH where used + and LOW where used \u00ac "
              "(big positive gap).")
        for i in range(K):
            means = {d: float(C[(Y == d), i].mean()) for d in range(10)}
            pos = [d for d in range(10) if P[i, d] > 0]
            neg = [d for d in range(10) if P[i, d] < 0]
            mp = sum(means[d] for d in pos) / max(len(pos), 1)
            mn = sum(means[d] for d in neg) / max(len(neg), 1)
            gap = mp - mn
            tag = " [clean]" if gap > 0.3 else (" [INCOHERENT]" if gap < 0.1 else "")
            print(f"  c{i} = {NAMES.get(i, i):<10} gap {gap:+.2f}  "
                  f"(+digits {pos} mean {mp:.2f} | \u00acdigits {neg} mean {mn:.2f}){tag}")
            idx = torch.topk(C[:, i], args.n).indices
            contact_sheet(
                X[idx], C[idx, i].tolist(),
                f"c{i} = '{NAMES.get(i, i)}'  |  top-{args.n} activators "
                f"(digits: {Y[idx].tolist()})",
                os.path.join(args.outdir, f"concept{i}_top.png"))
            concept_gallery(
                X, C, Y, i, NAMES.get(i, str(i)),
                os.path.join(args.outdir, f"concept{i}_bydigit.png"), P[i], model,
                per_digit=args.n if args.n <= 10 else 8)
        return

    if args.concept is not None and args.digit is not None:
        i, d = args.concept, args.digit
        mask = (Y == d).nonzero(as_tuple=True)[0]
        order = mask[torch.argsort(C[mask, i], descending=True)][:args.n]
        contact_sheet(
            X[order], C[order, i].tolist(),
            f"digit {d}  ranked by c{i} = '{NAMES.get(i, i)}'  "
            f"(top-{args.n}; does the concept show?)",
            os.path.join(args.outdir, f"digit{d}_by_concept{i}.png"))
        # also the LOW end, to contrast
        order_lo = mask[torch.argsort(C[mask, i])][:args.n]
        contact_sheet(
            X[order_lo], C[order_lo, i].tolist(),
            f"digit {d}  LOWEST c{i} = '{NAMES.get(i, i)}'  (contrast)",
            os.path.join(args.outdir, f"digit{d}_by_concept{i}_low.png"))
        return

    ap.error("give --all-concepts, or --concept i --digit d")


if __name__ == "__main__":
    main()
