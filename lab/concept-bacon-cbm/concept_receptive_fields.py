"""Ground emergent concepts in the model's own input distribution.

Instead of synthetic shape prototypes (which conflate prototype co-activation
with distributional correlation), we interpret each emerged concept c_i using
the REAL MNIST test images and the REAL frozen encoder, via three views:

  1. MAX/MIN ACTIVATING IMAGES  -- the test digits that most / least switch
     c_i on.  (results/rf_max_c{i}.png)

  2. OCCLUSION SALIENCY  -- slide a small occluder patch over each top image and
     measure the drop in c_i; averaging the drop-maps over the top images gives
     a spatial heatmap of WHERE on the 28x28 canvas the concept reads.
     (results/rf_occ_c{i}.png)   Positive = removing ink there lowers c_i
     (supporting evidence); negative = removing ink there raises c_i.

  3. RECEPTIVE-FIELD PATCH CLUSTERS  -- from each top image, crop the patch at
     the peak of its occlusion map (the region most responsible for c_i),
     cluster those patches (k-means), and show the cluster exemplars.  This is
     the faithful version of "cluster real patches and correlate with the
     concept": the patches are selected BY the attribution, so a human can look
     at the montage and assign a label.  (results/rf_patches_c{i}.png)

    python concept_receptive_fields.py --load saved/k5_harden.pt

All views are per-concept; a human inspects the three panels and assigns a name.
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, _REPO_ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402

from train_emergent_concepts import MultiTreeBaconCBM, make_loaders  # noqa: E402

# MNIST normalization used by make_loaders.
_MEAN, _STD = 0.1307, 0.3081
_BG = (0.0 - _MEAN) / _STD          # normalized value of a black (ink-free) pixel


def load_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    m = MultiTreeBaconCBM(ckpt["K"], weight_mode=ckpt.get("weight_mode", "trainable"),
                          no_negation=ckpt.get("no_negation", False),
                          device=device).to(device)
    if ckpt.get("frozen"):
        m.prepare_frozen_structure(); m.load_state_dict(ckpt["state_dict"])
    else:
        m.load_state_dict(ckpt["state_dict"]); m.anneal(1.0)
    m.eval()
    return m, ckpt


def denorm(x):
    """(.,1,28,28) normalized -> (.,28,28) in [0,1] for display."""
    return (x * _STD + _MEAN).clamp(0, 1).squeeze(1)


@torch.no_grad()
def collect(model, loader, device):
    xs, cs, ys = [], [], []
    for x, y in loader:
        xs.append(x)
        cs.append(model.concept_probs(x.to(device)).cpu())
        ys.append(y)
    return torch.cat(xs), torch.cat(cs), torch.cat(ys)


@torch.no_grad()
def occlusion_maps(model, imgs, ci, device, patch=6, stride=2):
    """For each image, per-pixel drop in concept ci when an occluder of size
    `patch` (set to background) is centered there.  Returns (N,28,28)."""
    N = imgs.shape[0]
    base = model.concept_probs(imgs.to(device))[:, ci].cpu()      # (N,)
    H = W = 28
    heat = torch.zeros(N, H, W)
    cnt = torch.zeros(H, W)
    positions = list(range(0, H - patch + 1, stride))
    for top in positions:
        for left in positions:
            occ = imgs.clone()
            occ[:, :, top:top + patch, left:left + patch] = _BG
            val = model.concept_probs(occ.to(device))[:, ci].cpu()  # (N,)
            drop = base - val                                       # >0 => support
            heat[:, top:top + patch, left:left + patch] += drop[:, None, None]
            cnt[top:top + patch, left:left + patch] += 1
    heat /= cnt.clamp_min(1)[None]
    return heat


def montage(images, title, path, ncol=8, cmap="gray"):
    n = len(images)
    nrow = (n + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(ncol, nrow + 0.4))
    axes = np.array(axes).reshape(-1)
    for k, ax in enumerate(axes):
        ax.axis("off")
        if k < n:
            ax.imshow(images[k], cmap=cmap, vmin=0, vmax=1)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def heat_panel(mean_heat, title, path):
    fig, ax = plt.subplots(figsize=(3, 3))
    v = float(np.abs(mean_heat).max()) or 1e-6
    im = ax.imshow(mean_heat, cmap="bwr", vmin=-v, vmax=v)
    ax.set_title(title, fontsize=10); ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


def kmeans(x, k, iters=25, seed=0):
    """Tiny Lloyd's k-means on (N,D) tensor. Returns (labels, centroids)."""
    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(x.shape[0], generator=g)[:k]
    C = x[idx].clone()
    for _ in range(iters):
        d = torch.cdist(x, C)               # (N,k)
        lab = d.argmin(1)
        for j in range(k):
            m = lab == j
            if m.any():
                C[j] = x[m].mean(0)
    return lab, C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", required=True)
    ap.add_argument("--data", default="./data")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--top", type=int, default=64, help="# top images per concept")
    ap.add_argument("--show", type=int, default=24, help="# images shown in montage")
    ap.add_argument("--patch", type=int, default=8, help="occluder/crop patch size")
    ap.add_argument("--clusters", type=int, default=6)
    ap.add_argument("--outdir", default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_model(args.load, device)
    K = ckpt["K"]
    outdir = args.outdir or os.path.join(_HERE, "results")
    os.makedirs(outdir, exist_ok=True)

    _, test_ld = make_loaders(args.data, args.batch_size)
    X, C, Y = collect(model, test_ld, device)          # X normalized, C probs
    print(f"\nRECEPTIVE FIELDS  {os.path.basename(args.load)}  K={K}  N={X.shape[0]}"
          f"  -> {outdir}")

    for i in range(K):
        order = C[:, i].argsort(descending=True)
        top_idx = order[:args.top]
        bot_idx = order[-args.show:]

        top_imgs = denorm(X[top_idx[:args.show]]).numpy()
        bot_imgs = denorm(X[bot_idx]).numpy()
        top_digits = Y[top_idx].tolist()
        # digit histogram among top activators
        hist = {d: top_digits.count(d) for d in sorted(set(top_digits))}
        hist_str = " ".join(f"{d}:{n}" for d, n in
                             sorted(hist.items(), key=lambda kv: -kv[1]))

        montage(top_imgs, f"c{i}  MAX-activating  (top digits: {hist_str})",
                os.path.join(outdir, f"rf_max_c{i}.png"))
        montage(bot_imgs, f"c{i}  MIN-activating",
                os.path.join(outdir, f"rf_min_c{i}.png"))

        # occlusion saliency over the top images
        top_full = X[top_idx]
        heat = occlusion_maps(model, top_full, i, device, patch=args.patch // 2 + 3)
        mean_heat = heat.mean(0).numpy()
        heat_panel(mean_heat, f"c{i}  occlusion saliency  (red=supports)",
                   os.path.join(outdir, f"rf_occ_c{i}.png"))

        # receptive-field patch clusters: crop the peak-attribution patch
        p = args.patch
        patches = []
        for n in range(top_full.shape[0]):
            hm = heat[n]
            # peak location, clamped so the patch stays in-bounds
            flat = hm.flatten().argmax().item()
            r, cc = divmod(flat, 28)
            r = min(max(r - p // 2, 0), 28 - p)
            cc = min(max(cc - p // 2, 0), 28 - p)
            patches.append(denorm(top_full[n:n + 1])[0, r:r + p, cc:cc + p])
        P = torch.stack(patches).reshape(len(patches), -1)
        lab, cent = kmeans(P, min(args.clusters, len(patches)))
        # one exemplar (nearest to centroid) per cluster
        exemplars = []
        for j in range(cent.shape[0]):
            m = lab == j
            if not m.any():
                continue
            members = P[m]
            nearest = (members - cent[j]).pow(2).sum(1).argmin()
            exemplars.append(members[nearest].reshape(p, p).numpy())
        montage(exemplars, f"c{i}  receptive-field patch motifs "
                f"({len(exemplars)} clusters)",
                os.path.join(outdir, f"rf_patches_c{i}.png"), ncol=len(exemplars) or 1)

        print(f"  c{i}: top-digit hist [{hist_str}]  "
              f"saliency|peak|={np.abs(mean_heat).max():.3f}  "
              f"-> rf_max_c{i}.png / rf_occ_c{i}.png / rf_patches_c{i}.png")

    print("\nInspect results/rf_max_c*.png (what fires it), rf_occ_c*.png (where),"
          " rf_patches_c*.png (motif) and assign each concept a human label.")


if __name__ == "__main__":
    main()
