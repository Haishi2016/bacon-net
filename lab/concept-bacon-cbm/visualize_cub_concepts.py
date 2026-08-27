"""Cross-model CUB concept visualization: attribute x model grid + OCBM rule.

For six human CUB attributes spanning distinct families, and each head
(OCBM / MLP / linear), we find the head's best-matching emergent concept (by
direction-agnostic 1-vs-rest AUC) and show that concept's top-activating birds
plus the AUC. This makes the "comparable concepts across heads" finding visual.
A bottom panel shows a readable OCBM species rule -- the auditable-logic
differentiator the black-box heads lack.

    python visualize_cub_concepts.py \
        --ontology saved/cub_tree_k24_800ep_harden.pt \
        --mlp saved/cub_bb_mlp.pt --linear saved/cub_bb_linear.pt --K 24
    -> results/cub_concepts_across_models.png
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from PIL import Image
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "table"))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                  # noqa: E402

import _cub                                                      # noqa: E402
from cub_emergent import CUBEmergent                             # noqa: E402
from ablation_head_cub import CUBBlackBox                        # noqa: E402
from analyze_k24_trees import leaf_concept_map                   # noqa: E402
from eval_shapes import roc_auc                                  # noqa: E402

# six human attributes spanning distinct families (index into the 112 canonical)
ATTRS = [110, 3, 92, 81, 106, 78]
ATTR_LABELS = ["wing pattern:\nstriped", "bill shape:\ncone",
               "primary color:\nyellow", "body shape:\nduck-like",
               "crown color:\nblack", "size:\nsmall"]
# candidate species for the readable rule panel (visually distinctive)
RULE_CANDIDATES = ["Painted_Bunting", "Green_Jay", "American_Goldfinch",
                   "Red_winged_Blackbird", "Indigo_Bunting", "Blue_Jay"]


def load_ontology(path, K, device):
    m = CUBEmergent(K, head="tree").to(device)
    ck = torch.load(path, map_location=device)
    m.load_state_dict(ck["state_dict"], strict=False)            # tolerate freeze buffers
    m.eval()
    return m


def load_bb(path, device):
    ck = torch.load(path, map_location=device)
    m = CUBBlackBox(ck["K"], head=ck["head"]).to(device)
    m.load_state_dict(ck["state_dict"])
    m.eval()
    return m


@torch.no_grad()
def collect(model, loader, device):
    Cs, As, Ys = [], [], []
    for img, c, y in loader:
        Cs.append(model.concept_probs(img.to(device)).cpu())
        As.append(c); Ys.append(y)
    return torch.cat(Cs), torch.cat(As), torch.cat(Ys)


def best_concept(C, A, a):
    """Best-matching concept for attribute a: returns (concept, AUC, direction)."""
    col = A[:, a].long()
    bi, bauc, bdir = -1, 0.5, 1
    if 0 < col.sum() < len(col):
        for i in range(C.shape[1]):
            au = roc_auc(C[:, i], col)
            v = max(au, 1 - au)
            if v > bauc:
                bauc, bi, bdir = v, i, (1 if au >= 0.5 else -1)
    return bi, bauc, bdir


def top_images(C, i, direction, entries, n=3):
    score = C[:, i] * direction
    idx = score.argsort(descending=True)[:n].tolist()
    imgs = []
    for j in idx:
        p = _cub._local_path(entries[j]["img_path"])
        im = Image.open(p).convert("RGB")
        w, h = im.size
        s = min(w, h)
        im = im.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
        imgs.append(im.resize((150, 150)))
    return imgs


def species_index(name):
    with open(os.path.join(_cub.CUB, "classes.txt"), "r", encoding="utf-8") as f:
        for line in f:
            p = line.strip().split(" ", 1)
            if len(p) == 2 and p[1].split(".", 1)[1] == name:
                return int(p[0]) - 1
    return None


def build_rule(model, C, A, Y, names112, device):
    """Readable OCBM rule for the cleanest candidate species: top concepts (by
    species-vs-rest AUC) that the species tree routes on, named by attribute."""
    leaf2c = leaf_concept_map(model, device)                     # (n_cls, K)
    best = None
    for name in RULE_CANDIDATES:
        c = species_index(name)
        if c is None:
            continue
        lab = (Y == c).long()
        if lab.sum() < 3:
            continue
        concepts = leaf2c[c].tolist()
        scored = []
        for ci in set(concepts):
            au = roc_auc(C[:, ci], lab)
            scored.append((max(au, 1 - au), ci, 1 if au >= 0.5 else -1))
        scored.sort(reverse=True)
        top = scored[:5]
        mean_auc = sum(s[0] for s in top) / len(top)
        if best is None or mean_auc > best[0]:
            best = (mean_auc, name, top)
    if best is None:
        return None
    _, name, top = best
    parts = []
    for au, ci, d in top:
        # name concept by its best global attribute
        ai, aauc, adir = -1, 0.5, 1
        for a in range(A.shape[1]):
            col = A[:, a].long()
            if 0 < col.sum() < len(col):
                x = roc_auc(C[:, ci], col)
                if max(x, 1 - x) > aauc:
                    aauc, ai, adir = max(x, 1 - x), a, (1 if x >= 0.5 else -1)
        aname = names112[ai].replace("has_", "").replace("::", ": ") if ai >= 0 else f"c{ci}"
        neg = "\u00ac" if (d < 0) else ""          # concept low on the species
        parts.append(f"{neg}({aname})")
    return name.replace("_", " "), parts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ontology", required=True)
    ap.add_argument("--mlp", required=True)
    ap.add_argument("--linear", required=True)
    ap.add_argument("--K", type=int, default=24)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vl = DataLoader(_cub._CUBImages("test", False), batch_size=128,
                    shuffle=False, num_workers=args.workers, pin_memory=True)
    entries = _cub._CUBImages("test", False).entries
    names112 = _cub._load_attr_groups()[0]

    models = [("OCBM", load_ontology(args.ontology, args.K, device)),
              ("MLP", load_bb(args.mlp, device)),
              ("linear", load_bb(args.linear, device))]
    data = {}
    for name, m in models:
        data[name] = collect(m, vl, device)                      # (C, A, Y)
        print(f"collected {name}")

    nrow, ncol = len(ATTRS), 9                                   # 3 models x 3 imgs
    fig, axes = plt.subplots(nrow, ncol, figsize=(11.5, 9.2))
    for r, a in enumerate(ATTRS):
        for mi, (mname, _) in enumerate(models):
            C, A, _ = data[mname]
            ci, auc, d = best_concept(C, A, a)
            imgs = top_images(C, ci, d, entries) if ci >= 0 else []
            for k in range(3):
                ax = axes[r, mi * 3 + k]
                ax.set_xticks([]); ax.set_yticks([])
                if k < len(imgs):
                    ax.imshow(imgs[k])
                if k == 1:                                       # middle image = label
                    neg = "\u00ac" if d < 0 else ""
                    ax.set_title(f"c{ci}{neg}  AUC {auc:.2f}", fontsize=8)
            if r == 0:
                axes[r, mi * 3 + 1].annotate(
                    mname, xy=(0.5, 1.42), xycoords="axes fraction",
                    ha="center", fontsize=12, fontweight="bold")
        axes[r, 0].set_ylabel(ATTR_LABELS[r], fontsize=8.5, rotation=0,
                              ha="right", va="center", labelpad=32)

    fig.suptitle("Emergent concepts across heads: each head's best-matching "
                 "concept per human attribute (top-activating birds + AUC)",
                 fontsize=11, y=0.99)

    rule = build_rule(models[0][1], *data["OCBM"], names112, device)
    if rule is not None:
        sp, parts = rule
        txt = (r"$\bf{OCBM\ readable\ rule}$   " + sp + r"  $\approx$  "
               + r"$\wedge$".join(parts)
               + "      (graded conjunction over named concepts; "
                 "MLP / linear heads offer no such rule)")
        fig.text(0.5, 0.02, txt, ha="center", va="bottom", fontsize=8.5,
                 wrap=True, bbox=dict(boxstyle="round", fc="#f2f2f2", ec="#999"))

    fig.tight_layout(rect=(0.05, 0.06, 1, 0.96))
    out = args.out or os.path.join(_HERE, "results", "cub_concepts_across_models.png")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
