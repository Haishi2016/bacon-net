r"""Figure: a row of (bird photo, its graded-logic rule tree) pairs.

Renders ``n`` species side by side -- each column shows a representative,
correctly-classified CUB test image on top and a COMPACT drawing of that
species' frozen rule tree below (top-weighted children only, so it stays
legible). Uses the same faithful extraction as ``interpret_fulltree_cub.py``.

    py -3 figure_bird_rules.py --species 0,10,50,131 --per-row 4
    py -3 figure_bird_rules.py --n 6 --per-row 3 --out results/rules/birds.png
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import torch                             # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "table"))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

import _cub                              # noqa: E402
from interpret_fulltree_cub import (     # noqa: E402
    DEFAULT_CKPT, load_model, load_species_names, extract_tree,
    andness_label, leaf_summary, parse_species)

_OP_FC = "#dce8f7"
_OP_EC = "#1560b0"
_LIT_POS = "#0a7d3c"
_LIT_NEG = "#c0392b"


def short_attr(name: str) -> str:
    """'has_wing_color::black' -> 'wing color: black' (compact leaf label)."""
    s = name.replace("has_", "").replace("::", ": ").replace("_", " ")
    return s


def wrap(text: str, width: int = 15) -> str:
    words, lines, cur = text.split(" "), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width and cur:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + " " + w).strip()
    if cur:
        lines.append(cur)
    return "\n".join(lines)


# ---------------------------------------------------------- compact display tree
def build_display(node, names, max_children, max_depth, min_weight, depth=0):
    """Prune the extracted DAG into a small tree for drawing."""
    if node["kind"] == "leaf":
        return {"leaf": True, "label": short_attr(names[node["concept"]]),
                "neg": node["neg"], "children": []}
    a = float(node["andness"])
    if depth >= max_depth:
        # collapse the whole subtree to its top path-weighted literals (chips)
        ch = []
        for lit, w in leaf_summary(node, names)[:max_children]:
            neg = lit.startswith("NOT ")
            nm = lit[4:] if neg else lit
            ch.append((w, {"leaf": True, "label": short_attr(nm), "neg": neg,
                           "children": []}))
        return {"leaf": False, "a": a, "label": andness_label(a), "children": ch}
    kids = sorted(node["children"], key=lambda t: -t[0])
    kids = [k for k in kids if k[0] >= min_weight][:max_children]
    ch = [(w, build_display(c, names, max_children, max_depth, min_weight, depth + 1))
          for w, c in kids]
    if not ch:                           # everything pruned -> show top literals
        for lit, w in leaf_summary(node, names)[:max_children]:
            neg = lit.startswith("NOT ")
            nm = lit[4:] if neg else lit
            ch.append((w, {"leaf": True, "label": short_attr(nm), "neg": neg,
                           "children": []}))
    return {"leaf": False, "a": a, "label": andness_label(a), "children": ch}


def layout(dn):
    """Assign (x, depth) to every node; x by leaf order, returns (n_leaves, max_depth)."""
    leaves = [0]
    maxd = [0]

    def assign(n, depth):
        maxd[0] = max(maxd[0], depth)
        if not n["children"]:
            n["x"] = leaves[0]
            n["y"] = depth
            leaves[0] += 1
            return
        for _, c in n["children"]:
            assign(c, depth + 1)
        xs = [c["x"] for _, c in n["children"]]
        n["x"] = sum(xs) / len(xs)
        n["y"] = depth
    assign(dn, 0)
    return leaves[0], maxd[0]


def draw_tree(ax, dn, n_leaves, max_depth):
    ax.axis("off")
    ax.set_xlim(-0.7, max(n_leaves - 1, 1) + 0.7)
    ax.set_ylim(-(max_depth + 1.4), 0.6)

    def yc(depth):
        return -depth

    def node_xy(n):
        """Display position; leaves are staggered vertically to avoid label overlap."""
        x, y = n["x"], yc(n["y"])
        if n["leaf"]:
            y -= 0.65 * (int(round(n["x"])) % 2)     # alternate low/lower
        return x, y

    def draw(n):
        x, y = node_xy(n)
        for w, c in n["children"]:
            cx, cy = node_xy(c)
            ax.plot([x, cx], [y, cy], "-", color="#b8c2cc", lw=0.9, zorder=1)
            ax.text((x + cx) / 2, (y + cy) / 2, f"{w:.2f}", fontsize=6,
                    color="#8a94a0", ha="center", va="center", zorder=2,
                    bbox=dict(boxstyle="round,pad=0.05", fc="white", ec="none",
                              alpha=0.7))
            draw(c)
        if n["leaf"]:
            col = _LIT_NEG if n["neg"] else _LIT_POS
            lab = ("\u00ac " if n["neg"] else "") + wrap(n["label"], 12)
            ax.text(x, y, lab, fontsize=6, color=col, ha="center", va="top",
                    zorder=3, linespacing=0.9)
        else:
            ax.text(x, y, f"{n['label']}\n{n['a']:+.2f}", fontsize=7.5,
                    color=_OP_EC, ha="center", va="center", zorder=3, weight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", fc=_OP_FC, ec=_OP_EC, lw=1.1))
    draw(dn)


# ------------------------------------------------------------- representative imgs
@torch.no_grad()
def pick_images(model, heads, device, workers):
    """First correctly-classified test image per requested species (fallback: any)."""
    from torch.utils.data import DataLoader
    loader = DataLoader(_cub._CUBImages("test", False), batch_size=128,
                        shuffle=False, num_workers=workers, pin_memory=True)
    mean = torch.tensor(_cub._MEAN).view(3, 1, 1)
    std = torch.tensor(_cub._STD).view(3, 1, 1)
    hset = set(heads)
    chosen = {}                          # h -> (img_hwc, correct)
    for img, c, y in loader:
        logits = model(img.to(device))[0]
        pred = logits.argmax(1).cpu()
        for b in range(img.size(0)):
            yl = int(y[b])
            if yl not in hset:
                continue
            ok = int(pred[b]) == yl
            have = chosen.get(yl)
            if have is None or (ok and not have[1]):
                de = (img[b] * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
                chosen[yl] = (de, ok)
        if all(h in chosen and chosen[h][1] for h in heads):
            break
    return chosen


# ------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--species", default=None,
                    help="comma list of class indices or name substrings")
    ap.add_argument("--n", type=int, default=4,
                    help="number of species if --species not given (0..n-1)")
    ap.add_argument("--per-row", type=int, default=4,
                    help="how many (photo, tree) pairs per row")
    ap.add_argument("--max-children", type=int, default=2)
    ap.add_argument("--max-depth", type=int, default=2)
    ap.add_argument("--min-weight", type=float, default=0.08)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(_HERE, "results", "rules",
                                                   "bird_rule_row.png"))
    ap.add_argument("--dpi", type=int, default=170)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(args.ckpt):
        raise SystemExit(f"checkpoint not found: {args.ckpt}")
    species_names = load_species_names()
    names112, *_ = _cub._load_attr_groups()
    if args.species:
        heads = parse_species(args.species, species_names)
    else:
        heads = list(range(max(1, args.n)))

    model, K, branching, negation, coeff, max_egress = load_model(args.ckpt, device)
    print(f"loaded {os.path.basename(args.ckpt)}  species={heads}")

    imgs = pick_images(model, heads, device, args.workers)

    # pre-build display trees
    disp = {}
    for h in heads:
        dn = build_display(extract_tree(model.head, h), names112,
                           args.max_children, args.max_depth, args.min_weight)
        disp[h] = (dn, *layout(dn))

    per_row = max(1, args.per_row)
    n = len(heads)
    n_blocks = math.ceil(n / per_row)
    # per block: row 0 = photos, row 1 = trees
    fig, axes = plt.subplots(
        n_blocks * 2, per_row,
        figsize=(per_row * 3.7, n_blocks * 5.6),
        gridspec_kw={"height_ratios": [1.0, 1.5] * n_blocks},
        squeeze=False)

    for idx, h in enumerate(heads):
        blk, col = divmod(idx, per_row)
        ax_img = axes[blk * 2][col]
        ax_tree = axes[blk * 2 + 1][col]
        # photo
        ax_img.axis("off")
        if h in imgs:
            de, ok = imgs[h]
            ax_img.imshow(de)
            tag = "" if ok else "  (mispred)"
            ax_img.set_title(f"{species_names[h]}{tag}", fontsize=9)
        else:
            ax_img.set_title(species_names[h], fontsize=9)
        # tree
        dn, nl, md = disp[h]
        draw_tree(ax_tree, dn, nl, md)

    # blank any unused cells
    for idx in range(n, n_blocks * per_row):
        blk, col = divmod(idx, per_row)
        axes[blk * 2][col].axis("off")
        axes[blk * 2 + 1][col].axis("off")

    fig.suptitle("CUB emergent OCBM (full tree, 71.25%) \u2014 species photo & its "
                 "faithful graded-logic rule (nodes = GCD andness degrees)",
                 fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    print(f"  wrote {args.out}  ({n} species, {per_row}/row)")


if __name__ == "__main__":
    main()
