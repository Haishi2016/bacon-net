r"""Compare a single class's learned graded-logic tree across rect-head WIDTHS.

Given several hardened recttree checkpoints that differ only in hidden width
(e.g. w64 / w8 / w4), this renders -- for ONE species -- their faithful GL trees
side by side in a single figure, and writes a side-by-side text report of the
full expression + plain-language verbalization + rule size (and, with --eval,
the frozen test accuracy and that class's recall). The point: show how forcing a
narrower head compresses the rule to fewer concepts with little accuracy loss.

    py -3 compare_widths.py --species Albatross \
        --ckpts saved/w64.pt saved/w8.pt saved/w4.pt --eval --out results/compare_albatross.png
"""
from __future__ import annotations

import argparse
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import torch                             # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "table"))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

import _cub                                                          # noqa: E402
from interpret_recttree_cub import (                                 # noqa: E402
    load_model, load_concept_names, extract_tree, full_expression,
    verbalize_english)
from interpret_fulltree_cub import (                                 # noqa: E402
    load_species_names, parse_species, tree_stats, leaf_summary)
from figure_bird_rules import build_display, layout, draw_tree      # noqa: E402


def _label_from(path):
    m = re.search(r"_w(\d+)_", os.path.basename(path))
    if m:
        return f"width {m.group(1)}"
    m = re.search(r"_br(\d+)_", os.path.basename(path))
    return f"branch {m.group(1)}" if m else os.path.basename(path)[:16]


@torch.no_grad()
def frozen_eval(model, loader, device, species):
    correct = total = 0
    cls_hit = cls_tot = 0
    for img, c, y in loader:
        img = img.to(device)
        pred = model(img)[0].argmax(1).cpu()
        correct += (pred == y).sum().item()
        total += y.numel()
        m = (y == species)
        cls_tot += int(m.sum())
        cls_hit += int((pred[m] == species).sum())
    return 100.0 * correct / max(total, 1), 100.0 * cls_hit / max(cls_tot, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", required=True, help="one class index or name substring")
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--eval", action="store_true", help="also report frozen test acc + class recall")
    ap.add_argument("--tree-children", type=int, default=3)
    ap.add_argument("--tree-depth", type=int, default=3)
    ap.add_argument("--tree-min-weight", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out", default=os.path.join(_HERE, "results", "recttree_rules", "compare_widths.png"))
    ap.add_argument("--txt", default=None, help="side-by-side expression/verbalization text (default: <out>.txt)")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    species_names = load_species_names()
    sidx = parse_species(args.species, species_names)[0]
    labels = args.labels or [_label_from(p) for p in args.ckpts]

    entries = []           # (label, node, n_leaves, names, model, attr312, backbone)
    for path, lab in zip(args.ckpts, labels):
        if not os.path.exists(path):
            print(f"[skip] {lab}: {path} not found")
            continue
        model, K, attr312, backbone, neg, coeff = load_model(path, device)
        names = load_concept_names(K, attr312)
        node = extract_tree(model.head, sidx)
        n_ops, n_leaves, _ = tree_stats(node)
        entries.append({"label": lab, "node": node, "n_leaves": n_leaves,
                        "n_ops": n_ops, "names": names, "model": model,
                        "attr312": attr312, "backbone": backbone})
        print(f"  {lab}: {n_leaves} literals, {n_ops} ops  (widths={model.head.widths})")

    if not entries:
        raise SystemExit("no checkpoints loaded")

    # optional frozen accuracy per model
    accs = {}
    if args.eval:
        from torch.utils.data import DataLoader
        e0 = entries[0]
        image_size = 299 if e0["backbone"] == "inception_v3" else 224
        loader = DataLoader(
            _cub._CUBImages("test", False, attr312=e0["attr312"], image_size=image_size),
            batch_size=64, shuffle=False, num_workers=args.workers, pin_memory=True)
        for e in entries:
            acc, rec = frozen_eval(e["model"], loader, device, sidx)
            accs[e["label"]] = (acc, rec)
            print(f"  {e['label']}: frozen test {acc:.2f}%  |  {species_names[sidx]} recall {rec:.1f}%")

    # ---- figure: trees side by side ----
    ncol = len(entries)
    fig, axes = plt.subplots(1, ncol, figsize=(5.2 * ncol, 6.0), squeeze=False)
    for j, e in enumerate(entries):
        disp = build_display(e["node"], e["names"], args.tree_children,
                             args.tree_depth, args.tree_min_weight)
        nl, md = layout(disp)
        draw_tree(axes[0][j], disp, nl, md)
        acc_txt = ""
        if e["label"] in accs:
            a, r = accs[e["label"]]
            acc_txt = f"   |   test {a:.1f}%  recall {r:.0f}%"
        axes[0][j].set_title(f"{e['label']}   ({e['n_leaves']} literals, "
                             f"{e['n_ops']} ops){acc_txt}", fontsize=11)
    fig.suptitle(f"{species_names[sidx]} (species {sidx}) -- graded-logic rule vs "
                 f"rect-head width  (pruned to top {args.tree_children}/node)", fontsize=13, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out, dpi=130, bbox_inches="tight")
    print(f"\n  wrote {args.out}")

    # ---- side-by-side expression + verbalization text ----
    txt = args.txt or (os.path.splitext(args.out)[0] + ".txt")
    with open(txt, "w", encoding="utf-8") as f:
        f.write(f"{species_names[sidx]} (species {sidx}) -- rule vs rect-head width\n")
        for e in entries:
            f.write("\n" + "=" * 78 + "\n")
            head = f"{e['label']}   {e['n_leaves']} literals, {e['n_ops']} ops"
            if e["label"] in accs:
                a, r = accs[e["label"]]
                head += f"   test {a:.2f}%  recall {r:.1f}%"
            f.write(head + "\n" + "-" * 78 + "\n")
            f.write("full expression:\n  " + full_expression(e["node"], e["names"]) + "\n\n")
            f.write("verbalization:\n" + verbalize_english(e["node"], e["names"], indent=1) + "\n")
    print(f"  wrote {txt}")


if __name__ == "__main__":
    main()
