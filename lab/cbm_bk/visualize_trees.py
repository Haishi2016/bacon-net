#!/usr/bin/env python3
"""Visualize / analyze the per-class BACON logic trees from a checkpoint.

A ``bacon_cbm`` checkpoint stores all ``n_classes`` rules inside one vectorized
head, so the standard single-tree tools cannot read it directly. This script
unpacks a chosen class (or all of them) into a scalar ``binaryTreeLogicNet`` via
:meth:`BaconCBM.extract_head_as_scalar_tree` and feeds it to the **existing**
visualization / analysis helpers.

Usage
-----
    # ASCII tree of class 0 in the terminal
    python lab/cbm/visualize_trees.py \
        --config lab/cbm/configs/cub_bacon_paper.yaml \
        --checkpoint runs/cub_bacon_paper/best.pt --class 0 --format ascii

    # PNG figure for class 12
    python lab/cbm/visualize_trees.py --config ... --checkpoint ... \
        --class 12 --format png --out-dir runs/cub_bacon_paper/viz

    # Dump JSON for every class (same artifact train.py writes to trees/)
    python lab/cbm/visualize_trees.py --config ... --checkpoint ... \
        --class all --format json --out-dir runs/cub_bacon_paper/trees
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from cbm_bench import build_dataset, build_model  # noqa: E402
from cbm_bench.config import load_config  # noqa: E402

from bacon.utils import save_tree_structure_to_json  # noqa: E402
from bacon.visualization import (  # noqa: E402
    print_tree_structure,
    visualize_tree_structure,
)


def main() -> None:
    # The existing ASCII tree printer emits Unicode box-drawing/emoji glyphs;
    # the default Windows console code page (cp1252) cannot encode them. Force
    # UTF-8 on stdout so ``--format ascii`` works there.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass

    parser = argparse.ArgumentParser(description="Visualize BACON CBM class trees.")
    parser.add_argument("--config", required=True, help="Path to the training YAML config.")
    parser.add_argument("--checkpoint", required=True, help="Path to a .pt checkpoint.")
    parser.add_argument(
        "--class",
        dest="cls",
        default="0",
        help="Class index to render, or 'all' for every class.",
    )
    parser.add_argument(
        "--format",
        choices=["ascii", "png", "json"],
        default="ascii",
        help="ascii=terminal tree, png=matplotlib figure, json=structured export.",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory for png/html/json (defaults next to the checkpoint).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    bundle = build_dataset(cfg.dataset["name"], cfg.dataset)
    model = build_model(cfg.model["name"], cfg.model, bundle.n_concepts, bundle.n_classes)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model_state"], strict=False)
    model.eval()

    if getattr(model, "vector_net", None) is None or not hasattr(
        model, "extract_head_as_scalar_tree"
    ):
        raise SystemExit(
            "This checkpoint is not a vectorized bacon_cbm model; nothing to extract."
        )

    concept_names = getattr(bundle, "concept_names", None)
    class_names = getattr(bundle, "class_names", None)

    out_dir = args.out_dir or os.path.join(
        os.path.dirname(os.path.abspath(args.checkpoint)), "viz"
    )

    if args.cls == "all":
        indices = list(range(model.n_classes))
    else:
        indices = [int(args.cls)]

    if args.format in ("png", "json"):
        os.makedirs(out_dir, exist_ok=True)

    for k in indices:
        tree = model.extract_head_as_scalar_tree(k)
        cname = class_names[k] if class_names and k < len(class_names) else f"class_{k}"

        if args.format == "ascii":
            print(f"\n===== {cname} (class {k}) =====")
            print_tree_structure(tree, labels=concept_names)
        elif args.format == "json":
            path = os.path.join(out_dir, f"class_{k:03d}.json")
            save_tree_structure_to_json(tree, path, feature_names=concept_names)
            print(f"[json] {path}")
        elif args.format == "png":
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            visualize_tree_structure(tree, labels=concept_names)
            path = os.path.join(out_dir, f"class_{k:03d}.png")
            plt.savefig(path, bbox_inches="tight", dpi=150)
            plt.close("all")
            print(f"[png] {path}")


if __name__ == "__main__":
    main()
