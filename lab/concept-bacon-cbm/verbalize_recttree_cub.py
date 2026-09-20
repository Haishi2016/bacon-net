r"""Verbalize a recttree species rule with an LLM (prompt + optional --call).

Loads a rectangular graded-logic OCBM checkpoint, extracts a species' frozen
reasoning DAG, prunes it to its most influential children, serializes it to the
LSP/GL verbalization JSON schema, and writes a ready-to-paste LLM prompt per
species. With ``--call`` (and ``anthropic`` + ``ANTHROPIC_API_KEY`` or ``openai``
+ ``OPENAI_API_KEY`` available) it calls the model and also writes the
plain-language report.

The prompt assembly, GCD nomenclature table, and LLM call are reused verbatim
from :mod:`verbalize_bird_rules` (they are head-agnostic -- they operate on the
extracted node-dict schema); only the model loader and tree extractor here are
recttree-specific.

Examples
--------
  py -3 verbalize_recttree_cub.py --ckpt saved/<recttree>.pt --species 13,15 \
      --max-children 3 --max-depth 3
  py -3 verbalize_recttree_cub.py --ckpt saved/<recttree>.pt \
      --species "Indigo Bunting" --call
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from interpret_recttree_cub import (                             # noqa: E402
    load_model, extract_tree, load_species_names, load_concept_names,
    resolve_species)
from verbalize_bird_rules import build_json, build_prompt, call_llm  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--species", default="0",
                    help="index(es), species name, or 'all' (comma-separated)")
    ap.add_argument("--max-children", type=int, default=3,
                    help="max children per node kept in the pruned rule")
    ap.add_argument("--max-depth", type=int, default=3, help="prune depth")
    ap.add_argument("--min-weight", type=float, default=0.0,
                    help="drop child edges below this normalized weight")
    ap.add_argument("--call", action="store_true",
                    help="call an installed LLM SDK (anthropic/openai) and save the report")
    ap.add_argument("--out-dir",
                    default=os.path.join(_HERE, "results", "recttree_verbalized"))
    args = ap.parse_args()

    if not os.path.exists(args.ckpt):
        raise SystemExit(f"checkpoint not found: {args.ckpt}")
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cpu")   # extraction only -- keep the GPU free for training
    species_names = load_species_names()
    model, K, attr312, backbone, negation, coeff = load_model(args.ckpt, device)
    names = load_concept_names(K, attr312)
    heads = resolve_species(args.species, species_names, model.head.num_heads)
    print(f"loaded {os.path.basename(args.ckpt)}  K={K} backbone={backbone} "
          f"species={len(heads)}  (call={'on' if args.call else 'off'})", flush=True)

    for h in heads:
        node = extract_tree(model.head, h)
        tree_json = build_json(node, names, args.max_children, args.max_depth,
                               args.min_weight)
        prompt = build_prompt(species_names[h], tree_json)
        base = os.path.join(args.out_dir, f"species{h:03d}")
        with open(base + "_prompt.md", "w", encoding="utf-8") as f:
            f.write(prompt)
        with open(base + "_rule.json", "w", encoding="utf-8") as f:
            json.dump(tree_json, f, indent=2)
        msg = f"  species {h:>3} ({species_names[h]}): wrote {os.path.basename(base)}_prompt.md/_rule.json"
        if args.call:
            out = call_llm(prompt)
            if out is None:
                msg += "   [--call: no LLM SDK/key found -> prompt only]"
            else:
                with open(base + "_report.md", "w", encoding="utf-8") as f:
                    f.write(out)
                msg += f" + _report.md ({len(out)} chars)"
        print(msg, flush=True)


if __name__ == "__main__":
    main()
