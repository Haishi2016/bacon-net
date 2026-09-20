r"""Rank species by how much DISJUNCTION (OR-like, andness < 0.5) their frozen
graded-logic rule uses, so we can pick good disjunctive sample figures.

For each class tree: distribute a unit of weight from the root down the
normalized child edges; a node's "influence" is the weight reaching it. We sum
influence over disjunctive op-nodes (andness < 0.5 - margin) -> "disjunctive
mass". We also flag the root's andness (a disjunctive ROOT = an "any of these
profiles" rule, the most interpretable disjunction).

  py -3 find_disjunctive.py --ckpt saved/<recttree>.pt --top 15
"""
from __future__ import annotations

import argparse
import torch

from interpret_recttree_cub import (load_model, extract_tree, load_species_names,
                                     load_concept_names)
from interpret_fulltree_cub import gcd_info

MARGIN = 0.05


def scan(node):
    """Return (root_andness, disj_mass, n_disj, top_disj) for one tree.
    top_disj = (influence, andness, code) of the most influential OR node."""
    root_a = float(node["andness"]) if node["kind"] == "op" else 1.0
    disj_mass = 0.0
    n_disj = 0
    best = (0.0, None, None)

    def rec(n, w):
        nonlocal disj_mass, n_disj, best
        if n["kind"] == "leaf":
            return
        a = float(n["andness"])
        if a < 0.5 - MARGIN:
            disj_mass += w
            n_disj += 1
            if w > best[0]:
                code, _, _ = gcd_info(a)
                best = (w, a, code)
        for cw, ch in n.get("children", []):
            rec(ch, w * cw)

    rec(node, 1.0)
    return root_a, disj_mass, n_disj, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--top", type=int, default=15)
    args = ap.parse_args()

    sp = load_species_names()
    model, K, attr312, *_ = load_model(args.ckpt, torch.device("cpu"))
    head = model.head
    rows = []
    for h in range(head.num_heads):
        node = extract_tree(head, h)
        root_a, dmass, ndisj, best = scan(node)
        rows.append((h, root_a, dmass, ndisj, best))

    print(f"loaded {args.ckpt}\n")
    print("--- species with a DISJUNCTIVE ROOT (root andness < 0.5) ---")
    droot = sorted([r for r in rows if r[1] < 0.5], key=lambda r: r[1])
    for h, ra, dm, nd, best in droot[:args.top]:
        rc = gcd_info(ra)[0]
        print(f"  s{h:>3} {sp[h]:<28} root andness {ra:.3f} ({rc})  "
              f"disj-mass {dm:.2f}  n_disj {nd}")
    if not droot:
        print("  (none)")

    print("\n--- species by DISJUNCTIVE MASS (influence-weighted OR usage) ---")
    for h, ra, dm, nd, best in sorted(rows, key=lambda r: -r[2])[:args.top]:
        bw, ba, bc = best
        bstr = f"top OR: infl {bw:.2f} andness {ba:.3f} ({bc})" if bc else ""
        print(f"  s{h:>3} {sp[h]:<28} disj-mass {dm:.2f}  n_disj {nd}  "
              f"root a {ra:.2f}   {bstr}")


if __name__ == "__main__":
    main()
