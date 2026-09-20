r"""Aggregation-type usage per head width (CUB recttree).

Loads one or more frozen recttree checkpoints (CPU only, no forward pass so it is
safe to run alongside a live training job), extracts every class tree's active
DAG, and tallies how each aggregation node is used:

  * conjunctive  (andness a > 0.5, AND-leaning)  -- "must have"
  * disjunctive  (andness a < 0.5, OR-leaning)   -- "any of"
  * neutral      (|a - 0.5| < NEUTRAL, ~mean)

Nodes are counted once per unique node reachable from the root (DAG reuse is not
double counted). Checkpoints are grouped by hidden width parsed from the filename
(``_w<W>_``) and stats are averaged across the group's seeds.

Usage:
  py -3 analyze_aggregation.py --ckpts saved/A.pt saved/B.pt ...
"""
from __future__ import annotations

import argparse
import re
import statistics as st
from collections import Counter

import torch

from interpret_recttree_cub import load_model, extract_tree
from interpret_fulltree_cub import gcd_info

NEUTRAL = 0.05   # |a-0.5| band treated as ~arithmetic-mean (neither AND nor OR)


def walk(root):
    """Yield each unique op-node dict reachable from ``root`` (DAG-dedup), plus
    return leaf count. Returns (ops:list[dict], n_leaves:int)."""
    seen = set()
    ops = []
    leaves = 0
    stack = [root]
    seen_leaf = set()
    while stack:
        node = stack.pop()
        nid = id(node)
        if nid in seen:
            continue
        seen.add(nid)
        if node["kind"] == "leaf":
            # count unique leaf *instances* in the reachable structure
            if nid not in seen_leaf:
                seen_leaf.add(nid)
            continue
        ops.append(node)
        for _w, ch in node.get("children", []):
            stack.append(ch)
    # count leaves as unique leaf nodes reached
    # (re-walk cheaply: leaves are those with kind==leaf we skipped above)
    # recompute using a fresh traversal counting leaf ids
    seen2 = set()
    leaf_ids = set()
    stack = [root]
    while stack:
        node = stack.pop()
        nid = id(node)
        if nid in seen2:
            continue
        seen2.add(nid)
        if node["kind"] == "leaf":
            leaf_ids.add(nid)
            continue
        for _w, ch in node.get("children", []):
            stack.append(ch)
    return ops, len(leaf_ids)


def feature_importance(root):
    """Path-weight importance per concept index. Distributes a unit of weight
    from the root down the normalized child weights; a leaf accumulates the sum
    of path-weights over every route that reaches it (DAG-safe, weights conserve
    so the totals sum to 1). Gives each concept's structural share of the
    decision -- the basis for an *effective* feature count."""
    imp = {}

    def rec(node, w):
        if node["kind"] == "leaf":
            imp[node["concept"]] = imp.get(node["concept"], 0.0) + w
            return
        for cw, ch in node.get("children", []):
            rec(ch, w * cw)

    rec(root, 1.0)
    return imp


def eff_count(imp, cover):
    """Fewest concepts whose importance covers `cover` of the total weight."""
    tot = sum(imp.values())
    if tot <= 0:
        return 0
    acc = 0.0
    n = 0
    for v in sorted(imp.values(), reverse=True):
        acc += v
        n += 1
        if acc >= cover * tot:
            break
    return n


def eff_concepts(imp, cover):
    """The concept indices covering `cover` of total weight (for global union)."""
    tot = sum(imp.values())
    out = set()
    if tot <= 0:
        return out
    acc = 0.0
    for k, v in sorted(imp.items(), key=lambda t: -t[1]):
        acc += v
        out.add(k)
        if acc >= cover * tot:
            break
    return out


def analyze_ckpt(path):
    model, K, attr312, backbone, negation, coeff = load_model(path, torch.device("cpu"))
    head = model.head
    H = head.num_heads
    all_and = []          # andness of every reachable op node (all trees)
    codes = Counter()
    per_tree_ops = []
    per_tree_leaves = []
    per_tree_conj = []
    per_tree_disj = []
    used_list = []        # distinct concepts per tree
    eff90_list = []       # concepts covering 90% of decision weight per tree
    eff50_list = []       # concepts covering 50% per tree
    global_used = set()   # union of concepts used across all trees (model vocab)
    global_eff90 = set()  # union of 90%-weight concepts across all trees
    for h in range(H):
        root = extract_tree(head, h)
        ops, nleaf = walk(root)
        per_tree_ops.append(len(ops))
        per_tree_leaves.append(nleaf)
        c = d = 0
        for nd in ops:
            a = float(nd["andness"])
            all_and.append(a)
            code, _, _ = gcd_info(a)
            codes[code] += 1
            if a > 0.5 + NEUTRAL:
                c += 1
            elif a < 0.5 - NEUTRAL:
                d += 1
        per_tree_conj.append(c)
        per_tree_disj.append(d)
        imp = feature_importance(root)
        used_list.append(sum(1 for v in imp.values() if v > 0))
        eff90_list.append(eff_count(imp, 0.9))
        eff50_list.append(eff_count(imp, 0.5))
        global_used.update(k for k, v in imp.items() if v > 0)
        global_eff90.update(eff_concepts(imp, 0.9))
    tot_ops = sum(per_tree_ops)
    conj = sum(per_tree_conj)
    disj = sum(per_tree_disj)
    neut = tot_ops - conj - disj
    return {
        "K": K, "H": H,
        "ops_total": tot_ops,
        "ops_per_tree": tot_ops / H,
        "leaves_per_tree": sum(per_tree_leaves) / H,
        "conj": conj, "disj": disj, "neutral": neut,
        "conj_frac": conj / max(1, tot_ops),
        "disj_frac": disj / max(1, tot_ops),
        "ratio_cd": (conj / disj) if disj else float("inf"),
        "mean_andness": st.mean(all_and) if all_and else float("nan"),
        "codes": codes,
        "used_list": used_list, "eff90_list": eff90_list, "eff50_list": eff50_list,
        "vocab_used": len(global_used), "vocab_eff90": len(global_eff90),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--hist", default=None,
                    help="save a rule-size histogram PNG (uses the FIRST checkpoint)")
    ap.add_argument("--csv", default=None,
                    help="write per-class rule sizes CSV (uses the FIRST checkpoint)")
    args = ap.parse_args()

    groups = {}   # width -> list[(path, stats)]
    ordered = []  # (path, stats) in load order
    first_stats = None
    for p in args.ckpts:
        m = re.search(r"_w(\d+)_", p)
        w = int(m.group(1)) if m else -1
        print(f"[load] W{w}  {p}", flush=True)
        stats = analyze_ckpt(p)
        if first_stats is None:
            first_stats = stats
        ordered.append((p, stats))
        groups.setdefault(w, []).append((p, stats))

    print("\n================ AGGREGATION USAGE BY HEAD WIDTH ================")
    hdr = (f"{'width':>6} {'seeds':>5} {'ops/tree':>9} {'leaves/tree':>11} "
           f"{'conj%':>7} {'disj%':>7} {'neut%':>7} {'conj:disj':>9} {'mean_a':>7}")
    print(hdr)
    print("-" * len(hdr))
    for w in sorted(groups, reverse=True):
        rows = [s for _p, s in groups[w]]
        n = len(rows)
        def avg(k):
            return sum(r[k] for r in rows) / n
        opt = avg("ops_per_tree")
        lpt = avg("leaves_per_tree")
        cf = avg("conj_frac") * 100
        df = avg("disj_frac") * 100
        nf = 100 - cf - df
        # ratio from aggregate counts (robust to per-seed inf)
        C = sum(r["conj"] for r in rows)
        D = sum(r["disj"] for r in rows)
        ratio = (C / D) if D else float("inf")
        ma = avg("mean_andness")
        print(f"{w:>6} {n:>5} {opt:>9.2f} {lpt:>11.2f} "
              f"{cf:>6.1f}% {df:>6.1f}% {nf:>6.1f}% {ratio:>9.2f} {ma:>7.3f}")

    # ---- feature usage / conciseness ----
    print("\n================ FEATURE USAGE (concepts per class tree) ================")
    K = next(iter(next(iter(groups.values()))))[1]["K"]
    print(f"(K = {K} available concepts)")
    fhdr = (f"{'width':>6} {'seeds':>5} {'concepts/tree':>14} {'median':>7} {'P90':>5} "
            f"{'eff90':>7} {'eff50':>7} {'model vocab':>12} {'vocab@90%':>10}")
    print(fhdr)
    print("-" * len(fhdr))
    for w in sorted(groups, reverse=True):
        rows = [s for _p, s in groups[w]]
        n = len(rows)
        used = [x for r in rows for x in r["used_list"]]
        e90 = [x for r in rows for x in r["eff90_list"]]
        e50 = [x for r in rows for x in r["eff50_list"]]

        def ms(v):
            return (st.mean(v), (st.pstdev(v) if len(v) > 1 else 0.0))
        um, us = ms(used)
        med = st.median(used)
        p90 = sorted(used)[int(0.9 * (len(used) - 1))]
        e9m = st.mean(e90)
        e5m = st.mean(e50)
        vocab = st.mean([r["vocab_used"] for r in rows])
        vocab90 = st.mean([r["vocab_eff90"] for r in rows])
        print(f"{w:>6} {n:>5} {um:>7.1f}±{us:<5.1f} {med:>7.0f} {p90:>5.0f} "
              f"{e9m:>7.1f} {e5m:>7.1f} {vocab:>6.0f}/{K:<5d} {vocab90:>5.0f}/{K:<4d}")
    print("  concepts/tree = distinct concepts reachable in the frozen rule "
          "(mean±std, median, 90th pct);")
    print("  eff90/eff50 = fewest concepts carrying 90%/50% of the decision weight (mean over trees);")
    print("  model vocab = distinct concepts used across ALL trees; "
          "vocab@90% = union of each tree's 90%-weight concepts.")

    # GCD code distribution per width (aggregated)
    print("\n---------------- GCD code distribution (share of op nodes) ----------------")
    for w in sorted(groups, reverse=True):
        agg = Counter()
        for _p, s in groups[w]:
            agg.update(s["codes"])
        tot = sum(agg.values())
        top = ", ".join(f"{k} {100*v/tot:.0f}%" for k, v in agg.most_common(8))
        print(f"  W{w}: {top}")

    if args.csv and first_stats is not None:
        write_csv(first_stats, args.csv)
    if args.hist:
        def _lbl(p):
            wm = re.search(r"_w(\d+)_", p)
            sm = re.search(r"_s(\d+)_", p)
            return f"W{wm.group(1) if wm else '?'}" + (f" s{sm.group(1)}" if sm else "")
        plot_hist([(_lbl(p), s) for p, s in ordered], args.hist)


def write_csv(stats, out):
    import csv
    try:
        from interpret_recttree_cub import load_species_names
        sp = load_species_names()
    except Exception:
        sp = [f"class_{i}" for i in range(stats["H"])]
    with open(out, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["species_idx", "species", "raw_concepts", "eff90", "eff50"])
        for h in range(stats["H"]):
            wr.writerow([h, sp[h] if h < len(sp) else h,
                         stats["used_list"][h], stats["eff90_list"][h],
                         stats["eff50_list"][h]])
    print(f"  wrote {out}  ({stats['H']} classes)")


def plot_hist(items, out):
    """items = list of (label, stats). Overlays each model's per-class rule-size
    distribution (raw reachable concepts) and the effective (50%-weight) count,
    so seed-to-seed variance in conciseness is visible rather than hidden."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B3", "#CCB974", "#64B5CD"]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    for i, (lab, s) in enumerate(items):
        col = colors[i % len(colors)]
        used = s["used_list"]
        e50 = s["eff50_list"]
        ax[0].hist(used, bins=40, range=(0, 312), histtype="step", lw=2,
                   color=col, label=f"{lab}  (med {st.median(used):.0f})")
        ax[1].hist(e50, bins=30, range=(0, 40), histtype="step", lw=2,
                   color=col, label=f"{lab}  (med {st.median(e50):.0f})")
    ax[0].set(title="Distinct concepts per class rule (raw, reachable)",
              xlabel="concepts in rule", ylabel="# classes")
    ax[0].legend(fontsize=8)
    ax[1].set(title="Core concepts (carry 50% of decision weight)",
              xlabel="concepts", ylabel="# classes")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
