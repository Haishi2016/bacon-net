r"""Interpret / verbalize / visualize a frozen CUB *recttree* OCBM.

The rectangular graded-logic head (:class:`VectorRectTreeHead`) is a stack of
constant-width layers whose tree shape EMERGES from learned convergence
penalties (single-root pointer, <=N-parent fan-out) and per-edge L0 hard-concrete
gates. After :meth:`harden`, every species is a small, discrete DAG over the K
concept leaves, read directly from the frozen buffers -- and because the run is
concept-SUPERVISED, leaf ``i`` is pinned to the ``i``-th named CUB attribute, so
each rule reads in human bird vocabulary (AND / OR via continuous andness, NOT
via the identity/negation gate).

This is the recttree twin of ``interpret_fulltree_cub.py`` -- it reuses that
module's andness table, symbolic evaluator, text/HTML renderers and species
parser, and only swaps in a recttree-specific model loader and frozen-DAG
extractor.

Deliverables (flags), for one/many/ALL species (``--species 0,10,50`` or ``all``):
  --rules        symbolic rule extraction  (nested AND/OR/NOT over named attrs)
  --viz          visualization             (self-contained collapsible HTML DAG)
  --verbalize    LLM prompt / JSON schema  (paste-ready plain-language report)
  --audit        faithfulness audit        (symbolic reading reproduces the head)
  --grounding    concept-grounding check   (per-concept ROC-AUC vs CUB attribute)
  --examples     per-example explanations  (which literals fired, text only)
  --samples      sample images + metrics   (PNG: real test birds + rule + metrics)
  --export-all   dump every species' rule to one text file
  --all          rules + viz + audit + grounding + examples

    py -3 interpret_recttree_cub.py --ckpt saved/<recttree>.pt --species Albatross --rules --viz
    py -3 interpret_recttree_cub.py --ckpt saved/<recttree>.pt --species all --export-all
    py -3 interpret_recttree_cub.py --ckpt saved/<recttree>.pt --species 0,10,50 --samples --n 4
"""

from __future__ import annotations

import argparse
import html
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "table"))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

import _cub                                                     # noqa: E402
from cub_emergent import CUBEmergent, collect                  # noqa: E402
from eval_shapes import roc_auc                                 # noqa: E402

# Reuse the fulltree interpreter's pure helpers (andness table, text/HTML
# renderers, species parser) -- identical node schema. The recttree head applies
# a per-node *presence* attenuation toward 0.5 (soft-OR of the coefficient-scaled
# edges), so we use a local presence-aware evaluator rather than fulltree's.
from interpret_fulltree_cub import (                            # noqa: E402
    gcd_info, andness_label, tree_stats,
    format_tree_dedup, leaf_summary, _viz_node_html,
    load_species_names, parse_species)
from figure_bird_rules import build_display, layout, draw_tree  # noqa: E402
from bacon.aggregators.lsp.full_weight import lsp_power_mean    # noqa: E402

_PRES_CLAMP = 1.0 - 1e-6


def symbolic_eval(node, c_vec):
    """Evaluate an extracted recttree node on a concept vector, matching the
    head EXACTLY: leaves apply identity/negation, ops take the LSP andness power
    mean of their children and then attenuate toward 0.5 by the node's presence
    (``V = presence * pm + (1 - presence) * 0.5``), reproducing the frozen head."""
    if node["kind"] == "leaf":
        v = float(c_vec[node["concept"]])
        return (1.0 - v) if node["neg"] else v
    if not node["children"]:
        return 0.5                                             # absent node -> neutral
    vals, ws = [], []
    for w, ch in node["children"]:
        vals.append(symbolic_eval(ch, c_vec))
        ws.append(w)
    X = torch.tensor(vals, dtype=torch.float32).view(-1, 1)
    W = torch.tensor(ws, dtype=torch.float32).view(-1, 1)
    W = W / W.sum().clamp_min(1e-8)
    pm = float(lsp_power_mean(X, torch.tensor(float(node["andness"])), W,
                              eps=1e-6).view(-1)[0])
    pres = float(node.get("presence", 1.0))
    return pres * pm + (1.0 - pres) * 0.5


# --------------------------------------------------------------------------- io
def load_concept_names(K: int, attr312: bool):
    """Return list[str] of length ``K`` naming each concept leaf.

    Supervised runs pin concept ``i`` to attribute ``i``; use the raw 312 CUB
    attribute names for a 312-attr run, else the canonical 112 concept names.
    Falls back to ``concept{i}`` if a lookup is short.
    """
    if attr312:
        names = list(_cub.load_names_312())
    else:
        names, *_ = _cub._load_attr_groups()
        names = list(names)
    if len(names) < K:
        names = names + [f"concept{i}" for i in range(len(names), K)]
    return names[:K]


def load_model(ckpt_path, device):
    """Rebuild a recttree :class:`CUBEmergent` from a checkpoint and load weights.

    Head geometry (width/depth/layer_widths/max_parents/leaf_shortcut/negation/
    coefficients) is read from the saved metadata where present, and otherwise
    recovered from the frozen buffer SHAPES so reconstruction always matches the
    stored tensors.
    """
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    K = int(ck.get("K", 312))
    attr312 = bool(ck.get("attr312", K == 312))
    backbone = ck.get("backbone", "inception_v3")
    negation = any(k.endswith("head.transform_logits") for k in sd)
    coeff = any(".coeff_logits." in k for k in sd)

    # frozen buffer shapes are the source of truth for geometry.
    n_layers = 0
    while f"head.frozen_edge_{n_layers}" in sd:
        n_layers += 1
    depth = int(ck.get("rect_depth", n_layers)) if n_layers == 0 else n_layers
    widths = [K]
    src_dims = []
    for l in range(depth):
        e = sd[f"head.frozen_edge_{l}"]                        # (H, src, w_out)
        src_dims.append(int(e.shape[1]))
        widths.append(int(e.shape[2]))
    # leaf_shortcut iff an upper layer's source pool exceeds its prev-layer width.
    leaf_shortcut = any(src_dims[l] > widths[l] for l in range(1, depth)) \
        if depth > 1 else bool(ck.get("rect_leaf_shortcut", False))
    layer_widths = widths[1:]                                   # explicit schedule
    max_parents = int(ck.get("rect_max_parents", 2))

    model = CUBEmergent(
        K, n_species=200, head="recttree", backbone_kind=backbone,
        rect_layer_widths=layer_widths, rect_max_parents=max_parents,
        rect_leaf_shortcut=leaf_shortcut,
        fulltree_negation=negation, fulltree_coefficients=coeff)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [m for m in missing if "num_batches_tracked" not in m]
    if missing or unexpected:
        print(f"  [load] missing={missing[:4]} unexpected={unexpected[:4]}", flush=True)
    model.to(device).eval()
    return model, K, attr312, backbone, negation, coeff


# ---------------------------------------------------------- tree extraction
@torch.no_grad()
def extract_tree(head, h: int):
    """Build the frozen DAG for head ``h`` as a nested node dict.

    node = {"kind": "leaf", "concept": k, "neg": bool}
         | {"kind": "op", "andness": float, "children": [(weight, node), ...]}

    Mirrors the head's frozen inference exactly: child weights are the frozen
    edges (0/1) times the per-source coefficient relevance, normalized per
    destination over its live sources; the root is the argmax of the frozen
    pointer over the top layer. With ``leaf_shortcut`` the source pool of an
    upper layer is [prev-layer nodes] ++ [K concept leaves].
    """
    depth = head.depth
    widths = head.widths                                       # [K, w1, w2, ...]
    K = head.input_size
    leaf_shortcut = bool(head.leaf_shortcut)
    neg = None
    if head.use_negation and bool(head.transform_frozen):
        neg = head.frozen_transform[h]                        # (K,) 1=identity 0=NOT

    def leaf(k):
        is_neg = bool(head.use_negation and neg is not None and neg[k].item() < 0.5)
        return {"kind": "leaf", "concept": int(k), "neg": is_neg}

    node_layers = [[leaf(k) for k in range(widths[0])]]        # layer 0 = leaves
    for l in range(depth):
        w_out = widths[l + 1]
        src_dim = head.src_dims[l]
        E = getattr(head, f"frozen_edge_{l}")[h]              # (src_dim, w_out) {0,1}
        if head.use_coefficients:
            rel = torch.exp(head.coeff_logits[l][h].clamp(-10.0, 10.0))
        else:
            rel = torch.ones(src_dim)
        Ew = E * rel.unsqueeze(1)                             # (src_dim, w_out)
        a_bias = head.andness_bias[l][h]                      # (w_out,)
        andness = (torch.sigmoid(a_bias) * 3.0 - 1.0
                   if head.normalize_andness else a_bias)
        prev = node_layers[l]                                 # widths[l] nodes
        src_nodes = (prev + node_layers[0]) if (leaf_shortcut and l >= 1) else prev
        layer_nodes = []
        for d in range(w_out):
            col = Ew[:, d]
            tot = float(col.sum())
            if tot < 1e-9:                                    # absent node -> neutral
                layer_nodes.append({"kind": "op", "andness": float(andness[d]),
                                    "children": [], "presence": 0.0, "dead": True})
                continue
            srcs = [i for i in range(src_dim) if float(col[i]) > 0]
            ws = [float(col[i] / tot) for i in srcs]
            children = [(ws[j], src_nodes[srcs[j]]) for j in range(len(srcs))]
            # presence = soft-OR of the coefficient-scaled edges (clamped), the
            # head's node-activity attenuation toward 0.5 (matches _presence).
            clamped = col.clamp(0.0, _PRES_CLAMP)
            presence = float(1.0 - torch.prod(1.0 - clamped))
            layer_nodes.append({"kind": "op", "andness": float(andness[d]),
                                "children": children, "presence": presence})
        node_layers.append(layer_nodes)
    root_idx = int(head.frozen_root[h].argmax().item())
    return node_layers[depth][root_idx]


# ------------------------------------------------ full expression / verbalize
def _readable_name(nm: str, neg: bool) -> str:
    """'has_wing_color::black' -> 'wing color: black' (NOT-prefixed if negated)."""
    core = nm.replace("has_", "").replace("::", ": ").replace("_", " ")
    return ("NOT " + core) if neg else core


def full_expression(node, names, _seen=None, _counter=None, top=True):
    """Complete, UNPRUNED inline expression of the rule as nested GCD operators.

    Every operator is written as ``CODE[andness](w1*child1, w2*child2, ...)`` with
    all children and full depth. Shared subtrees (fan-out > 1 reuses a node) are
    printed once as ``CODE...=[N#]`` and referenced later as ``@N#`` so the string
    reflects the true DAG rather than exploding the reused paths.
    """
    if _seen is None:
        _seen, _counter = {}, [0]
    if node["kind"] == "leaf":
        return _readable_name(names[node["concept"]], node["neg"])
    nid = id(node)
    if nid in _seen:
        return "@" + _seen[nid]
    _counter[0] += 1
    label = f"N{_counter[0]}"
    _seen[nid] = label
    code, _, _ = gcd_info(node["andness"])
    kids = sorted(node["children"], key=lambda t: -t[0])
    parts = [f"{w:.2f}*{full_expression(ch, names, _seen, _counter, top=False)}"
             for w, ch in kids]
    tag = f"[{label} a={node['andness']:+.2f}]"
    return f"{code}{tag}(" + ", ".join(parts) + ")"


def verbalize_english(node, names, indent=0, weight=None):
    """Deterministic plain-language reading of the GL rule (no LLM needed).

    Each operator becomes an English clause driven by its GCD reading -- e.g.
    conjunctions become "must have all of", soft conjunctions "should have most
    of", disjunctions "enough to have any of" -- with child relevances as
    percentages. Fully recursive, so it mirrors the whole tree.
    """
    pad = "  " * indent
    wtxt = "" if weight is None else f"({weight * 100:.0f}%) "
    if node["kind"] == "leaf":
        return f"{pad}- {wtxt}{_readable_name(names[node['concept']], node['neg'])}"
    if not node["children"]:
        return f"{pad}- {wtxt}(no evidence)"
    code, name, reading = gcd_info(node["andness"])
    kids = sorted(node["children"], key=lambda t: -t[0])
    lines = [f"{pad}- {wtxt}{reading} ({name}, {len(kids)} of):"]
    for w, ch in kids:
        lines.append(verbalize_english(ch, names, indent + 1, w))
    return "\n".join(lines)


def do_expression(model, heads, names, species_names):
    head = model.head
    print("\n" + "=" * 78)
    print("FULL EXPRESSION  (complete unpruned GCD expression + English reading)")
    print("=" * 78)
    for h in heads:
        node = extract_tree(head, h)
        n_ops, n_leaves, n_paths = tree_stats(node)
        print(f"\n### species {h}: {species_names[h]}   "
              f"({n_ops} ops, {n_leaves} distinct literals)")
        print("\n  full expression:")
        print("    " + full_expression(node, names))
        print("\n  verbalization (plain language):")
        print(verbalize_english(node, names, indent=2))


# ----------------------------------------------------------------- subcommands
def do_rules(model, heads, names, species_names, max_children, min_weight):
    head = model.head
    print("\n" + "=" * 78)
    print("SYMBOLIC RULE EXTRACTION  (nested AND/OR/NOT over named CUB attributes)")
    print("=" * 78)
    for h in heads:
        node = extract_tree(head, h)
        n_ops, n_leaves, n_paths = tree_stats(node)
        print(f"\n### species {h}: {species_names[h]}   "
              f"({n_ops} ops, {n_leaves} distinct literals, {n_paths} expanded paths)")
        print(format_tree_dedup(node, names, max_children=max_children,
                                min_weight=min_weight), end="")
        print("  top literals (path-weighted):")
        for lit, w in leaf_summary(node, names)[:12]:
            print(f"    {w:6.3f}  {lit}")


def do_export_all(model, names, species_names, out_path, max_children, min_weight):
    head = model.head
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    sizes = []
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("CUB recttree OCBM -- per-species graded-logic rules.\n")
        f.write("Learned-convergence rectangular DAG, hardened to a discrete tree.\n")
        f.write("Shared subtrees are labelled [N#] and referenced as -> N#.\n")
        for h in range(head.num_heads):
            node = extract_tree(head, h)
            n_ops, n_leaves, n_paths = tree_stats(node)
            sizes.append(n_leaves)
            f.write("\n" + "=" * 78 + "\n")
            f.write(f"### species {h}: {species_names[h]}   "
                    f"({n_ops} ops, {n_leaves} distinct literals, {n_paths} paths)\n")
            f.write(format_tree_dedup(node, names, max_children=max_children,
                                      min_weight=min_weight))
            f.write("  top literals (path-weighted):\n")
            for lit, w in leaf_summary(node, names)[:15]:
                f.write(f"    {w:6.3f}  {lit}\n")
    st = torch.tensor(sizes, dtype=torch.float32)
    print(f"  wrote all {head.num_heads} species rules -> {out_path}")
    print(f"  rule size (distinct literals/class): median {st.median():.1f} "
          f"mean {st.mean():.1f} min {int(st.min())} max {int(st.max())}")


@torch.no_grad()
def do_audit(model, heads, loader, device):
    head = model.head
    print("\n" + "=" * 78)
    print("FAITHFULNESS AUDIT")
    print("=" * 78)
    C, _ = collect(model, loader, device)
    Cb = C[:256]
    max_abs = 0.0
    for h in heads:
        node = extract_tree(head, h)
        head_out = head(Cb.to(device))[:, h].cpu()
        sym = torch.tensor([symbolic_eval(node, Cb[i]) for i in range(Cb.size(0))])
        d = (sym - head_out).abs().max().item()
        max_abs = max(max_abs, d)
        print(f"  species {h:3d}: max|symbolic - head| = {d:.2e}")
    print(f"  -> worst extraction error across shown species: {max_abs:.2e} "
          f"({'LOSSLESS' if max_abs < 1e-4 else 'CHECK'})")
    correct = total = 0
    for img, c, y in loader:
        img, y = img.to(device), y.to(device)
        correct += (model(img)[0].argmax(1) == y).sum().item()
        total += y.numel()
    print(f"  frozen test accuracy: {100.0 * correct / total:.2f}%  (N={total})")


@torch.no_grad()
def do_grounding(model, loader, device, names, topn=15):
    print("\n" + "=" * 78)
    print("CONCEPT-GROUNDING CHECK  (per-concept AUC vs its NAMED CUB attribute)")
    print("=" * 78)
    C, A = collect(model, loader, device)                      # (N,K), (N,K)
    K = min(C.shape[1], A.shape[1])
    aucs = []
    for i in range(K):
        col = A[:, i].long()
        if 0 < int(col.sum()) < len(col):
            aucs.append((roc_auc(C[:, i], col), i))
        else:
            aucs.append((float("nan"), i))
    valid = [a for a, _ in aucs if a == a]
    mean_auc = sum(valid) / max(len(valid), 1)
    print(f"  mean self-attribute AUC over {len(valid)} concepts: {mean_auc:.3f}")
    ordered = sorted([t for t in aucs if t[0] == t[0]], key=lambda t: -t[0])
    print(f"\n  best-grounded {topn} concepts:")
    for au, i in ordered[:topn]:
        print(f"    AUC {au:.3f}  c{i:3d}  {names[i]}")
    print(f"\n  weakest-grounded {topn} concepts:")
    for au, i in ordered[-topn:]:
        print(f"    AUC {au:.3f}  c{i:3d}  {names[i]}")


@torch.no_grad()
def do_examples(model, heads, loader, device, names, species_names, n_examples=3):
    print("\n" + "=" * 78)
    print("PER-EXAMPLE EXPLANATIONS  (literals that fired for real test birds)")
    print("=" * 78)
    head = model.head
    imgs_seen = {h: 0 for h in heads}
    hset = set(heads)
    for img, c, y in loader:
        img = img.to(device)
        logits, C, _ = model(img)
        pred = logits.argmax(1).cpu()
        C = C.cpu()
        for b in range(img.size(0)):
            yl = int(y[b])
            if yl in hset and imgs_seen[yl] < n_examples and int(pred[b]) == yl:
                imgs_seen[yl] += 1
                node = extract_tree(head, yl)
                truth = symbolic_eval(node, C[b])
                print(f"\n  {species_names[yl]} (species {yl}) "
                      f"pred={'OK' if int(pred[b]) == yl else 'X'} "
                      f"root-truth={truth:.3f}")
                cv = C[b]
                shown = 0
                for lit, w in leaf_summary(node, names):
                    neg = lit.startswith("NOT ")
                    nm = lit[4:] if neg else lit
                    k = names.index(nm)
                    fired = (1.0 - float(cv[k])) if neg else float(cv[k])
                    mark = "+" if fired >= 0.5 else "-"
                    print(f"      {mark} truth={fired:.2f} w={w:.3f}  {lit}")
                    shown += 1
                    if shown >= 10:
                        break
        if all(imgs_seen[h] >= n_examples for h in heads):
            break


# ------------------------------------------------- aggregation coherence probe
def _family(name: str) -> str:
    """Attribute family = the part before '::' (mutually-exclusive value set),
    e.g. 'has_throat_color::olive' -> 'has_throat_color'."""
    return name.split("::", 1)[0]


def _direct_leaf_children(node, seen):
    """Yield (op_node, [(concept_idx, neg), ...]) for every op with >=2 leaf
    direct children (the concepts literally combined by ONE operator). Dedups
    shared DAG nodes within a head via ``seen`` (ids)."""
    if node["kind"] != "op" or id(node) in seen:
        return
    seen.add(id(node))
    leaf_kids = [(ch["concept"], ch["neg"]) for _, ch in node["children"]
                 if ch["kind"] == "leaf"]
    if len(leaf_kids) >= 2:
        yield node, leaf_kids
    for _, ch in node["children"]:
        yield from _direct_leaf_children(ch, seen)


@torch.no_grad()
def do_coherence(model, heads, loader, device, names, seed=0):
    """Evidence probe: are the concepts an operator AGGREGATES arbitrary, or do
    they carry logical/semantic relations?

    For every operator that directly combines >=2 concept literals, we measure
    (over the test set) the mean pairwise CORRELATION of those sibling literals
    and the fraction that belong to the SAME attribute family (same '::'-prefix,
    i.e. mutually-exclusive values of one attribute). We split by the operator's
    andness. Hypothesis: conjunctions (AND) group co-present CROSS-family
    features (positive corr); disjunctions (OR) group mutually-exclusive
    SAME-family values (negative corr, high same-family) -- both logically
    coherent, not arbitrary. A random-pair baseline calibrates 'arbitrary'.
    """
    import numpy as np
    print("\n" + "=" * 78)
    print("FEATURE-AGGREGATION COHERENCE  (are co-aggregated concepts related?)")
    print("=" * 78)
    C, _ = collect(model, loader, device)                      # (N, K)
    Z = (C - C.mean(0)) / C.std(0).clamp_min(1e-6)             # standardized preds
    # ground-truth semantic co-occurrence: correlation of the two attributes over
    # the 200-class attribute matrix (clean, no noisy predicted concepts).
    M = _cub.load_class_attr_312()                             # (200, 312) {0,1}
    Zg = (M - M.mean(0)) / M.std(0).clamp_min(1e-6)

    def lit_corr(Zm, i, si, j, sj):
        return float((Zm[:, i] * Zm[:, j]).mean()) * si * sj

    buckets = {"AND (a>0.6)": [], "MID (0.4-0.6)": [], "OR (a<0.4)": []}
    head = model.head
    all_used = set()
    for h in heads:
        node = extract_tree(head, h)
        for op, kids in _direct_leaf_children(node, set()):
            a = op["andness"]
            key = ("AND (a>0.6)" if a > 0.6 else
                   "OR (a<0.4)" if a < 0.4 else "MID (0.4-0.6)")
            pc, gc, same = [], [], []
            for x in range(len(kids)):
                for y in range(x + 1, len(kids)):
                    (i, ni), (j, nj) = kids[x], kids[y]
                    si, sj = (-1 if ni else 1), (-1 if nj else 1)
                    all_used.add(i); all_used.add(j)
                    pc.append(lit_corr(Z, i, si, j, sj))
                    gc.append(lit_corr(Zg, i, si, j, sj))
                    same.append(1.0 if _family(names[i]) == _family(names[j]) else 0.0)
            if pc:
                buckets[key].append((float(np.mean(pc)), float(np.mean(gc)),
                                     float(np.mean(same)), len(kids)))

    # random-pair baseline over the concepts actually used by these rules.
    rng = np.random.default_rng(seed)
    used = sorted(all_used)
    rp, rg, rs = [], [], []
    if len(used) >= 2:
        for _ in range(5000):
            i, j = rng.choice(used, size=2, replace=False)
            rp.append(lit_corr(Z, int(i), 1, int(j), 1))
            rg.append(lit_corr(Zg, int(i), 1, int(j), 1))
            rs.append(1.0 if _family(names[int(i)]) == _family(names[int(j)]) else 0.0)
    print(f"  concepts covered: {len(used)}   (over {len(heads)} species' rules)")
    print(f"\n  {'operator group':<16}{'#nodes':>7}{'pred-corr':>11}{'GT-corr':>10}"
          f"{'same-fam%':>11}{'fan-in':>8}")
    for key, items in buckets.items():
        if not items:
            print(f"  {key:<16}{0:>7}{'--':>11}{'--':>10}{'--':>11}{'--':>8}")
            continue
        mp = sum(t[0] for t in items) / len(items)
        mg = sum(t[1] for t in items) / len(items)
        ms = sum(t[2] for t in items) / len(items) * 100
        mf = sum(t[3] for t in items) / len(items)
        print(f"  {key:<16}{len(items):>7}{mp:>11.3f}{mg:>10.3f}{ms:>11.1f}{mf:>8.2f}")
    if rp:
        print(f"  {'RANDOM baseline':<16}{'':>7}{float(np.mean(rp)):>11.3f}"
              f"{float(np.mean(rg)):>10.3f}{float(np.mean(rs)) * 100:>11.1f}{'':>8}")
    print("\n  reading: pred-corr = co-activation of predicted concepts; GT-corr = "
          "\n  co-occurrence of the named attributes across the 200 species (clean "
          "\n  semantic signal). corr >> random for AND (co-present) and high "
          "\n  same-family%% for OR (mutually-exclusive values) => logically coherent.")


# ------------------------------------------------------------- visualization
def do_viz(model, heads, names, species_names, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    head = model.head
    for h in heads:
        node = extract_tree(head, h)
        body = _viz_node_html(node, names)
        title = html.escape(f"{species_names[h]} (species {h})")
        page = f"""<!doctype html><html><head><meta charset="utf-8">
<title>{title} rule</title><style>
body{{font:13px/1.5 system-ui,sans-serif;margin:24px;color:#222}}
h1{{font-size:16px}}
ul{{list-style:none;margin:0 0 0 14px;padding:0;border-left:1px dotted #bbb}}
li{{margin:2px 0;padding-left:8px}}
summary.op{{cursor:pointer;font-weight:600;color:#1560b0}}
.op.ref{{color:#8a6d00;font-weight:600}}
.shared{{margin-left:8px}}
.lit{{color:#0a7d3c}} .lit.neg{{color:#c0392b}}
.w{{display:inline-block;min-width:34px;color:#888;font-variant-numeric:tabular-nums}}
details{{margin-left:8px}}
</style></head><body>
<h1>{title} &mdash; graded-logic rule (frozen recttree, faithful)</h1>
<p>Blue = graded-logic aggregator, labelled with its GCD degree (andness a:
conjunctive a&gt;=1, A neutral a=0.5, disjunctive a&lt;=0); green = attribute
literal; red = negated literal. Weights are normalized child relevances used in
the power mean. Each shared subtree is expanded once as <b>[N#]</b>; later reuses
show as <b>&#8618; N# (shared)</b>.</p>
{body}
</body></html>"""
        out = os.path.join(out_dir, f"rule_species{h:03d}.html")
        with open(out, "w", encoding="utf-8") as f:
            f.write(page)
        print(f"  wrote {out}")


# ------------------------------------------------------------- sample images
@torch.no_grad()
def do_samples(model, heads, names, species_names, device, out_dir, attr312,
               image_size, n=4, correct_only=True, tree_children=3, tree_depth=3,
               tree_min_weight=0.05):
    """Render, per species, a row of real CUB test photos with per-image metrics
    (predicted probability, root truth) ABOVE a drawing of the species' faithful
    graded-logic rule TREE (GCD-andness nodes, weighted edges, signed literals)
    -- so the result shows the actual GL structure, not just feature importance."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    os.makedirs(out_dir, exist_ok=True)
    head = model.head
    ds = _cub._CUBImages("test", False, attr312=attr312, image_size=image_size)
    by_cls = {}
    for idx, e in enumerate(ds.entries):
        by_cls.setdefault(int(e["class_label"]), []).append(idx)

    for h in heads:
        node = extract_tree(head, h)
        # compact, legible display tree of the faithful GL rule.
        disp = build_display(node, names, tree_children, tree_depth, tree_min_weight)
        n_leaves_disp, max_depth_disp = layout(disp)

        idxs = by_cls.get(h, [])
        picks, n_correct, n_total = [], 0, 0
        for idx in idxs:
            x, c, y = ds[idx]
            logits, C, _ = model(x.unsqueeze(0).to(device))
            pred = int(logits.argmax(1))
            prob = float(torch.softmax(logits, dim=1)[0, h])
            root = symbolic_eval(node, C[0].cpu())
            n_total += 1
            ok = (pred == h)
            n_correct += int(ok)
            picks.append((idx, ok, prob, root))
        recall = 100.0 * n_correct / max(n_total, 1)
        chosen = [p for p in picks if p[1]] if correct_only else picks
        if not chosen:
            chosen = picks
        chosen = sorted(chosen, key=lambda p: -p[2])[:n]

        cols = max(len(chosen), 1)
        n_ops, n_leaves, n_paths = tree_stats(node)
        # top row = photos, bottom = one wide axis for the GL tree.
        fig = plt.figure(figsize=(max(3.4 * cols, 7.0), 8.2))
        gs = fig.add_gridspec(2, cols, height_ratios=[1.0, 1.7], hspace=0.12)
        for j, (idx, ok, prob, root) in enumerate(chosen):
            ax = fig.add_subplot(gs[0, j])
            e = ds.entries[idx]
            img = Image.open(_cub._local_path(e["img_path"])).convert("RGB")
            ax.imshow(img)
            ax.set_title(f"{'OK' if ok else 'MISS'}  p={prob:.2f}\nroot={root:.2f}",
                         fontsize=10, color=("#0a7d3c" if ok else "#c0392b"))
            ax.axis("off")
        ax_tree = fig.add_subplot(gs[1, :])
        draw_tree(ax_tree, disp, n_leaves_disp, max_depth_disp)

        caption = (f"{species_names[h]} (species {h})   test recall {recall:.1f}%   "
                   f"|   faithful GL rule: {n_leaves} literals, {n_ops} ops   "
                   f"(shown pruned to top {tree_children}/node, depth {tree_depth})")
        fig.suptitle(caption, fontsize=11, y=0.98)
        fig.text(0.5, 0.008,
                 f"Drawing is the reasoning DAG unrolled into a tree for readability: "
                 f"a leaf repeated across branches (e.g. a shared sub-condition) is a "
                 f"SINGLE node feeding \u2264{head.max_parents} parents, not a duplicate.",
                 ha="center", fontsize=8, style="italic", color="#555555")
        fig.tight_layout(rect=(0, 0.02, 1, 0.96))
        out = os.path.join(out_dir, f"samples_species{h:03d}.png")
        fig.savefig(out, dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"  wrote {out}   (recall {recall:.1f}%, {n_leaves} literals)")


# ------------------------------------------------------------- verbalize (LLM)
def do_verbalize(model, heads, names, species_names, out_dir, max_children,
                 max_depth, min_weight):
    """Write a paste-ready LLM prompt per species (delegates to the shared
    verbalizer if present; otherwise emits a compact JSON rule + instructions)."""
    import json
    os.makedirs(out_dir, exist_ok=True)
    head = model.head

    def _readable(nm, neg):
        core = nm.replace("has_", "").replace("::", ": ").replace("_", " ")
        return ("NOT " + core) if neg else core

    def to_json(node, depth=0):
        if node["kind"] == "leaf":
            return {"feature": _readable(names[node["concept"]], node["neg"])}
        code, name, verb = gcd_info(node["andness"])
        kids = sorted(node["children"], key=lambda t: -t[0])
        kids = [k for k in kids if k[0] >= min_weight]
        if max_children:
            kids = kids[:max_children]
        if depth >= max_depth:
            children = [{"feature": _readable(*_split(lit)), "weight": round(w, 3)}
                        for lit, w in leaf_summary(node, names)[:max_children]]
        else:
            children = [{"weight": round(w, 3), **to_json(ch, depth + 1)}
                        for w, ch in kids]
        return {"operator": code, "operator_name": name, "reading": verb,
                "children": children}

    def _split(lit):
        neg = lit.startswith("NOT ")
        return (lit[4:] if neg else lit, neg)

    for h in heads:
        node = extract_tree(head, h)
        rule = to_json(node)
        prompt = (
            f"You are given a graded-logic (LSP) rule a CUB-200 bird classifier "
            f"learned for the species '{species_names[h]}'. Each operator has a "
            f"GCD andness code (conjunctive C/HC/SC = 'must have', disjunctive "
            f"D/HD/SD = 'enough to have any', A = neutral). Weights are child "
            f"relevances. Write a concise, plain-language field-guide description "
            f"of what this rule looks for.\n\nRULE (JSON):\n"
            f"{json.dumps(rule, indent=2)}\n")
        out = os.path.join(out_dir, f"verbalize_species{h:03d}.txt")
        with open(out, "w", encoding="utf-8") as f:
            f.write(prompt)
        # also print a deterministic English reading straight to the console so
        # the rule is legible without an LLM (the .txt is the LLM prompt).
        print(f"\n### species {h}: {species_names[h]}")
        print(verbalize_english(node, names, indent=1))
        print(f"  (LLM prompt -> {out})")


# ------------------------------------------------------------------------ main
def resolve_species(spec, species_names, num_heads):
    if spec is not None and spec.strip().lower() == "all":
        return list(range(num_heads))
    return parse_species(spec, species_names)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="recttree checkpoint (.pt)")
    ap.add_argument("--species", default=None,
                    help="comma list of class indices / name substrings, or 'all' "
                         "(default 0,10,50)")
    ap.add_argument("--rules", action="store_true")
    ap.add_argument("--expression", action="store_true",
                    help="print the full unpruned GCD expression + English reading")
    ap.add_argument("--viz", action="store_true")
    ap.add_argument("--verbalize", action="store_true")
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--grounding", action="store_true")
    ap.add_argument("--coherence", action="store_true",
                    help="probe whether co-aggregated concepts are logically related")
    ap.add_argument("--examples", action="store_true")
    ap.add_argument("--samples", action="store_true",
                    help="render real test photos + per-image metrics + rule (PNG)")
    ap.add_argument("--export-all", action="store_true")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--n", type=int, default=4, help="sample images per species")
    ap.add_argument("--max-children", type=int, default=6)
    ap.add_argument("--max-depth", type=int, default=3, help="verbalize prune depth")
    ap.add_argument("--min-weight", type=float, default=0.0)
    ap.add_argument("--tree-children", type=int, default=3,
                    help="max children per node in the drawn GL tree (--samples)")
    ap.add_argument("--tree-depth", type=int, default=3,
                    help="max depth of the drawn GL tree (--samples)")
    ap.add_argument("--tree-min-weight", type=float, default=0.05,
                    help="drop child edges below this weight in the drawn tree")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--out-dir", default=os.path.join(_HERE, "results", "recttree_rules"))
    args = ap.parse_args()

    if args.all:
        args.rules = args.viz = args.audit = args.grounding = args.examples = True
    if not any([args.rules, args.viz, args.verbalize, args.audit, args.grounding,
                args.examples, args.samples, args.export_all, args.expression,
                args.coherence]):
        args.rules = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(args.ckpt):
        raise SystemExit(f"checkpoint not found: {args.ckpt}")

    species_names = load_species_names()
    model, K, attr312, backbone, negation, coeff = load_model(args.ckpt, device)
    names = load_concept_names(K, attr312)
    heads = resolve_species(args.species, species_names, model.head.num_heads)
    print(f"loaded {os.path.basename(args.ckpt)}  K={K} backbone={backbone} "
          f"attr312={attr312} negation={negation} coeff={coeff} "
          f"widths={model.head.widths} shortcut={model.head.leaf_shortcut} "
          f"max_parents={model.head.max_parents}")

    image_size = 299 if backbone == "inception_v3" else 224
    mc = None if not args.max_children else args.max_children

    need_data = args.audit or args.grounding or args.examples or args.coherence
    loader = None
    if need_data:
        from torch.utils.data import DataLoader
        loader = DataLoader(
            _cub._CUBImages("test", False, attr312=attr312, image_size=image_size),
            batch_size=64, shuffle=False, num_workers=args.workers, pin_memory=True)

    if args.rules:
        do_rules(model, heads, names, species_names, mc, args.min_weight)
    if args.expression:
        do_expression(model, heads, names, species_names)
    if args.viz:
        do_viz(model, heads, names, species_names, args.out_dir)
    if args.verbalize:
        do_verbalize(model, heads, names, species_names, args.out_dir, mc,
                     args.max_depth, args.min_weight)
    if args.audit:
        do_audit(model, heads, loader, device)
    if args.grounding:
        do_grounding(model, loader, device, names)
    if args.coherence:
        do_coherence(model, heads, loader, device, names)
    if args.examples:
        do_examples(model, heads, loader, device, names, species_names)
    if args.samples:
        do_samples(model, heads, names, species_names, device, args.out_dir,
                   attr312, image_size, n=args.n, tree_children=args.tree_children,
                   tree_depth=args.tree_depth, tree_min_weight=args.tree_min_weight)
    if args.export_all:
        do_export_all(model, names, species_names,
                      os.path.join(args.out_dir, "all_species_rules.txt"),
                      mc, args.min_weight)


if __name__ == "__main__":
    main()
