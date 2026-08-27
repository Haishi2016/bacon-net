r"""Interpret a frozen CUB full-tree OCBM (b8+eg2+negation+supervised, 71.25%).

The head is a permutation-free, egress-hardened graded-logic tree: after freeze,
every species is a small DAG over the 112 concepts, and -- because this run was
SUPERVISED (concept-lam) -- concept ``i`` is pinned to the ``i``-th *named* CUB
attribute.  So each species rule reads in human bird vocabulary, with AND / OR
(continuous andness) and NOT (the identity/negation gate).

Five deliverables (flags):
  --rules        symbolic rule extraction  (nested AND/OR/NOT over named attrs)
  --audit        faithfulness audit        (our reading reproduces the head; frozen acc)
  --grounding    concept-grounding check   (per-concept ROC-AUC vs CUB attribute)
  --examples     per-example explanations  (which literals fired, truths up the tree)
  --viz          visualization             (self-contained collapsible HTML DAG)
  --all          all of the above

    py -3 interpret_fulltree_cub.py --all --species 0,10,50
"""

from __future__ import annotations

import argparse
import html
import math
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
from bacon.aggregators.lsp.full_weight import lsp_power_mean    # noqa: E402

DEFAULT_CKPT = os.path.join(
    _HERE, "saved", "cub_ocbm_k112_fulltree_b8_eg2_neg_sup1_coef_800ep.pt")


# --------------------------------------------------------------------------- io
def load_species_names():
    """Return list[str] of 200 CUB species names (index = class_label)."""
    path = os.path.join(_cub.CUB, "classes.txt")
    names = [None] * 200
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            p = line.split(None, 1)
            if len(p) == 2:
                cid = int(p[0]) - 1                             # 0-based class id
                nm = p[1].strip().split(".", 1)[-1]            # drop "001." prefix
                names[cid] = nm.replace("_", " ")
    return names


def load_model(ckpt_path, device):
    """Rebuild CUBEmergent from a full-tree checkpoint and load its weights.

    The final .pt only stores {state_dict, K, head, branching, decorr_lam}; the
    remaining head options (negation / coefficients) are detected from the saved
    parameter keys so reconstruction matches the tensor shapes exactly.
    """
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = ck["state_dict"]
    K = int(ck.get("K", 112))
    branching = int(ck.get("branching", 8))
    negation = any(k.endswith("head.transform_logits") for k in sd)
    coefficients = any(".coeff_logits." in k for k in sd)
    # max_egress affects only freeze logic (not param shapes); read the frozen
    # routing to recover the actual value (max parents any source feeds).
    max_egress = 1
    r0 = sd.get("head.frozen_route_0")
    if r0 is not None:
        max_egress = int(r0.sum(dim=2).max().item())           # per-source fan-out
    model = CUBEmergent(K, n_species=200, head="fulltree", branching=branching,
                        fulltree_coefficients=coefficients,
                        fulltree_max_egress=max(max_egress, 1),
                        fulltree_negation=negation)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    missing = [m for m in missing if "num_batches_tracked" not in m]
    if missing or unexpected:
        print(f"  [load] missing={missing[:4]} unexpected={unexpected[:4]}", flush=True)
    model.to(device).eval()
    return model, K, branching, negation, coefficients, max_egress


# ------------------------------------------------------------- andness labels
# Graded-logic (GCD) nomenclature: map an aggregator's andness a in [-1, 2] to
# its named GL degree (Table 3, "Andness-directed GCD").  Boundaries are the
# mid-points between adjacent 1/14 grades in the hard/soft core; the hyper
# (t-norm / t-conorm) regions outside [0,1] use the table's ranges.
def gcd_info(a: float):
    """Return (code, name, verbalization) for andness ``a`` per the GCD table."""
    a = float(a)
    if a >= 1.9375:   return ("CC",  "drastic conjunction",     "must all be completely satisfied")
    if a >= 1.3125:   return ("HHC", "high hyper-conjunction",  "below the lowest")
    if a >= 1.1875:   return ("CP",  "product t-norm",          "below the lowest")
    if a >= 1.03125:  return ("LHC", "low hyper-conjunction",   "below the lowest")
    if a >= 0.96875:  return ("C",   "pure conjunction",        "decided by lowest")
    if a >= 0.892857: return ("HC+", "high hard conjunction",   "must have all")
    if a >= 0.821429: return ("HC",  "medium hard conjunction", "must have all")
    if a >= 0.75:     return ("HC-", "low hard conjunction",    "must have all")
    if a >= 0.678571: return ("SC+", "high soft conjunction",   "nice to have most")
    if a >= 0.607143: return ("SC",  "medium soft conjunction", "nice to have most")
    if a >= 0.535714: return ("SC-", "low soft conjunction",    "nice to have most")
    if a >= 0.464286: return ("A",   "arithmetic mean",         "nice to have")
    if a >= 0.392857: return ("SD-", "low soft disjunction",    "nice to have some")
    if a >= 0.321429: return ("SD",  "medium soft disjunction", "nice to have some")
    if a >= 0.25:     return ("SD+", "high soft disjunction",   "nice to have some")
    if a >= 0.178571: return ("HD-", "low hard disjunction",    "enough to have any")
    if a >= 0.107143: return ("HD",  "medium hard disjunction", "enough to have any")
    if a >= 0.03125:  return ("HD+", "high hard disjunction",   "enough to have any")
    if a >= -0.03125: return ("D",   "pure disjunction",        "decided by highest")
    if a >= -0.1875:  return ("LHD", "low hyper-disjunction",   "above the highest")
    if a >= -0.3125:  return ("DP",  "product t-conorm",        "above the highest")
    if a >= -0.9375:  return ("HHD", "high hyper-disjunction",  "above the highest")
    return                   ("DD",  "drastic disjunction",     "true unless all are zeros")


def andness_label(a: float) -> str:
    """Short GCD code (e.g. HC+, A, SD) for andness ``a``."""
    return gcd_info(a)[0]


# ---------------------------------------------------------- tree extraction
@torch.no_grad()
def extract_tree(head, h: int):
    """Build the frozen DAG for head ``h`` as a nested node dict.

    node = {"kind": "leaf", "concept": k, "neg": bool}
         | {"kind": "op", "andness": float, "children": [(weight, node), ...]}

    Children weights mirror the head's inference-time ``_child_weights``:
    ``E = frozen_route * relevance``; normalize over sources; if a destination is
    orphaned (no routed source) fall back to a uniform mean over all sources.
    """
    depth = head.depth
    widths = head.widths
    neg = None
    if head.use_negation and bool(head.transform_frozen):
        neg = head.frozen_transform[h]                          # (K,) in {0,1} 1=identity

    def leaf(k):
        is_neg = bool(head.use_negation and neg is not None and neg[k].item() < 0.5)
        return {"kind": "leaf", "concept": int(k), "neg": is_neg}

    # nodes[l] = list of node dicts for each index at layer l
    nodes = [[leaf(k) for k in range(widths[0])]]
    for l in range(depth):
        w_in, w_out = widths[l], widths[l + 1]
        route = getattr(head, f"frozen_route_{l}")[h]          # (w_in, w_out) in {0,1}
        rel = (torch.exp(head.coeff_logits[l][h].clamp(-10, 10))
               if head.use_coefficients else torch.ones(w_in))
        E = route * rel.unsqueeze(1)                           # (w_in, w_out)
        a_bias = head.andness_bias[l][h]                       # (w_out,)
        if head.normalize_andness:
            andness = (torch.sigmoid(a_bias) * 3.0 - 1.0)
        else:
            andness = a_bias
        layer_nodes = []
        for d in range(w_out):
            col = E[:, d]
            if float(col.sum()) < 1e-6:                        # orphaned -> uniform
                srcs = list(range(w_in))
                ws = [1.0 / w_in] * w_in
            else:
                w = col / col.sum()
                srcs = [s for s in range(w_in) if float(col[s]) > 0]
                ws = [float(w[s]) for s in srcs]
            children = [(ws[i], nodes[l][srcs[i]]) for i in range(len(srcs))]
            layer_nodes.append({"kind": "op", "andness": float(andness[d]),
                                "children": children})
        nodes.append(layer_nodes)
    return nodes[depth][0]                                      # root (width 1)


def symbolic_eval(node, c_vec):
    """Evaluate an extracted node on a concept vector ``c_vec`` (K,), in [0,1].

    Mirrors the head exactly: leaves apply identity/negation, ops use the LSP
    andness power mean with the extracted child weights.  Used to PROVE the
    symbolic reading reproduces the numeric head (faithfulness of extraction).
    """
    if node["kind"] == "leaf":
        v = float(c_vec[node["concept"]])
        return (1.0 - v) if node["neg"] else v
    vals, ws = [], []
    for w, ch in node["children"]:
        vals.append(symbolic_eval(ch, c_vec))
        ws.append(w)
    X = torch.tensor(vals, dtype=torch.float32).view(-1, 1)
    W = torch.tensor(ws, dtype=torch.float32).view(-1, 1)
    W = W / W.sum().clamp_min(1e-8)
    out = lsp_power_mean(X, torch.tensor(float(node["andness"])), W, eps=1e-6)
    return float(out.view(-1)[0])


def count_literals(node, seen=None):
    if node["kind"] == "leaf":
        return 1
    return sum(count_literals(ch, seen) for _, ch in node["children"])


# ----------------------------------------------------------- text formatting
def format_tree(node, names, indent=0, weight=None, max_children=None,
                min_weight=0.0):
    """Indented, human-readable rendering of a rule tree."""
    pad = "  " * indent
    wtxt = "" if weight is None else f"[{weight:.2f}] "
    if node["kind"] == "leaf":
        nm = names[node["concept"]]
        lit = f"NOT {nm}" if node["neg"] else nm
        return f"{pad}{wtxt}{lit}\n"
    a = node["andness"]
    code, name, _ = gcd_info(a)
    line = f"{pad}{wtxt}{code} ({name}, andness={a:+.2f}, {len(node['children'])} children)\n"
    kids = sorted(node["children"], key=lambda t: -t[0])
    kids = [k for k in kids if k[0] >= min_weight]
    if max_children is not None and len(kids) > max_children:
        shown, rest = kids[:max_children], kids[max_children:]
    else:
        shown, rest = kids, []
    for w, ch in shown:
        line += format_tree(ch, names, indent + 1, w, max_children, min_weight)
    if rest:
        line += "  " * (indent + 1) + f"... (+{len(rest)} smaller-weight)\n"
    return line


def tree_stats(node):
    """Unique-op / unique-leaf / expanded-leaf-path counts for a shared DAG."""
    ops, leaves, paths = set(), set(), [0]

    def walk(n):
        if n["kind"] == "leaf":
            leaves.add((n["concept"], n["neg"]))
            paths[0] += 1
            return
        if id(n) in ops:
            return                      # shared subtree: count structure once
        ops.add(id(n))
        for _, ch in n["children"]:
            walk(ch)
    walk(node)
    return len(ops), len(leaves), paths[0]


def format_tree_dedup(node, names, max_children=None, min_weight=0.0):
    """Like :func:`format_tree` but renders each shared subtree ONCE.

    Repeated subtrees (the DAG reuses nodes when ``max_egress>1``) are labelled
    ``[N#]`` on first expansion and referenced as ``-> N#`` thereafter, so the
    printout reflects the true (small) structure instead of the expanded paths.
    """
    seen = {}                           # id(op node) -> label
    counter = [0]
    out = []

    def rec(n, indent, weight):
        pad = "  " * indent
        wtxt = "" if weight is None else f"[{weight:.2f}] "
        if n["kind"] == "leaf":
            nm = names[n["concept"]]
            lit = f"NOT {nm}" if n["neg"] else nm
            out.append(f"{pad}{wtxt}{lit}\n")
            return
        nid = id(n)
        if nid in seen:
            out.append(f"{pad}{wtxt}-> {seen[nid]}\n")
            return
        counter[0] += 1
        label = f"N{counter[0]}"
        seen[nid] = label
        a = n["andness"]
        code, name, _ = gcd_info(a)
        out.append(f"{pad}{wtxt}{code} [{label}] "
                   f"({name}, andness={a:+.2f}, {len(n['children'])} children)\n")
        kids = sorted(n["children"], key=lambda t: -t[0])
        kids = [k for k in kids if k[0] >= min_weight]
        rest = []
        if max_children is not None and len(kids) > max_children:
            kids, rest = kids[:max_children], kids[max_children:]
        for w, ch in kids:
            rec(ch, indent + 1, w)
        if rest:
            out.append("  " * (indent + 1) + f"... (+{len(rest)} smaller-weight)\n")

    rec(node, 0, None)
    return "".join(out)


def leaf_summary(node, names):
    """Flat top-literal summary: signed relevance = product of weights on the path."""
    acc = {}

    def walk(n, w):
        if n["kind"] == "leaf":
            key = (n["concept"], n["neg"])
            acc[key] = acc.get(key, 0.0) + w
        else:
            for cw, ch in n["children"]:
                walk(ch, w * cw)
    walk(node, 1.0)
    items = sorted(acc.items(), key=lambda kv: -kv[1])
    out = []
    for (k, neg), w in items:
        nm = names[k]
        out.append((f"NOT {nm}" if neg else nm, w))
    return out


# ----------------------------------------------------------------- subcommands
def do_rules(model, heads, names112, species_names, max_children, min_weight):
    head = model.head
    print("\n" + "=" * 78)
    print("SYMBOLIC RULE EXTRACTION  (nested AND/OR/NOT over named CUB attributes)")
    print("=" * 78)
    for h in heads:
        node = extract_tree(head, h)
        n_ops, n_leaves, n_paths = tree_stats(node)
        print(f"\n### species {h}: {species_names[h]}   "
              f"({n_ops} ops, {n_leaves} distinct literals, {n_paths} expanded paths)")
        print(format_tree_dedup(node, names112, max_children=max_children,
                                min_weight=min_weight), end="")
        print("  top literals (path-weighted):")
        for lit, w in leaf_summary(node, names112)[:12]:
            print(f"    {w:6.3f}  {lit}")


def do_export_all(model, names112, species_names, out_path, max_children, min_weight):
    """Write the dedup rule + top literals for ALL 200 species to a text file."""
    head = model.head
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("CUB full-tree OCBM (b8+eg2+negation+supervised, 71.25% frozen)\n")
        f.write("Per-species graded-logic rules over the 112 named CUB attributes.\n")
        f.write("Shared subtrees are labelled [N#] and referenced as -> N#.\n")
        for h in range(model.head.num_heads):
            node = extract_tree(head, h)
            n_ops, n_leaves, n_paths = tree_stats(node)
            f.write("\n" + "=" * 78 + "\n")
            f.write(f"### species {h}: {species_names[h]}   "
                    f"({n_ops} ops, {n_leaves} distinct literals, {n_paths} paths)\n")
            f.write(format_tree_dedup(node, names112, max_children=max_children,
                                      min_weight=min_weight))
            f.write("  top literals (path-weighted):\n")
            for lit, w in leaf_summary(node, names112)[:15]:
                f.write(f"    {w:6.3f}  {lit}\n")
    print(f"  wrote all {model.head.num_heads} species rules -> {out_path}")



@torch.no_grad()
def do_audit(model, heads, loader, device):
    """Prove the extracted symbolic rules reproduce the head, and report frozen acc."""
    head = model.head
    print("\n" + "=" * 78)
    print("FAITHFULNESS AUDIT")
    print("=" * 78)
    # (a) extraction fidelity: symbolic_eval(root) vs head(c) on a concept batch.
    C, _ = collect(model, loader, device)
    Cb = C[:256]
    trees = {h: extract_tree(head, h) for h in heads}
    max_abs = 0.0
    for h in heads:
        node = trees[h]
        head_out = head(Cb.to(device))[:, h].cpu()             # (b,)
        sym = torch.tensor([symbolic_eval(node, Cb[i]) for i in range(Cb.size(0))])
        d = (sym - head_out).abs().max().item()
        max_abs = max(max_abs, d)
        print(f"  species {h:3d}: max|symbolic - head| = {d:.2e}")
    print(f"  -> worst extraction error across shown species: {max_abs:.2e} "
          f"({'LOSSLESS' if max_abs < 1e-4 else 'CHECK'})")
    # (b) whole-model frozen accuracy on the test set.
    correct = total = 0
    for img, c, y in loader:
        img, y = img.to(device), y.to(device)
        correct += (model(img)[0].argmax(1) == y).sum().item()
        total += y.numel()
    print(f"  frozen test accuracy: {100.0 * correct / total:.2f}%  (N={total})")


@torch.no_grad()
def do_grounding(model, loader, device, names112, topn=15):
    """Per-concept ROC-AUC of the emergent concept vs its pinned CUB attribute."""
    print("\n" + "=" * 78)
    print("CONCEPT-GROUNDING CHECK  (per-concept AUC vs its NAMED CUB attribute)")
    print("=" * 78)
    C, A = collect(model, loader, device)                      # (N,K), (N,112)
    K = C.shape[1]
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
        print(f"    AUC {au:.3f}  c{i:3d}  {names112[i]}")
    print(f"\n  weakest-grounded {topn} concepts:")
    for au, i in ordered[-topn:]:
        print(f"    AUC {au:.3f}  c{i:3d}  {names112[i]}")


@torch.no_grad()
def do_examples(model, heads, loader, device, names112, species_names,
                n_examples=3):
    """For a few correctly-classified test birds per species, show fired literals."""
    print("\n" + "=" * 78)
    print("PER-EXAMPLE EXPLANATIONS  (literals that fired for real test birds)")
    print("=" * 78)
    head = model.head
    # gather test concept probs + labels + predictions
    imgs_seen = {h: 0 for h in heads}
    hset = set(heads)
    for img, c, y in loader:
        img = img.to(device)
        logits, C, _ = model(img)
        pred = logits.argmax(1).cpu()
        C = C.cpu()
        y = y
        for b in range(img.size(0)):
            yl = int(y[b])
            if yl in hset and imgs_seen[yl] < n_examples and int(pred[b]) == yl:
                imgs_seen[yl] += 1
                node = extract_tree(head, yl)
                truth = symbolic_eval(node, C[b])
                print(f"\n  {species_names[yl]} (species {yl}) "
                      f"pred={'OK' if int(pred[b]) == yl else 'X'} "
                      f"root-truth={truth:.3f}")
                lits = leaf_summary(node, names112)
                cv = C[b]
                # show the highest path-weight literals and whether they fired
                shown = 0
                for lit, w in lits:
                    # recover concept idx + negation from the label text
                    neg = lit.startswith("NOT ")
                    nm = lit[4:] if neg else lit
                    k = names112.index(nm)
                    fired = (1.0 - float(cv[k])) if neg else float(cv[k])
                    mark = "+" if fired >= 0.5 else "-"
                    print(f"      {mark} truth={fired:.2f} w={w:.3f}  {lit}")
                    shown += 1
                    if shown >= 10:
                        break
        if all(imgs_seen[h] >= n_examples for h in heads):
            break


# ------------------------------------------------------------- visualization
def _viz_node_html(node, names, seen=None, counter=None):
    if seen is None:
        seen, counter = {}, [0]
    if node["kind"] == "leaf":
        nm = html.escape(names[node["concept"]])
        cls = "lit neg" if node["neg"] else "lit"
        txt = ("NOT " + nm) if node["neg"] else nm
        return f'<li class="{cls}">{txt}</li>'
    nid = id(node)
    if nid in seen:                     # shared subtree: reference, don't re-expand
        return (f'<details class="shared"><summary class="op ref">'
                f'&#8618; {seen[nid]} (shared)</summary></details>')
    counter[0] += 1
    label = f"N{counter[0]}"
    seen[nid] = label
    a = node["andness"]
    code, name, _ = gcd_info(a)
    kids = sorted(node["children"], key=lambda t: -t[0])
    inner = "".join(
        f'<li class="edge"><span class="w">{w:.2f}</span>'
        f'<ul>{_viz_node_html(ch, names, seen, counter)}</ul></li>'
        for w, ch in kids)
    op = html.escape(f"{code}  \u00b7 {name}  [{label}]  "
                     f"(andness {a:+.2f}, {len(kids)} children)")
    return (f'<details open><summary class="op">{op}</summary>'
            f'<ul>{inner}</ul></details>')


def do_viz(model, heads, names112, species_names, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    head = model.head
    for h in heads:
        node = extract_tree(head, h)
        body = _viz_node_html(node, names112)
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
<h1>{title} &mdash; graded-logic rule (frozen, faithful)</h1>
<p>Blue = graded-logic aggregator, labelled with its GCD degree (andness a:
CC/HHC/CP/LHC/C = conjunctive a&gt;=1, A = neutral a=0.5, D/LHD/DP/HHD/DD =
disjunctive a&lt;=0, with hard/soft grades HC/SC and HD/SD in between);
green = attribute literal; red = negated literal. Weights are normalized child
relevances used in the power mean. Each shared subtree is expanded once as
<b>[N#]</b>; later reuses show as <b>&#8618; N# (shared)</b>.</p>
{body}
</body></html>"""
        out = os.path.join(out_dir, f"rule_species{h:03d}.html")
        with open(out, "w", encoding="utf-8") as f:
            f.write(page)
        print(f"  wrote {out}")


# ------------------------------------------------------------------------ main
def parse_species(spec, species_names):
    if spec is None:
        return [0, 10, 50]
    out = []
    for tok in spec.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if tok.isdigit():
            out.append(int(tok))
        else:
            # name substring match
            hits = [i for i, n in enumerate(species_names)
                    if tok.lower() in n.lower()]
            if not hits:
                raise SystemExit(f"no species matches {tok!r}")
            out.append(hits[0])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--species", default=None,
                    help="comma list of class indices or name substrings (default 0,10,50)")
    ap.add_argument("--rules", action="store_true")
    ap.add_argument("--audit", action="store_true")
    ap.add_argument("--grounding", action="store_true")
    ap.add_argument("--examples", action="store_true")
    ap.add_argument("--viz", action="store_true")
    ap.add_argument("--export-all", action="store_true",
                    help="write dedup rules for ALL 200 species to a text file")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--max-children", type=int, default=6,
                    help="max children shown per node in text rules (0/None = all)")
    ap.add_argument("--min-weight", type=float, default=0.0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--out-dir", default=os.path.join(_HERE, "results", "rules"))
    args = ap.parse_args()

    if args.all:
        args.rules = args.audit = args.grounding = args.examples = args.viz = True
    if not any([args.rules, args.audit, args.grounding, args.examples,
                args.viz, args.export_all]):
        args.rules = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(args.ckpt):
        raise SystemExit(f"checkpoint not found: {args.ckpt}")
    species_names = load_species_names()
    names112, *_ = _cub._load_attr_groups()
    heads = parse_species(args.species, species_names)

    model, K, branching, negation, coeff, max_egress = load_model(args.ckpt, device)
    print(f"loaded {os.path.basename(args.ckpt)}  K={K} branching={branching} "
          f"egress={max_egress} negation={negation} coeff={coeff} "
          f"widths={model.head.widths}")

    need_data = args.audit or args.grounding or args.examples
    loader = None
    if need_data:
        from torch.utils.data import DataLoader
        loader = DataLoader(_cub._CUBImages("test", False), batch_size=128,
                            shuffle=False, num_workers=args.workers, pin_memory=True)

    mc = None if not args.max_children else args.max_children
    if args.rules:
        do_rules(model, heads, names112, species_names, mc, args.min_weight)
    if args.audit:
        do_audit(model, heads, loader, device)
    if args.grounding:
        do_grounding(model, loader, device, names112)
    if args.examples:
        do_examples(model, heads, loader, device, names112, species_names)
    if args.viz:
        do_viz(model, heads, names112, species_names, args.out_dir)
    if args.export_all:
        do_export_all(model, names112, species_names,
                      os.path.join(args.out_dir, "all_species_rules.txt"),
                      mc, args.min_weight)


if __name__ == "__main__":
    main()
