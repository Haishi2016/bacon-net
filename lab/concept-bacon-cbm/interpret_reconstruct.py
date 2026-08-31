r"""Interpret--reconstruct reasoning-recovery pipeline (LCA / DRA / SRA / RCA / CON_M).

Measures whether a learned graded-logic rule, once *verbalized* by an interpreter
LLM (LLM_I) with NO access to samples or the class name, can be *reconstructed*
back to the intended class by an independent reasoner LLM (LLM_R) that sees ONLY
the verbalization (no logic structure, no samples, no model prediction).

Two model configs are verbalized so we can test the explainability/accuracy
tradeoff of simplification:
  * full@112  -- the unpruned full-tree rule per class.
  * full@k    -- the IG-pruned rule (top-k leaves by GL-native integrated-gradients
                 importance; default k=32), which should verbalize/reconstruct
                 better because it is far smaller and less redundant.

The prompts are BATCHED (all classes in one prompt, chunked to fit context) and
provider-agnostic: run each in a VS Code Copilot Chat session, pick a model,
paste, save the CSV reply. Use different models for LLM_I vs LLM_R, and several
R models for CON_M; swap roles and average.

Subcommands
-----------
  prototype       compute per-class model decision yhat_L (argmax on class prototype)
                  and the private rule_id<->class map -> results/ir/manifest.json
  gen-interpret   write batched LLM_I prompts (hidden class names) for each config
  gen-reconstruct given an LLM_I verbalization CSV, write the batched LLM_R prompt
  analyze         given LLM_R reconstruction CSV(s), compute LCA/DRA/SRA/RCA/CON_M

    py -3 interpret_reconstruct.py prototype
    py -3 interpret_reconstruct.py gen-interpret --config full112 --chunk 25
    py -3 interpret_reconstruct.py gen-interpret --config full32  --chunk 40
    py -3 interpret_reconstruct.py gen-reconstruct --verbalized results/ir/verb_full32.csv --config full32
    py -3 interpret_reconstruct.py analyze --config full32 --recon results/ir/recon_full32_*.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import random
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "table"))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

import _cub                                                          # noqa: E402
from interpret_fulltree_cub import (                                 # noqa: E402
    DEFAULT_CKPT, load_model, load_species_names, extract_tree,
    gcd_info, leaf_summary)
from verbalize_bird_rules import (                                   # noqa: E402
    build_json, _readable, _GCD_TABLE, class_mean_concepts)
from prune_and_infer_cub import ig_importance, head_negation         # noqa: E402

_IR = os.path.join(_HERE, "results", "ir")
CONFIGS = {"full112": 112, "full32": 32, "full24": 24, "full16": 16}


# --------------------------------------------------------------- tree pruning
def filter_tree(node, keep, _memo=None):
    """Prune the extracted DAG to leaves whose concept is in ``keep`` (a set of
    concept indices); drop internal nodes left with no surviving child. Memoized
    by node id so SHARED DAG nodes map to the SAME filtered object (preserving
    the DAG structure, so ``build_json_dag`` can still detect the reuse)."""
    if _memo is None:
        _memo = {}
    nid = id(node)
    if nid in _memo:
        return _memo[nid]
    if node["kind"] == "leaf":
        res = node if node["concept"] in keep else None
        _memo[nid] = res
        return res
    kids = []
    for w, ch in node["children"]:
        f = filter_tree(ch, keep, _memo)
        if f is not None:
            kids.append((w, f))
    res = None if not kids else {"kind": "op", "andness": node["andness"], "children": kids}
    _memo[nid] = res
    return res


def build_json_dag(node, names, mc, md, mw):
    """DAG-faithful serialization: the head is a DAG (``max_egress`` > 1 lets a
    sub-node feed two parents), so a plain tree expansion duplicates every shared
    sub-rule -- which makes the verbalization redundant ("duplicated profiles").

    Here each distinct op node is emitted ONCE with an ``id``; a later reuse of
    the same node is emitted as ``{"weight": w, "shared_ref": <id>}`` so the
    reader evaluates/describes it only once. Leaves stay inline (they are cheap).
    """
    labels = {}
    counter = [0]

    def rec(nd, depth):
        if nd["kind"] == "leaf":
            nm = names[nd["concept"]]
            return {"feature": _readable(("NOT " + nm) if nd["neg"] else nm)}
        nid = id(nd)
        if nid in labels:
            return {"shared_ref": labels[nid]}
        counter[0] += 1
        label = f"N{counter[0]}"
        labels[nid] = label
        a = float(nd["andness"])
        code, gname, verbal = gcd_info(a)
        head = {"id": label, "operator": code, "name": gname,
                "andness": round(a, 3), "verbalization": verbal}
        if depth >= md:
            kids = [{"weight": round(w, 3), "feature": _readable(lit)}
                    for lit, w in leaf_summary(nd, names)[:mc]]
            return {**head, "children": kids}
        kids = sorted(nd["children"], key=lambda t: -t[0])
        kids = [k for k in kids if k[0] >= mw][:mc]
        ch = []
        for w, c in kids:
            ch.append({"weight": round(w, 3), **rec(c, depth + 1)})
        if not ch:
            ch = [{"weight": round(w, 3), "feature": _readable(lit)}
                  for lit, w in leaf_summary(nd, names)[:mc]]
        return {**head, "children": ch}

    return rec(node, 0)




# ------------------------------------------------------------------ manifest
def _manifest_path():
    return os.path.join(_IR, "manifest.json")


def load_manifest():
    with open(_manifest_path(), "r", encoding="utf-8") as f:
        return json.load(f)


@torch.no_grad()
def cmd_prototype(args):
    """Compute yhat_L per class (argmax of the head over the class prototype) and
    a shuffled rule_id<->class map, saved to the manifest. yhat_L operationalizes
    'the class the model's logic assigns to class y's prototype' -- LCA=P(yhat_L=y)."""
    device = torch.device(args.device)
    model, K, *_ = load_model(args.ckpt, device)
    head = model.head
    H = head.num_heads
    protos = class_mean_concepts(model, device, args.workers, list(range(H)))  # {h:[K]}
    P = torch.stack([protos[h] for h in range(H)]).to(device)                  # [H,K]
    T = head(P)                                                                # [H,H]
    yhat_L = T.argmax(1).cpu().tolist()                                        # per class
    # private opaque rule ids (shuffled) so the reconstructor cannot exploit order
    rng = random.Random(args.seed)
    order = list(range(H))
    rng.shuffle(order)
    rid_of_class = {c: f"R{order.index(c) + 1:03d}" for c in range(H)}
    os.makedirs(_IR, exist_ok=True)
    manifest = {"num_classes": H, "seed": args.seed,
                "yhat_L": yhat_L,                       # yhat_L[class] = predicted class
                "rid_of_class": rid_of_class,           # class -> rule id
                "class_of_rid": {v: k for k, v in rid_of_class.items()}}
    with open(_manifest_path(), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    lca = sum(int(yhat_L[c] == c) for c in range(H)) / H
    print(f"wrote {_manifest_path()}  (H={H})")
    print(f"  prototype LCA = P(yhat_L=y) = {lca*100:.2f}%  "
          f"(self-consistency of the logic on its own class prototypes)")


# --------------------------------------------------------- interpreter prompts
_L_EXPLAINER = f"""\
You are reasoning about a Graded Logic (GL / LSP) decision rule. A rule is a tree
of aggregators over fuzzy attribute-truths in [0,1] (1 = attribute present; "NOT"
= the attribute is absent). Each internal node combines its weighted children
with an *andness* (a GCD code) that sets how strictly the children must hold:
high-andness nodes are conjunctive ("must have all" / decided by the LOWEST
child), low/negative-andness nodes are disjunctive ("enough to have any" / decided
by the HIGHEST child), and mid values are soft averages. Weights are relative
relevances inside the weighted power mean. A node may carry an "id"; a child of
the form {{"weight": w, "shared_ref": <id>}} means the SAME sub-rule (already
given above under that id) is reused as another input -- evaluate/describe it
ONCE, do not treat it as a separate duplicate profile. GCD codes:

{_GCD_TABLE}
IMPORTANT: Reason ONLY from the information in this prompt. Do NOT use outside
knowledge, memorized facts, or long-term memory beyond what is stated here."""


def _interpreter_prompt(rules):
    """rules: list of (rid, tree_json). Returns one batched LLM_I prompt string."""
    blocks = []
    for rid, tj in rules:
        blocks.append(f"### {rid}\n```json\n{json.dumps(tj, indent=1)}\n```")
    body = "\n\n".join(blocks)
    return f"""{_L_EXPLAINER}

TASK
====
Below are {len(rules)} graded-logic decision rules, each identified only by an
opaque id (the class it detects is hidden). For EACH rule, write a concise
natural-language decision rule (1-3 sentences) that states what an input must
look like to satisfy it -- naming the key attributes and how they are logically
combined (which are required/conjunctive, which are alternatives/disjunctive,
which must be ABSENT). Describe ONLY the logic; do not name or guess any class.

OUTPUT
======
Return a CSV with a header row and one row per rule, EXACTLY:
rule_id,verbalization
Quote the verbalization field (wrap in double quotes; escape internal quotes).
Output only the CSV, nothing else.

RULES
=====
{body}
"""


def cmd_gen_interpret(args):
    device = torch.device(args.device)
    model, K, *_ = load_model(args.ckpt, device)
    head = model.head
    H = head.num_heads
    names112 = _cub._load_attr_groups()[0]
    man = load_manifest()
    rid_of_class = {int(k): v for k, v in man["rid_of_class"].items()}
    k = CONFIGS[args.config]

    k = CONFIGS[args.config]

    keepsets = None
    if k < K:
        print(f"computing IG importance for top-{k} keepsets ...", flush=True)
        protos = class_mean_concepts(model, device, args.workers, list(range(H)))
        neg = head_negation(head)
        keepsets = {}
        for h in range(H):
            phi = ig_importance(head, h, protos[h], neg[h], device)
            keepsets[h] = set(phi.argsort(descending=True)[:k].tolist())
            if (h + 1) % 40 == 0:
                print(f"    {h+1}/{H}", flush=True)

    rules = []
    for h in range(H):
        node = extract_tree(head, h)
        if keepsets is not None:
            node = filter_tree(node, keepsets[h]) or node
        tj = build_json_dag(node, names112, mc=args.max_children, md=args.max_depth,
                            mw=args.min_weight)
        rules.append((rid_of_class[h], tj))

    out = os.path.join(_IR, "interpret", args.config)
    os.makedirs(out, exist_ok=True)
    n = args.chunk
    chunks = [rules[i:i + n] for i in range(0, len(rules), n)]
    for ci, ch in enumerate(chunks):
        p = os.path.join(out, f"interpret_{args.config}_chunk{ci+1:02d}of{len(chunks):02d}.md")
        with open(p, "w", encoding="utf-8") as f:
            f.write(_interpreter_prompt(ch))
    print(f"wrote {len(chunks)} interpreter prompt(s) to {out}  "
          f"({len(rules)} rules, {n}/chunk). Run each in a chat session (model = LLM_I), "
          f"append the CSV rows into one file results/ir/verb_{args.config}.csv "
          f"(header: rule_id,verbalization).")


# -------------------------------------------------------- reconstructor prompt
def _reconstructor_prompt(species_names, verbs):
    """species_names: list[str] (candidate closed set). verbs: list[(rid, text)]."""
    cand = "\n".join(f"{i+1}. {nm}" for i, nm in enumerate(species_names))
    rule_lines = "\n".join(f"{rid}\t{txt}" for rid, txt in verbs)
    return f"""You are an expert ornithologist reasoning about CUB-200 bird species.
Below are decision rules, each describing ONE bird species using its visual
attributes. For each rule, identify the species it most likely describes.

IMPORTANT: Reason ONLY from the rule text and the candidate list in this prompt.
Do NOT use outside knowledge or memory of specific image datasets; rely only on
the attribute semantics stated in each rule and general bird-attribute reasoning.

CANDIDATE SPECIES (choose only from these {len(species_names)} names)
=================
{cand}

RULES (rule_id <TAB> rule text)
=====
{rule_lines}

TASK
====
For each rule, output the 5 most likely candidate species, ranked most- to
least-likely, using the EXACT names from the candidate list.

OUTPUT
======
Return a CSV with a header row and one row per rule, EXACTLY:
rule_id,rank1,rank2,rank3,rank4,rank5
Output only the CSV, nothing else.
"""


def _read_verbalized(path):
    verbs = {}
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rid = (row.get("rule_id") or "").strip()
            txt = (row.get("verbalization") or "").strip()
            if rid:
                verbs[rid] = txt                             # dedup: keep last occurrence
    return list(verbs.items())


def cmd_gen_reconstruct(args):
    species = load_species_names()
    verbs = _read_verbalized(args.verbalized)
    out = os.path.join(_IR, "reconstruct", args.config)
    os.makedirs(out, exist_ok=True)
    n = args.chunk
    chunks = [verbs[i:i + n] for i in range(0, len(verbs), n)]
    for ci, ch in enumerate(chunks):
        p = os.path.join(out, f"reconstruct_{args.config}_chunk{ci+1:02d}of{len(chunks):02d}.md")
        with open(p, "w", encoding="utf-8") as f:
            f.write(_reconstructor_prompt(species, ch))
    print(f"wrote {len(chunks)} reconstruction prompt(s) to {out} ({len(verbs)} rules). "
          f"Run each in a SEPARATE chat session per reconstruction model (LLM_R), "
          f"save each model's CSV as results/ir/recon_{args.config}_<model>.csv "
          f"(header: rule_id,rank1..rank5).")


# ------------------------------------------------------------------- analysis
def _norm(name):
    """Normalize a species name for matching: lowercase, alphanumeric only."""
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _name_to_class(species):
    return {_norm(nm): i for i, nm in enumerate(species)}


def _read_recon(path, name2c, class_of_rid):
    """Return ({class: [ranked class ids]}, n_unmatched) from a reconstruction CSV."""
    out = {}
    unmatched = 0
    with open(path, "r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            rid = (row.get("rule_id") or "").strip()
            if rid not in class_of_rid:
                continue
            cls = int(class_of_rid[rid])
            ranked = []
            for r in ("rank1", "rank2", "rank3", "rank4", "rank5"):
                nm = _norm((row.get(r) or "").strip())
                cid = name2c.get(nm, -1)
                if nm and cid < 0:
                    unmatched += 1
                ranked.append(cid)
            out[cls] = ranked
    return out, unmatched


def cmd_analyze(args):
    species = load_species_names()
    name2c = _name_to_class(species)
    man = load_manifest()
    H = man["num_classes"]
    yhat_L = man["yhat_L"]
    class_of_rid = man["class_of_rid"]

    recon_files = []
    for pat in args.recon:
        recon_files.extend(sorted(glob.glob(pat)))
    if not recon_files:
        raise SystemExit(f"no reconstruction CSVs matched {args.recon}")
    parsed = [_read_recon(p, name2c, class_of_rid) for p in recon_files]
    models = [p[0] for p in parsed]
    print(f"loaded {len(models)} reconstruction model CSV(s): "
          f"{[os.path.basename(p) for p in recon_files]}")
    for p, (_, unm) in zip(recon_files, parsed):
        if unm:
            print(f"  [warn] {os.path.basename(p)}: {unm} ranked name(s) did not match "
                  f"any of the 200 species (counted as no-match).")

    KS = (1, 3, 5)

    def metrics(recon):
        n = 0
        lca = 0
        dra = {k: 0 for k in KS}
        sra = {k: 0 for k in KS}
        rca_num = {k: 0 for k in KS}
        rca_den = {k: 0 for k in KS}
        RC = RW = NC = NW = 0                                     # top-1 matrix
        for c in range(H):
            if c not in recon:
                continue
            n += 1
            ranked = recon[c]
            yL = yhat_L[c]
            correct = (yL == c)
            lca += int(correct)
            for k in KS:
                topk = ranked[:k]
                recov = yL in topk
                if recov:
                    dra[k] += 1
                    rca_den[k] += 1
                    rca_num[k] += int(correct)
                if c in topk:
                    sra[k] += 1
            # top-1 alignment matrix
            rec1 = (ranked[0] == yL)
            if rec1 and correct: RC += 1
            elif rec1 and not correct: RW += 1
            elif (not rec1) and correct: NC += 1
            else: NW += 1
        out = {"n": n, "LCA": lca / n if n else 0.0,
               "RC": RC, "RW": RW, "NC": NC, "NW": NW}
        for k in KS:
            out[f"DRA{k}"] = dra[k] / n if n else 0.0
            out[f"SRA{k}"] = sra[k] / n if n else 0.0
            out[f"RCA{k}"] = rca_num[k] / rca_den[k] if rca_den[k] else float("nan")
        return out

    for p, recon in zip(recon_files, models):
        m = metrics(recon)
        print(f"\n===== {os.path.basename(p)}  (N={m['n']}) =====")
        print(f"  LCA = {m['LCA']*100:.2f}%   (P(yhat_L=y); top-1 matrix "
              f"RC/RW/NC/NW = {m['RC']}/{m['RW']}/{m['NC']}/{m['NW']})")
        print(f"  {'k':>3} {'DRA':>7} {'SRA':>7} {'RCA':>7}")
        for k in KS:
            rca = m[f'RCA{k}']
            rca_s = "  nan " if rca != rca else f"{rca*100:6.2f}%"
            print(f"  {k:>3} {m[f'DRA{k}']*100:6.2f}% {m[f'SRA{k}']*100:6.2f}% {rca_s}")

    # CON_M: Fleiss per-item agreement on the top-1 reconstruction across models.
    M = len(models)
    if M >= 2:
        from collections import Counter
        Pi_sum = 0.0
        cnt = 0
        for c in range(H):
            top1 = [rec[c][0] for rec in models if c in rec]
            if len(top1) < 2:
                continue
            m = len(top1)
            counts = Counter(top1)
            Pi = sum(v * (v - 1) for v in counts.values()) / (m * (m - 1))
            Pi_sum += Pi
            cnt += 1
        con = Pi_sum / cnt if cnt else float("nan")
        print(f"\n  CON_M (top-1 reconstruction consensus, Fleiss per-item, "
              f"{M} models) = {con*100:.1f}%  over {cnt} classes")



def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("prototype")
    p.add_argument("--ckpt", default=DEFAULT_CKPT)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.set_defaults(func=cmd_prototype)

    p = sub.add_parser("gen-interpret")
    p.add_argument("--config", choices=list(CONFIGS), default="full32")
    p.add_argument("--ckpt", default=DEFAULT_CKPT)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--chunk", type=int, default=25)
    p.add_argument("--max-children", type=int, default=6)
    p.add_argument("--max-depth", type=int, default=4)
    p.add_argument("--min-weight", type=float, default=0.0)
    p.set_defaults(func=cmd_gen_interpret)

    p = sub.add_parser("gen-reconstruct")
    p.add_argument("--verbalized", required=True, help="LLM_I CSV: rule_id,verbalization")
    p.add_argument("--config", choices=list(CONFIGS), default="full32")
    p.add_argument("--chunk", type=int, default=100)
    p.set_defaults(func=cmd_gen_reconstruct)

    p = sub.add_parser("analyze")
    p.add_argument("--config", choices=list(CONFIGS), default="full32")
    p.add_argument("--recon", nargs="+", required=True,
                   help="glob(s) for reconstruction CSVs: rule_id,rank1..rank5")
    p.set_defaults(func=cmd_analyze)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
