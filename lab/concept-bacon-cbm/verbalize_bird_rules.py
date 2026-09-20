r"""Verbalize a species' pruned graded-logic rule with an LLM.

The repo's "verbalization tool" is the prompt template ``llm/sample-prompt.md``:
it feeds an LSP/GL aggregation tree (operator = GCD code, weighted children) to
an LLM and asks for a plain-language report. This script produces exactly that
input for CUB species -- it extracts the frozen full-tree rule, PRUNES it to a
compact tree (top-weighted children, same knobs as the figure), serializes it to
the prompt's JSON schema, and writes a ready-to-paste prompt per species.

If an LLM SDK + key is available it can also call it directly (--call):
  * anthropic + ANTHROPIC_API_KEY, or openai + OPENAI_API_KEY.

    py -3 verbalize_bird_rules.py --species 0,10,50 --max-children 3 --max-depth 3
    py -3 verbalize_bird_rules.py --species Albatross --call
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "table"))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

import _cub                              # noqa: E402
from interpret_fulltree_cub import (     # noqa: E402
    DEFAULT_CKPT, load_model, load_species_names, extract_tree,
    gcd_info, leaf_summary, parse_species)

# GCD nomenclature table (Table 3) embedded so the LLM needs no external lookup.
_GCD_TABLE = """\
| Code | Name | Andness a | Verbalization |
|------|------|-----------|---------------|
| CC  | Drastic conjunction     | 2        | "must all be completely satisfied" |
| HHC | High hyper-conjunction  | [5/4, 2) | "below the lowest" |
| CP  | Product t-norm          | 5/4      | "below the lowest" |
| LHC | Low hyper-conjunction   | [1, 5/4) | "below the lowest" |
| C   | Pure conjunction        | 1        | "decided by the lowest" |
| HC+ | High hard conjunction   | 13/14    | "must have all" |
| HC  | Medium hard conjunction | 12/14    | "must have all" |
| HC- | Low hard conjunction    | 11/14    | "must have all" |
| SC+ | High soft conjunction   | 10/14    | "nice to have most" |
| SC  | Medium soft conjunction | 9/14     | "nice to have most" |
| SC- | Low soft conjunction    | 8/14     | "nice to have most" |
| A   | Arithmetic mean         | 7/14     | "nice to have" |
| SD- | Low soft disjunction    | 6/14     | "nice to have some" |
| SD  | Medium soft disjunction | 5/14     | "nice to have some" |
| SD+ | High soft disjunction   | 4/14     | "nice to have some" |
| HD- | Low hard disjunction    | 3/14     | "enough to have any" |
| HD  | Medium hard disjunction | 2/14     | "enough to have any" |
| HD+ | High hard disjunction   | 1/14     | "enough to have any" |
| D   | Pure disjunction        | 0        | "decided by the highest" |
| LHD | Low hyper-disjunction   | (-1/4, 0]| "above the highest" |
| DP  | Product t-conorm        | -1/4     | "above the highest" |
| HHD | High hyper-disjunction  | [-1, -1/4)| "above the highest" |
| DD  | Drastic disjunction     | -1       | "true unless all are zeros" |
"""


def _readable(name: str) -> str:
    """'has_wing_color::black' -> 'wing color: black' (keep NOT prefix)."""
    neg = name.startswith("NOT ")
    core = name[4:] if neg else name
    core = core.replace("has_", "").replace("::", ": ").replace("_", " ")
    return ("NOT " + core) if neg else core


# ------------------------------------------------------------- prune -> JSON
def build_json(node, names, mc, md, mw, depth=0):
    """Prune the extracted DAG and serialize to the verbalization JSON schema."""
    if node["kind"] == "leaf":
        nm = names[node["concept"]]
        return {"feature": _readable(("NOT " + nm) if node["neg"] else nm)}
    a = float(node["andness"])
    code, gname, verbal = gcd_info(a)
    head = {"operator": code, "name": gname, "andness": round(a, 3),
            "verbalization": verbal}
    if depth >= md:
        kids = [{"weight": round(w, 3), "feature": _readable(lit)}
                for lit, w in leaf_summary(node, names)[:mc]]
        return {**head, "children": kids}
    kids = sorted(node["children"], key=lambda t: -t[0])
    kids = [k for k in kids if k[0] >= mw][:mc]
    ch = []
    for w, c in kids:
        ch.append({"weight": round(w, 3), **build_json(c, names, mc, md, mw, depth + 1)})
    if not ch:
        ch = [{"weight": round(w, 3), "feature": _readable(lit)}
              for lit, w in leaf_summary(node, names)[:mc]]
    return {**head, "children": ch}


# ------------------------------------------------------------- prompt assembly
def build_prompt(species_name, tree_json):
    js = json.dumps(tree_json, indent=2)
    return f"""System
=======
You are a report generator that produces a precise, human-readable, and \
insightful explanation of a Graded Logic (GL / LSP) aggregation tree. Use your \
knowledge of graded logic and ornithology to generate a clear, logical report.

Context
=======
This aggregation tree is a fine-grained bird-species classifier for the CUB-200 \
dataset. It scores how well an image matches the species **{species_name}**. \
The leaf "features" are fuzzy attribute-truths in [0, 1] (a value near 1 means \
the attribute is present); "NOT" denotes a negated attribute. Each internal node \
is a graded-logic aggregator whose andness (its GCD code) sets how strictly its \
weighted children must be satisfied. Weights are the normalized relevances used \
inside the power mean.

Background
==========
GL aggregators are named by their andness per this table:

{_GCD_TABLE}

Instructions
============
1. Organize the report into:
   - Plain-Language Statement: ONE intuitive natural-English sentence that a \
birder would understand, summarizing what the rule looks for -- weave in your \
ornithological knowledge (do NOT mention GL/andness/operators here), e.g. \
"It calls a bird an {species_name} when it sees a mostly-blue songbird with a \
dark masked face and a stout bill."
   - Overview: the overall decision logic in 2-3 sentences.
   - Decision Logic Walkthrough: step by step from the root to the leaves. For \
each node, name the operator by its code AND verbalization (e.g. "HC+ - high \
hard conjunction, a strict 'must have all'"), and explain how it combines its \
children given their weights.
   - Ornithological Interpretation: relate the logic to the real field marks of \
{species_name}; note whether the rule matches known diagnostic features, and \
flag anything surprising, redundant, or spurious.
2. Use plain, human-friendly language.
3. Emphasize the highest-weight features and how they drive the decision.
4. Treat low-weight children as "compensatory / nice-to-have" evidence.

Input
======
The aggregation tree (pruned to its most influential children):
```json
{js}
```
"""


def call_llm(prompt):
    """Optionally call an installed LLM SDK. Returns text or None."""
    if importlib.util.find_spec("anthropic") and os.environ.get("ANTHROPIC_API_KEY"):
        import anthropic
        client = anthropic.Anthropic()
        msg = client.messages.create(
            model=os.environ.get("ANTHROPIC_MODEL", "claude-3-5-sonnet-latest"),
            max_tokens=1600, messages=[{"role": "user", "content": prompt}])
        return "".join(getattr(b, "text", "") for b in msg.content)
    if importlib.util.find_spec("openai") and os.environ.get("OPENAI_API_KEY"):
        from openai import OpenAI
        client = OpenAI()
        r = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o"),
            messages=[{"role": "user", "content": prompt}])
        return r.choices[0].message.content
    return None


@__import__("torch").no_grad()
def class_mean_concepts(model, device, workers, classes):
    """Mean concept-truth vector (K,) per requested class over the CUB test set."""
    import torch
    from torch.utils.data import DataLoader
    loader = DataLoader(_cub._CUBImages("test", False), batch_size=128,
                        shuffle=False, num_workers=workers, pin_memory=True)
    K = model.head.input_size
    cset = set(classes)
    sums = {h: torch.zeros(K) for h in cset}
    counts = {h: 0 for h in cset}
    for img, c, y in loader:
        C = model.concept_probs(img.to(device)).cpu()
        for b in range(len(y)):
            yl = int(y[b])
            if yl in cset:
                sums[yl] += C[b]
                counts[yl] += 1
    return {h: sums[h] / max(counts[h], 1) for h in cset}


# ----------------------------------------------------- plain-English "guess" mode
def _feat_phrase(name: str):
    """'has_wing_color::black' -> ('black wings', is_neg). Species-agnostic, no GL."""
    neg = name.startswith("NOT ")
    core = name[4:] if neg else name
    core = core.replace("has_", "")
    if "::" in core:
        part, value = core.split("::", 1)
    else:
        part, value = core, ""
    part = part.replace("_", " ").strip()
    value = value.replace("_", " ").strip()
    if part.endswith(" color"):
        body = part[:-6].strip()
        phrase = f"{value} {body}"                      # "grey underparts"
    elif part == "shape":
        phrase = f"a {value} shape"
    elif part.endswith("shape"):
        phrase = f"a {value} {part}"                    # "a hooked seabird bill shape"
    elif part == "size":
        phrase = f"{value} size"
    elif part.endswith("pattern"):
        phrase = f"a {value} {part}"                    # "a solid tail pattern"
    elif part.endswith("length"):
        phrase = f"{value} {part}"
    else:
        phrase = f"{value} {part}".strip()
    return phrase, neg


def build_guess_text(node, names, n_pos=8, n_neg=4, fired=None, min_fire=0.5):
    """A clear-English, species-agnostic description for a fresh LLM to guess.

    Ranks literals by path-weight (overall influence), translates each to a plain
    field-mark phrase, and buckets by confidence tier. No GL codes, no species
    name, no weights -- just how a birder might describe what they see.

    ``fired`` (optional, (K,) mean concept-truth over the species' test images)
    GROUNDS the description in reality: a literal is kept only if it actually
    holds for the species (``fired >= min_fire`` for a positive, or the attribute
    is typically absent for a NOT), and the rank uses weight x fired. This drops
    high-weight-but-rarely-true literals (e.g. an OR alternative) that otherwise
    mislead the guesser.
    """
    lits = leaf_summary(node, names)                    # [(lit, weight)] desc
    pos, neg = [], []
    for lit, w in lits:
        is_neg = lit.startswith("NOT ")
        nm = lit[4:] if is_neg else lit
        f = 1.0
        if fired is not None:
            k = names.index(nm)
            f = float(1.0 - fired[k]) if is_neg else float(fired[k])
            if f < min_fire:
                continue                                # not actually true -> skip
        phrase, _ = _feat_phrase(lit)
        if not phrase.strip():
            continue
        (neg if is_neg else pos).append((phrase, w * f))
    pos.sort(key=lambda t: -t[1])
    neg.sort(key=lambda t: -t[1])
    pos, neg = pos[:n_pos], neg[:n_neg]

    strong = [p for p, s in pos if s >= 0.08]
    usual = [p for p, s in pos if s < 0.08]
    nots = [p for p, _ in neg]
    return _format_guess(strong, usual, nots)


def build_guess_prompt(node, names, n_pos=10, n_neg=5):
    """LLM prompt that turns the ranked field marks into a natural riddle
    (species-agnostic, no GL). Use with --call for a more fluent description."""
    lits = leaf_summary(node, names)
    pos, neg = [], []
    for lit, w in lits:
        phrase, is_neg = _feat_phrase(lit)
        if phrase.strip():
            (neg if is_neg else pos).append(f"{phrase} (importance {w:.2f})")
    pos, neg = pos[:n_pos], neg[:n_neg]
    feats = "Present features (most to least important):\n- " + "\n- ".join(pos)
    if neg:
        feats += "\n\nAbsent features:\n- " + "\n- ".join(neg)
    return (
        "You are a birder describing a bird you are looking at, so that a friend "
        "can guess the species. Write 3-5 natural sentences in plain English. "
        "Emphasize the most important features first and mention the absent ones "
        "as 'it lacks ...'. Do NOT name the species, do NOT use any logic or "
        "math jargon, do NOT mention weights or numbers. End by asking 'What "
        "species is it?'.\n\n" + feats)


def _join(xs):
    xs = [x for x in xs if x]
    if not xs:
        return ""
    if len(xs) == 1:
        return xs[0]
    return ", ".join(xs[:-1]) + " and " + xs[-1]


def _format_guess(strong, usual, nots):
    """Assemble the species-agnostic guess text with calibrated reasoning guidance.

    The guidance (not the answer) tells the guesser HOW to use the tiers: the GL
    importance ordering means the primary marks are the most diagnostic and
    should dominate, while secondary marks are weak/noisy (coarse traits like
    size can even be mislabeled) and must not be used to exclude a candidate.
    """
    lines = [
        "Guess the bird species from this field description of a single bird.",
        "",
        "How to weigh the evidence (ordered by how much it drives the identity):",
        "- PRIMARY marks are the most diagnostic and reliable \u2014 let these drive "
        "your answer.",
        "- SECONDARY marks are weaker and may be noisy (coarse traits such as size "
        "or bill length are often approximate or mislabeled); use them only as "
        "tie-breakers, and do NOT rule out a species on a single secondary mark.",
        "- ABSENT marks are soft exclusions, not hard rules.",
        "",
    ]
    if strong:
        lines.append(f"PRIMARY marks: {_join(strong)}.")
    if usual:
        lines.append(f"SECONDARY marks: {_join(usual)}.")
    if nots:
        lines.append(f"Usually ABSENT: {_join(nots)}.")
    lines += [
        "",
        "Give your single best guess (a specific species), then 2\u20133 ranked "
        "alternates. What species is it?",
    ]
    return "\n".join(lines)


@__import__("torch").no_grad()
def gl_global_importance(model, h, baseline, device):
    """GL/LSP global importance of each concept for species ``h`` at ``baseline``.

    Importance = the drop in the tree's root truth when a literal is removed
    (set to the value that stops it supporting the score), evaluated through the
    ACTUAL frozen graded-logic tree -- so it accounts for every node's andness,
    the child weights, negation, and DAG sharing, not just a path-weight product.
    A mandatory input in a hard conjunction scores high; a high-local-weight
    alternative inside a soft disjunction (that barely moves the root) scores low.

    Returns (importance[K] >= 0, negated[K] bool). ``negated`` says whether the
    literal is the concept's NEGATION (so its "support" state is absence).
    """
    import torch
    head = model.head
    K = head.input_size
    if head.use_negation and bool(head.transform_frozen):
        negated = (head.frozen_transform[h] < 0.5).cpu()
    else:
        negated = torch.zeros(K, dtype=torch.bool)
    base = baseline.detach().clone().float().cpu()
    X = base.unsqueeze(0).repeat(K + 1, 1)              # row 0 = baseline
    for k in range(K):
        # knock the literal to its NON-supporting state: identity -> 0, NOT -> 1
        X[k + 1, k] = 1.0 if bool(negated[k]) else 0.0
    out = head(X.to(device))[:, h].cpu()               # (K+1,)
    imp = (out[0] - out[1:]).clamp(min=0.0)            # >=0 contribution to score
    return imp, negated


@__import__("torch").no_grad()
def gl_shapley_importance(model, h, baseline, device, n_perm=200, seed=0):
    """Monte-Carlo Shapley importance of each concept through the frozen GL tree.

    Knockout (``gl_global_importance``) measures only ``f(N) - f(N\\{i})`` -- the
    marginal of removing ``i`` from the FULL coalition, one context. But GL
    operators are non-additive: the marginal effect of a literal depends on which
    other literals are present (a satisfied input in a hard conjunction is inert
    once a sibling is 0; an OR-alternative is redundant once a sibling fires). So
    single-knockout credit is context-dependent and breaks the aggregation.

    Shapley averages the marginal ``f(S+i) - f(S)`` over ALL coalitions ``S``
    (here Monte-Carlo over ``n_perm`` random feature orderings), which is the
    unique attribution that is order-invariant and interaction-aware. A feature
    is "present" at its baseline value and "absent" at its non-supporting state
    (0 for an identity literal, 1 for a negated one). Satisfies efficiency:
    ``sum_i phi_i == f(all present) - f(all absent)``.

    Returns (phi[K], negated[K] bool). ``phi`` may be negative (a literal that,
    on average, pushes the score DOWN).
    """
    import torch
    head = model.head
    K = head.input_size
    if head.use_negation and bool(head.transform_frozen):
        negated = (head.frozen_transform[h] < 0.5).cpu()
    else:
        negated = torch.zeros(K, dtype=torch.bool)
    present = baseline.detach().clone().float().cpu()          # feature "on" value
    absent = negated.float()                                   # non-supporting state
    g = torch.Generator().manual_seed(seed)
    phi = torch.zeros(K)
    steps = torch.arange(K + 1).unsqueeze(1)                   # [K+1,1]
    for _ in range(n_perm):
        perm = torch.randperm(K, generator=g)
        rank = torch.empty(K, dtype=torch.long)
        rank[perm] = torch.arange(K)                           # rank[k] = position of k
        # row t (0..K): features with rank < t are present, others absent
        mask = rank.unsqueeze(0) < steps                       # [K+1,K] bool
        X = torch.where(mask, present.unsqueeze(0), absent.unsqueeze(0))
        out = head(X.to(device))[:, h].cpu()                   # (K+1,)
        deltas = out[1:] - out[:-1]                            # marginal of perm[t-1]
        phi[perm] += deltas
    phi /= n_perm
    return phi, negated


def describe_by_importance(model, h, baseline, names, device,
                           topn=9, n_neg=4, eps=0.004, min_fire=0.5,
                           method="knockout", n_perm=200):
    """Species-agnostic guess text ranked by GL global importance (grounded).

    Ranking/tiering is by GL global importance (how much each literal drives the
    root score); a mark is only STATED if it also actually holds for the species
    (support truth >= ``min_fire``), so we never advertise an important-but-false
    literal (e.g. an OR alternative that rarely fires).
    """
    if method == "shapley":
        imp, negated = gl_shapley_importance(model, h, baseline, device, n_perm=n_perm)
        imp = imp.clamp(min=0.0)                        # rank by positive contribution
    else:
        imp, negated = gl_global_importance(model, h, baseline, device)
    order = sorted(range(len(imp)), key=lambda k: -float(imp[k]))
    pos, negs = [], []
    for k in order:
        s = float(imp[k])
        if s < eps:
            break
        support = float(1.0 - baseline[k]) if bool(negated[k]) else float(baseline[k])
        if support < min_fire:
            continue                                   # important but not true -> skip
        phrase, _ = _feat_phrase(names[k])
        if not phrase.strip():
            continue
        if bool(negated[k]):
            negs.append((phrase, s))
        else:
            pos.append((phrase, s))
    pos, negs = pos[:topn], negs[:n_neg]

    strong = usual = []
    if pos:
        mx = pos[0][1]
        strong = [p for p, s in pos if s >= 0.5 * mx]
        usual = [p for p, s in pos if s < 0.5 * mx]
    nots = [p for p, _ in negs]
    return _format_guess(strong, usual, nots)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=DEFAULT_CKPT)
    ap.add_argument("--species", default="0,10,50",
                    help="comma list of class indices or name substrings")
    ap.add_argument("--max-children", type=int, default=3)
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--min-weight", type=float, default=0.03)
    ap.add_argument("--guess", action="store_true",
                    help="emit a clear-English, species-agnostic 'guess the bird' "
                         "description (no GL, no name) instead of the GL report")
    ap.add_argument("--ungrounded", action="store_true",
                    help="guess mode: rank by rule-weight only (skip the test-set "
                         "pass); by default the description is grounded in what "
                         "actually fires for the species' test images")
    ap.add_argument("--min-fire", type=float, default=0.5,
                    help="ungrounded guess: keep a literal only if its mean truth "
                         "for the species is at least this (positives) / absent")
    ap.add_argument("--min-importance", type=float, default=0.004,
                    help="grounded guess: min GL global-importance (root-score drop "
                         "when the literal is removed) to include a mark")
    ap.add_argument("--importance", choices=("knockout", "shapley"),
                    default="knockout",
                    help="grounded guess ranking: 'knockout' removes each literal "
                         "from the full set (one context); 'shapley' averages the "
                         "marginal over random coalitions (order/interaction-aware)")
    ap.add_argument("--n-perm", type=int, default=200,
                    help="shapley: number of Monte-Carlo feature permutations")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--call", action="store_true",
                    help="also call an installed LLM (anthropic/openai) if a key is set")
    ap.add_argument("--out-dir", default=os.path.join(_HERE, "results", "rules", "verbalize"))
    args = ap.parse_args()

    import torch
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(args.ckpt):
        raise SystemExit(f"checkpoint not found: {args.ckpt}")
    species_names = load_species_names()
    names112, *_ = _cub._load_attr_groups()
    heads = parse_species(args.species, species_names)
    model, *_ = load_model(args.ckpt, device)
    os.makedirs(args.out_dir, exist_ok=True)

    # Grounded guess: mean concept-truth per requested species over its test imgs.
    fired_by_head = {}
    if args.guess and not args.ungrounded:
        print("  grounding descriptions in test-set firing (one pass) ...")
        fired_by_head = class_mean_concepts(model, device, args.workers, heads)

    for i, h in enumerate(heads):
        node = extract_tree(model.head, h)
        slug = re.sub(r"[^A-Za-z0-9]+", "_", species_names[h]).strip("_")
        base = os.path.join(args.out_dir, f"species{h:03d}_{slug}")

        if args.guess:
            fired = fired_by_head.get(h)
            if fired is not None:
                text = describe_by_importance(model, h, fired, names112, device,
                                              n_neg=4, eps=args.min_importance,
                                              method=args.importance, n_perm=args.n_perm)
            else:
                text = build_guess_text(node, names112)      # --ungrounded fallback
            with open(base + "_guess.txt", "w", encoding="utf-8") as f:
                f.write(text + "\n")
            print(f"  wrote {base}_guess.txt")
            if i == 0:
                print("\n" + "-" * 78)
                print(f"PREVIEW  (species {h} hidden \u2013 this is the copy-paste text)")
                print("-" * 78)
                print(text)
            if args.call:
                resp = call_llm(build_guess_prompt(node, names112))
                if resp is None:
                    print("  [--call] no LLM SDK + API key found; wrote text only.")
                else:
                    with open(base + "_guess_llm.txt", "w", encoding="utf-8") as f:
                        f.write(resp)
                    print(f"  wrote {base}_guess_llm.txt  ({len(resp)} chars)")
            continue

        tree = build_json(node, names112,
                          args.max_children, args.max_depth, args.min_weight)
        prompt = build_prompt(species_names[h], tree)
        with open(base + "_prompt.md", "w", encoding="utf-8") as f:
            f.write(prompt)
        print(f"  wrote {base}_prompt.md")
        if i == 0:
            print("\n" + "-" * 78)
            print(f"PREVIEW  (species {h}: {species_names[h]})")
            print("-" * 78)
            print(prompt)
        if args.call:
            resp = call_llm(prompt)
            if resp is None:
                print("  [--call] no LLM SDK + API key found; wrote prompt only.")
            else:
                with open(base + "_report.md", "w", encoding="utf-8") as f:
                    f.write(resp)
                print(f"  wrote {base}_report.md  ({len(resp)} chars)")


if __name__ == "__main__":
    main()
