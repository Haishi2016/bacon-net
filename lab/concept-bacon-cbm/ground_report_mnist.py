r"""End-to-end MNIST concept-grounding report generator.

Runs the full flow on the frozen K=5 OCBM and emits three artifacts:
  1. a FINAL REPORT (grounded concept meanings + structural-verifier verdict with
     evidence strength),
  2. a SIMPLIFIED expression table (dominant grounded concepts per digit),
  3. a FULL expression table (the exact left-fold trees, grounded).

All expressions are read directly from the frozen trees (deterministic left-fold:
acc=L0; acc=agg_i(acc, L_{i+1})), so the tables are reproducible, not transcribed.

  py -3 ground_report_mnist.py --load saved/k5_harden.pt --hypothesis results/k5_hypothesis_v2.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "table"))

from auto_ground_mnist import load_model                        # noqa: E402
from ground_structure_mnist import influence                    # noqa: E402
from structural_verifier import verify                          # noqa: E402


def short(meaning):
    return meaning.split("/")[0].split()[0]


def leaf_parts(tree, K):
    il = tree.input_to_leaf
    if hasattr(il, "P_hard") and il.P_hard is not None:
        leaf_concept = il.P_hard.argmax(1).tolist()
    else:
        P = il.sinkhorn(il.logits.detach(), temperature=float(il.temperature),
                        n_iters=il.sinkhorn_iters)
        leaf_concept = P.argmax(1).tolist()
    trans = tree.transformation_layer.logits.argmax(1).tolist()   # 0=id, 1=neg
    return leaf_concept, trans


def layer_params(tree):
    """Return list of (andness, [w_acc, w_leaf]) per fold layer."""
    out = []
    for i in range(tree.num_layers):
        a = float((torch.sigmoid(tree.biases[i]) * 3 - 1).item())
        w = torch.softmax(tree.weights[i].detach(), dim=0).tolist()
        out.append((a, w))
    return out


def _leaf_tex(names, leaf_concept, trans, j):
    nm = names[leaf_concept[j]]
    return rf"\overline{{{nm}}}" if trans[j] == 1 else nm


def full_expr(tree, names, K):
    lc, tr = leaf_parts(tree, K)
    lp = layer_params(tree)
    acc = _leaf_tex(names, lc, tr, 0)
    for i, (a, w) in enumerate(lp):
        op = r"\wedge" if a > 0.5 else r"\vee"
        acc = (rf"{w[0]:.2f}({acc}){op}_{{{a:+.2f}}}"
               rf"{w[1]:.2f}\,{_leaf_tex(names, lc, tr, i + 1)}")
    return acc


def path_weights(tree, K):
    """Total contribution weight of each leaf in the left fold."""
    lp = layer_params(tree)
    pw = [0.0] * K
    # L0 rides the acc side of every layer
    acc_w = 1.0
    w0s = [w[0] for _a, w in lp]
    w1s = [w[1] for _a, w in lp]
    # leaf L_{i+1} enters with w1[i], then multiplied by w0[k] for k>i
    for j in range(K):
        if j == 0:
            pw[0] = 1.0
            for k in range(len(lp)):
                pw[0] *= w0s[k]
        else:
            i = j - 1
            wj = w1s[i]
            for k in range(i + 1, len(lp)):
                wj *= w0s[k]
            pw[j] = wj
    return pw


def build_nested(tree, K):
    """The left fold as an explicit nested tree: root = agg_{K-2}(acc, L_{K-1})."""
    lc, tr = leaf_parts(tree, K)
    lp = layer_params(tree)
    node = ("leaf", 0)
    for i, (a, w) in enumerate(lp):
        node = ("agg", a, w[0], node, w[1], ("leaf", i + 1))
    return node, lc, tr


def _prune_node(node, keep):
    """Drop leaves not in ``keep`` and collapse single-child aggregations."""
    if node[0] == "leaf":
        return node if node[1] in keep else None
    _, a, w0, L, w1, R = node
    pL, pR = _prune_node(L, keep), _prune_node(R, keep)
    if pL is None and pR is None:
        return None
    if pL is None:
        return pR
    if pR is None:
        return pL
    return ("agg", a, w0, pL, w1, pR)


def _render(node, names, lc, tr):
    if node[0] == "leaf":
        return _leaf_tex(names, lc, tr, node[1])
    _, a, w0, L, w1, R = node
    op = r"\wedge" if a > 0.5 else r"\vee"

    def wrap(ch, w):
        s = _render(ch, names, lc, tr)
        if ch[0] == "agg":
            s = f"({s})"
        return rf"{w:.2f}\,{s}"
    return rf"{wrap(L, w0)}\mathbin{{{op}}}_{{{a:+.2f}}}{wrap(R, w1)}"


def simple_expr(tree, names, K, S_col, tau=0.02):
    """Faithful reduction: show the FULL tree, dropping ONLY leaves whose total
    path-weight is below ``tau`` (genuinely unused). With K=5 this keeps every
    substantively-used concept and never invents a summary."""
    node, lc, tr = build_nested(tree, K)
    pw = path_weights(tree, K)
    keep = {j for j in range(K) if pw[j] >= tau}
    pruned = _prune_node(node, keep) or node
    return _render(pruned, names, lc, tr)


def latex_table(rows, caption, label, sizecmd="footnotesize"):
    body = "\n".join(
        rf"{d} & ${expr}$ & {verb} \\[3pt]" for d, expr, verb in rows)
    return (rf"""\begin{{table*}}[t]
\caption{{{caption}}}
\label{{{label}}}
\begin{{center}}\{sizecmd}
\begin{{tabular}}{{c l l}}
{{\bf DIGIT}} & {{\bf GRADED-LOGIC EXPRESSION}} & {{\bf VERBALIZATION}} \\ \hline \\
{body}
\end{{tabular}}
\end{{center}}
\end{{table*}}""")


# Honest concept labels (verified via structural + polarity-aware visual inspection).
# Kept OUT of the table body; shown only in the caption legend so the rules stay
# neutral symbols c0-c4.
CONCEPT_DEFS = {
    0: "junction",
    1: "upper rounded arc / hook",
    2: "U-turn (varied direction)",
    3: "arc",
    4: "slant bar",
}

# Verbalizations reference concepts ONLY as symbols $c_i$; the parenthetical is the
# holistic digit-level intuition.
VERB_FULL = {
    0: "$c_1$ or $c_3$ (round top)",
    1: "$\\neg c_1$ \\textbf{AND} $\\neg c_0$ (bare stroke)",
    2: "$c_3$ with $c_4$ (no $c_0$), or $c_3$",
    3: "no $c_2$, $c_4$, or $c_1$ (all curves)",
    4: "$c_0$ \\textbf{AND} $c_4$ (a crossing)",
    5: "$c_0$, not $c_4$",
    6: "no $c_2$, no $c_3$ (bottom loop)",
    7: "$c_4$ \\textbf{AND} $c_0$ (angular)",
    8: "$c_0$ \\textbf{AND} $c_1$, with $c_2$ (stacked loops)",
    9: "$c_1$ with $c_3$, $c_4$ \\textbf{AND} $c_0$ (all)",
}


def concept_legend(K):
    return ("Concepts: "
            + "; ".join(rf"$c_{{{i}}}$ = {CONCEPT_DEFS[i]}" for i in range(K))
            + ".")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", default=os.path.join(_HERE, "saved", "k5_harden.pt"))
    ap.add_argument("--hypothesis", default=os.path.join(_HERE, "results", "k5_hypothesis_v2.json"))
    ap.add_argument("--min-support", type=int, default=2)
    ap.add_argument("--outdir", default=os.path.join(_HERE, "results", "v2"))
    args = ap.parse_args()

    model, K, acc = load_model(args.load, torch.device("cpu"))
    with open(args.hypothesis, encoding="utf-8") as f:
        hyp = json.load(f)

    # Table bodies use neutral symbols c0-c4 (definitions go in the caption legend).
    names = {i: rf"c_{{{i}}}" for i in range(K)}

    S = influence(model, K)
    sign = torch.zeros_like(S); sign[S > 0.05] = 1; sign[S < -0.05] = -1
    report = verify(hyp, sign, S, 0.05, min_support=args.min_support)

    # ---- FINAL REPORT ----
    lines = [f"# MNIST K=5 concept grounding — final report",
             f"model: {os.path.basename(args.load)}  |  frozen accuracy {acc*100:.2f}%\n",
             "## Grounded concepts (LLM meaning, verified against structure)\n",
             "| concept | meaning | consistency | support | status |",
             "|---|---|---|---|---|"]
    for i in range(K):
        c = report["concepts"][f"c{i}"]
        status = "well-evidenced" if c["well_evidenced"] else "**under-determined**"
        lines.append(f"| c{i} | {CONCEPT_DEFS[i]} | "
                     f"{c['consistency']} | {c['support']} | {status} |")
    s = report["summary"]
    lines += ["",
              f"**Verdict:** {s['verdict']}  ",
              f"overall consistency {s['overall_consistency']}, "
              f"{s['n_hits']} hits / {s['n_violations']} contradictions.",
              ""]
    report_md = "\n".join(lines)

    # ---- TABLES ----
    simp_rows = [(d, simple_expr(model.trees[d], names, K, S[:, d]), VERB_FULL[d]) for d in range(10)]
    full_rows = [(d, full_expr(model.trees[d], names, K), VERB_FULL[d]) for d in range(10)]
    legend = concept_legend(K)
    simp = latex_table(
        simp_rows,
        f"Emergent digit rules ($K{{=}}5$, hard-frozen; accuracy {acc*100:.2f}\\%). "
        f"The learned trees are shown faithfully; only concepts contributing under "
        f"2\\% of a digit's decision weight are omitted. $\\wedge_a$/$\\vee_a$ = "
        f"conj/disj with andness $a$; $\\overline{{\\cdot}}$ = negation. {legend}",
        "tab:digit-rules", "footnotesize")
    full = latex_table(
        full_rows,
        f"Full emergent digit rules ($K{{=}}5$, hard-frozen; accuracy {acc*100:.2f}\\%) --- "
        f"exact left-fold trees. $\\wedge_a$/$\\vee_a$ = conj/disj with andness $a$; "
        f"$\\overline{{\\cdot}}$ = negation. {legend}",
        "tab:digit-rules-full", "scriptsize")

    os.makedirs(args.outdir, exist_ok=True)
    open(os.path.join(args.outdir, "k5_final_report.md"), "w", encoding="utf-8").write(report_md)
    open(os.path.join(args.outdir, "k5_table_simplified.tex"), "w", encoding="utf-8").write(simp)
    open(os.path.join(args.outdir, "k5_table_full.tex"), "w", encoding="utf-8").write(full)

    print(report_md)
    print(f"\n----- SIMPLIFIED TABLE ({args.outdir}/k5_table_simplified.tex) -----\n")
    print(simp)
    print(f"\n----- FULL TABLE ({args.outdir}/k5_table_full.tex) -----\n")
    print(full)


if __name__ == "__main__":
    main()
