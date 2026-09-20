r"""Structural verifier tool for LLM-assessed concept grounding.

CONTRACT (clean separation of duties):
  * The LLM (subject expert) supplies a HYPOTHESIS -- for each emergent concept, a
    human meaning AND, from world knowledge, the classes it should be PRESENT /
    ABSENT for. The LLM must NOT look at the model's internals to build this.
  * This TOOL supplies the FACTS -- it reads each class's frozen graded-logic tree
    and derives the signed structural influence of every concept, then AUDITS the
    LLM's predictions against it. The tool makes no semantic judgement; it only
    reports agreement, contradictions, and coverage, so the LLM can revise.

This is the falsifiable half of the loop: a meaning is corroborated when the
class-rules USE the concept with the polarity that meaning predicts, and refuted
when a rule uses it the opposite way.

  py -3 structural_verifier.py --load saved/k5_harden.pt --hypothesis hyp.json

Hypothesis JSON schema:
  {"concepts": {
     "c1": {"meaning": "loop", "present": [0,6,8,9], "absent": [1,7]},
     ...}}
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


def load_structure(ckpt, thresh):
    """Return (sign[K,C], S[K,C], K, acc, class_names) structural facts from the
    frozen model. Currently wired for the MNIST emergent OCBM (per-digit trees);
    any OCBM reduces to the same [concepts x classes] signed-influence matrix, so
    only this loader is model-specific -- verify() below is dataset-agnostic."""
    model, K, acc = load_model(ckpt, torch.device("cpu"))
    S = influence(model, K)                                     # [K, C]
    sign = torch.zeros_like(S)
    sign[S > thresh] = 1.0
    sign[S < -thresh] = -1.0
    class_names = [str(d) for d in range(S.shape[1])]           # digits 0-9
    return sign, S, K, acc, class_names


def verify(hyp, sign, S, thresh, min_support=2):
    """Audit each concept's predicted present/absent classes against structure.
    Adds an EVIDENCE-STRENGTH gate: a concept is only 'well-evidenced' if at least
    ``min_support`` classes structurally constrain it (non-silent), so a perfect
    consistency on 1 class is not mistaken for a solid grounding."""
    K, C = S.shape
    report = {"concepts": {}, "violations": [], "summary": {}}
    total_hit = total_vio = 0
    under = []
    for i in range(K):
        key = f"c{i}"
        h = hyp.get("concepts", {}).get(key, {})
        present = set(int(x) for x in h.get("present", []))
        absent = set(int(x) for x in h.get("absent", []))
        hits, vios, neutral = [], [], []
        for d in range(C):
            s = int(sign[i, d])
            claimed = "+" if d in present else ("-" if d in absent else "?")
            if claimed == "?":
                continue
            if s == 0:
                neutral.append(d)                              # model indifferent
            elif (claimed == "+" and s > 0) or (claimed == "-" and s < 0):
                hits.append(d)
            else:
                vios.append(d)
                report["violations"].append(
                    {"concept": key, "meaning": h.get("meaning", "?"),
                     "class": d, "claimed": claimed,
                     "structure": "+" if s > 0 else "-", "influence": round(float(S[i, d]), 3)})
        n = len(hits) + len(vios)                              # support = evidenced classes
        cons = (len(hits) / n) if n else float("nan")
        well = n >= min_support
        if not well:
            under.append(key)
        total_hit += len(hits); total_vio += len(vios)
        report["concepts"][key] = {
            "meaning": h.get("meaning", "?"),
            "consistency": None if n == 0 else round(cons, 3),
            "support": n, "well_evidenced": well,
            "hits": hits, "violations": vios, "neutral(model-silent)": neutral}
    denom = total_hit + total_vio
    if total_vio > 0:
        verdict = f"{total_vio} contradiction(s) -- revise"
    elif under:
        verdict = f"CONSISTENT but under-determined: {under} (support < {min_support})"
    else:
        verdict = "CONSISTENT & well-evidenced"
    report["summary"] = {
        "overall_consistency": round(total_hit / denom, 3) if denom else None,
        "n_hits": total_hit, "n_violations": total_vio,
        "under_determined": under, "min_support": min_support, "verdict": verdict}
    return report


def per_class_rules(sign, S, hyp, thresh):
    """Reconstruct each class's structural rule under the hypothesised meanings."""
    K, C = S.shape
    names = {f"c{i}": hyp.get("concepts", {}).get(f"c{i}", {}).get("meaning", f"c{i}")
             for i in range(K)}
    out = {}
    for d in range(C):
        terms = sorted([i for i in range(K) if sign[i, d] != 0],
                       key=lambda i: -abs(S[i, d]))
        parts = [f"{'' if sign[i, d] > 0 else 'NOT '}{names[f'c{i}']}" for i in terms]
        out[d] = " , ".join(parts) if parts else "(no dominant concept)"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", default=os.path.join(_HERE, "saved", "k5_harden.pt"))
    ap.add_argument("--hypothesis", required=True, help="path to hypothesis JSON")
    ap.add_argument("--thresh", type=float, default=0.05)
    ap.add_argument("--min-support", type=int, default=2,
                    help="min #classes that must constrain a concept to call it well-evidenced")
    ap.add_argument("--out", default=None, help="write JSON report here")
    args = ap.parse_args()

    with open(args.hypothesis, encoding="utf-8") as f:
        hyp = json.load(f)
    sign, S, K, acc, class_names = load_structure(args.load, args.thresh)
    report = verify(hyp, sign, S, args.thresh, min_support=args.min_support)
    rules = per_class_rules(sign, S, hyp, args.thresh)

    print(f"structural verifier  |  {os.path.basename(args.load)}  acc={acc*100:.2f}%\n")
    print("=" * 74)
    print("PER-CONCEPT AUDIT  (LLM prediction vs frozen structure)")
    print("=" * 74)
    for i in range(K):
        c = report["concepts"][f"c{i}"]
        flag = "" if c["well_evidenced"] else "  <-- UNDER-DETERMINED"
        print(f"  c{i} = {c['meaning']:<16} consistency={c['consistency']} "
              f"support={c['support']}{flag}")
        print(f"        hits={c['hits']} viol={c['violations']} "
              f"model-silent={c['neutral(model-silent)']}")

    if report["violations"]:
        print("\n  CONTRADICTIONS to resolve:")
        for v in report["violations"]:
            print(f"    - {v['concept']}({v['meaning']}): you claim class {v['class']} "
                  f"is '{v['claimed']}' but its rule uses it '{v['structure']}' "
                  f"(influence {v['influence']:+})")

    print("\n" + "=" * 70)
    print("RECONSTRUCTED CLASS RULES  (structure under your meanings)")
    print("=" * 70)
    for d in range(len(rules)):
        print(f"  class {d}:  {rules[d]}")

    s = report["summary"]
    print("\n" + "=" * 70)
    print(f"SUMMARY: overall_consistency={s['overall_consistency']}  "
          f"hits={s['n_hits']} violations={s['n_violations']}  ->  {s['verdict']}")
    print("=" * 70)

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump({"report": report, "reconstructed_rules": rules}, f, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
