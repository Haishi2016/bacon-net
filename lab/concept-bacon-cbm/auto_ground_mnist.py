r"""LLM-in-the-loop AUTOMATIC concept grounding on the K=5 MNIST OCBM.

Prototype of the propose -> validate -> discard -> regenerate -> converge loop
(the objective, reproducible part is the VALIDATOR; the PROPOSER is enumerated
here but is where an LLM slots in via the domain + each concept's class-usage
signature):

  PROPOSE   round 1 = the 9 human stroke primitives (loop_upper, vertical_line,
            ...); round 2 (only for concepts nothing explained) = digit-FAMILY
            hypotheses drawn from the concept's top-activating classes.
  VALIDATE  each hypothesis is scored by consistency with the frozen model across
            ALL 10 digit classes: the direction-agnostic AUC of the concept's
            activation against the hypothesised stroke's per-digit truth table
            (spec.ctgt) -- i.e. "does this meaning hold for every class?".
  DISCARD   hypotheses below tau are dropped.
  REGENERATE concepts still unexplained trigger the next hypothesis round.
  CONVERGE  when every concept has an accepted grounding.

Also cross-checks each grounding against the model's LOGIC STRUCTURE: which digit
trees actually route the concept, and with what polarity (identity/negation).

Runs on CPU (keeps the GPU free). Compares the recovered groundings to the
paper's manual K=5 labels.

  py -3 auto_ground_mnist.py --load saved/k5_harden.pt
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys

import torch
from sklearn.metrics import roc_auc_score

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "table"))

import config as cfg                                            # noqa: E402
import _bench                                                   # noqa: E402
from train_emergent_concepts import MultiTreeBaconCBM, make_loaders  # noqa: E402

NAMED = cfg.CONCEPTS                                            # 9 human strokes
PAPER_LABELS = {0: "5/9 family", 1: "closed-loop", 2: "'4' (diffuse)",
                3: "'2' family", 4: "7/4 family"}
TAU = 0.85          # accept a stroke hypothesis at/above this alignment AUC


def load_model(path, device):
    ck = torch.load(path, map_location=device, weights_only=False)
    K = ck["K"]
    model = MultiTreeBaconCBM(K, weight_mode=ck.get("weight_mode", "trainable"),
                              no_negation=ck.get("no_negation", False), device=device)
    if ck.get("frozen"):
        model.prepare_frozen_structure()
        model.load_state_dict(ck["state_dict"])
    else:
        model.load_state_dict(ck["state_dict"])
        model.anneal(1.0)
    model.eval().to(device)
    return model, K, float(ck.get("acc", float("nan")))


@torch.no_grad()
def collect(model, loader, device):
    cs, ys = [], []
    for x, y in loader:
        cs.append(model.concept_probs(x.to(device)).cpu())
        ys.append(y)
    return torch.cat(cs), torch.cat(ys)


def build_truth_table():
    """T[10,9]: from config.DIGIT_RULES, stroke j is TRUE for digit d iff it
    appears as a positive literal (not under NOT) in d's definition. Complete
    over all 10 digits -- no spec masking, so AUCs are honest (full support)."""
    T = torch.zeros((10, len(NAMED)))
    for d, rule in cfg.DIGIT_RULES.items():
        for j, name in enumerate(NAMED):
            present = re.search(rf"\b{name}\b", rule) is not None
            negated = re.search(rf"NOT\s+{name}\b", rule) is not None
            T[d, j] = 1.0 if (present and not negated) else 0.0
    return T


def stroke_alignment(c, y, T, K):
    """A[K,9] direction-agnostic AUC of concept i vs stroke j across ALL 10
    digits; dir[K,9] = +1 aligns with stroke, -1 with its negation."""
    A = torch.full((K, len(NAMED)), float("nan"))
    D = torch.zeros((K, len(NAMED)))
    for j in range(len(NAMED)):
        lab = T[y, j].long().numpy()
        if lab.min() == lab.max():
            continue
        for i in range(K):
            raw = roc_auc_score(lab, c[:, i].numpy())
            A[i, j] = max(raw, 1.0 - raw)
            D[i, j] = 1.0 if raw >= 0.5 else -1.0
    return A, D


def best_single(A, Dir, i):
    j = int(torch.nan_to_num(A[i], nan=0.0).argmax())
    return j, float(A[i, j]), (Dir[i, j] < 0)


def build_proposer_prompt(i, Mdig, tried):
    """Ask an LLM for NEW atomic visual primitives that could explain concept i's
    per-digit activation profile (the tried atomic strokes did not match)."""
    prof = {d: round(float(Mdig[i, d]), 2) for d in range(10)}
    ranked = sorted(prof.items(), key=lambda kv: -kv[1])
    return (
        "You are grounding an UNSUPERVISED visual concept from an MNIST digit "
        "classifier. A concept is a single scalar in [0,1] per image (no internal "
        "structure). Concept c%d has this MEAN activation per digit class 0-9 "
        "(higher = fires more on that digit):\n%s\n\n"
        "These ATOMIC stroke primitives were already tried and did NOT match "
        "(AUC < 0.85): %s.\n\n"
        "Propose up to 5 NEW *atomic* visual primitives -- single, indivisible "
        "pen-stroke / shape features (e.g. 'open top', 'left cusp', 'descender'). "
        "Do NOT propose digit names, and do NOT combine primitives with AND/OR. "
        "For each, list the digits (0-9) that possess it.\n"
        "Return STRICT JSON only: "
        '{"primitives":[{"name":"...","digits":[..]}, ...]}'
        % (i, "\n".join(f"  digit {d}: {v}" for d, v in ranked), ", ".join(tried))
    )


def validate_primitive(prim, c, y, i):
    """Direction-agnostic AUC of concept i against a proposed primitive's per-digit
    membership. Returns (auc, negated) or None if degenerate."""
    digits = set(int(d) for d in prim.get("digits", []))
    if not digits or len(digits) == 10:
        return None
    T = torch.tensor([1.0 if d in digits else 0.0 for d in range(10)])
    lab = T[y].long().numpy()
    if lab.min() == lab.max():
        return None
    raw = roc_auc_score(lab, c[:, i].numpy())
    return max(raw, 1.0 - raw), (raw < 0.5)


def parse_primitives(text):
    """Extract the primitives list from an LLM reply (tolerates ```json fences)."""
    if not text:
        return []
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return []
    try:
        return json.loads(m.group(0)).get("primitives", [])
    except Exception:
        return []


def per_digit_mean(c, y, K):
    M = torch.zeros((K, 10))
    for d in range(10):
        m = y == d
        if m.any():
            M[:, d] = c[m].mean(0)
    return M


def logic_usage(model, K):
    """For each digit tree, which concept each leaf reads (bijection) and whether
    it is negated -> signed usage U[K,10] in {+1 identity, -1 negation, 0}."""
    U = torch.zeros((K, 10))
    for d, t in enumerate(model.trees):
        il = t.input_to_leaf
        try:
            if hasattr(il, "P_hard") and il.P_hard is not None:
                leaf_concept = il.P_hard.argmax(1).tolist()
            else:
                P = il.sinkhorn(il.logits, n_iters=il.sinkhorn_iters,
                                temperature=il.temperature)
                leaf_concept = P.argmax(1).tolist()
        except Exception:
            continue
        try:
            trans = t.transformation_layer.logits.argmax(1).tolist()  # 0=id,1=neg
        except Exception:
            trans = [0] * K
        for leaf, ci in enumerate(leaf_concept):
            sign = -1.0 if (leaf < len(trans) and trans[ci] == 1) else 1.0
            U[ci, d] = sign
    return U


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", default=os.path.join(_HERE, "saved", "k5_harden.pt"))
    ap.add_argument("--data", default=os.path.join(
        os.path.dirname(_HERE), "..", "benchmarks", "mnist-addition", "data"))
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--call", action="store_true",
                    help="for entangled concepts, ask an LLM to PROPOSE new atomic "
                         "primitives (anthropic/openai key), then validate by AUC")
    args = ap.parse_args()

    device = torch.device("cpu")
    model, K, acc = load_model(args.load, device)
    print(f"loaded {os.path.basename(args.load)}  K={K}  saved-acc={acc*100:.2f}%\n")
    _, test_ld = make_loaders(args.data, args.batch_size)
    c, y = collect(model, test_ld, device)
    T = build_truth_table()                                    # [10,9] unmasked

    A, Dir = stroke_alignment(c, y, T, K)                      # [K,9]
    Mdig = per_digit_mean(c, y, K)                             # [K,10]
    U = logic_usage(model, K)                                  # [K,10] signed

    print("=" * 72)
    print("AUTOMATIC GROUNDING LOOP  (propose -> validate -> accept / flag)")
    print("ATOMIC ONLY: each concept maps to ONE human stroke primitive, or none")
    print("=" * 72)
    groundings = {}
    # ---- propose atomic stroke primitives, validate across all 10 classes ----
    print("\n[propose] hypotheses = 9 atomic stroke primitives; [validate] cross-class AUC")
    for i in range(K):
        j, score, neg = best_single(A, Dir, i)
        if score >= TAU:
            name = ("NOT " if neg else "") + NAMED[j]
            groundings[i] = (name, score, "atomic")
            print(f"  c{i}: {name:<20} primitiveness AUC {score:.3f}  -> GROUNDED")
        else:
            # no atomic primitive explains it -> honestly unresolved (entangled).
            # (with --call, the LLM would PROPOSE a NEW atomic primitive here.)
            hint = ("NOT " if neg else "") + NAMED[j]
            groundings[i] = (f"UNRESOLVED (~{hint})", score, "entangled")
            print(f"  c{i}: no atomic stroke >= {TAU}  (nearest {hint} {score:.3f})  "
                  f"-> ENTANGLED / non-primitive")

    entangled = [i for i in range(K) if groundings[i][2] == "entangled"]
    # ---- regenerate: LLM proposes NEW atomic primitives for entangled concepts ----
    if entangled and args.call:
        from verbalize_bird_rules import call_llm                # noqa: E402
        print(f"\n[regenerate] LLM proposes new atomic primitives for c{entangled}")
        for i in entangled:
            prompt = build_proposer_prompt(i, Mdig, list(NAMED))
            reply = call_llm(prompt)
            if reply is None:
                print(f"  c{i}: [--call] no LLM SDK/key found -> prompt only "
                      f"(stays entangled). Prompt saved to results/.")
                os.makedirs("results", exist_ok=True)
                with open(f"results/auto_ground_c{i}_prompt.txt", "w",
                          encoding="utf-8") as f:
                    f.write(prompt)
                continue
            best = None
            for prim in parse_primitives(reply):
                v = validate_primitive(prim, c, y, i)
                if v is None:
                    continue
                auc, negd = v
                mark = "ACCEPT" if auc >= TAU else "reject"
                nm = ("NOT " if negd else "") + prim.get("name", "?")
                print(f"    c{i}: proposed {nm:<22} digits={prim.get('digits')} "
                      f"AUC {auc:.3f}  -> {mark}")
                if auc >= TAU and (best is None or auc > best[1]):
                    best = (nm, auc)
            if best is not None:
                groundings[i] = (best[0] + " (LLM)", best[1], "atomic")
                print(f"  c{i}: GROUNDED by LLM primitive '{best[0]}' AUC {best[1]:.3f}")
            else:
                print(f"  c{i}: no proposed primitive reached {TAU} -> stays ENTANGLED")
    elif entangled:
        print(f"\n[regenerate] {len(entangled)} entangled concept(s) c{entangled}; "
              f"re-run with --call (and an LLM key) to propose new atomic primitives.")

    n_prim = sum(1 for v in groundings.values() if v[2] == "atomic")
    # ---- converged report ----
    print("\n" + "=" * 72)
    print(f"CONVERGED: {n_prim}/{K} concepts ground to an atomic human stroke")
    print("=" * 72)
    print(f"{'concept':>7} {'atomic grounding':<26} {'AUC':>5} {'status':>10}  "
          f"{'top digits':<12} {'paper label':<16}")
    for i in range(K):
        name, score, kind = groundings[i]
        top = torch.topk(Mdig[i], 3).indices.tolist()
        paper = PAPER_LABELS.get(i, "?")
        print(f"c{i:>5}  {name:<26} {score:>5.2f} {kind:>10}  "
              f"{str(top):<12} {paper:<16}")

    print("\nlogic-structure cross-check (rule usage vs activation profile):")
    for i in range(K):
        pos = [d for d in range(10) if U[i, d] > 0]
        neg = [d for d in range(10) if U[i, d] < 0]
        hi = Mdig[i, pos].mean().item() if pos else float("nan")
        lo = Mdig[i, neg].mean().item() if neg else float("nan")
        print(f"  c{i}: mean act on +id digits {pos}={hi:.2f}  "
              f"vs -neg digits {neg}={lo:.2f}")


if __name__ == "__main__":
    main()
