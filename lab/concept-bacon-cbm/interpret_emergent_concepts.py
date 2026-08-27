"""
Interpret the semantic meaning of EMERGENT concepts.

Trains the multi-tree emergent-concept model (unnamed concepts, 10 separate
BACON trees) and then decodes what each learned concept c_i *means*:

  1. Semantic naming -- identification AUC of each emerged concept against the 9
     human stroke concepts (loop_upper, vertical_line, ...), using the per-digit
     rule targets as ground truth (same machinery as eval_concept_identification).
     AUC>0.5 => "c_i behaves like <named>"; AUC<0.5 => "like NOT <named>".
  2. Digit signature -- which digits switch each emerged concept ON (mean>0.5),
     i.e. the concept read as a subset of digits, and the per-digit code (each
     digit as a binary word over the emerged concepts).

    python interpret_emergent_concepts.py --concepts 4 --epochs 15

The emerged concepts are unsupervised, so a single concept usually aligns with a
*combination* of human strokes (K<9 must compress); the AUC + digit-membership
together give its interpretable meaning.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, _REPO_ROOT)
sys.path.insert(0, os.path.join(_HERE, "table"))

import config as cfg                                            # noqa: E402
import shapes as shapes_mod                                     # noqa: E402
from eval_shapes import roc_auc                                 # noqa: E402
from train import make_loaders                                  # noqa: E402
from train_emergent_concepts import MultiTreeBaconCBM, train_one  # noqa: E402
import _bench                                                   # noqa: E402

NAMED = cfg.CONCEPTS                                            # 9 human strokes


@torch.no_grad()
def collect(model, loader, device):
    model.eval()
    cs, ys = [], []
    for x, y in loader:
        cs.append(model.concept_probs(x.to(device)).cpu())
        ys.append(y)
    return torch.cat(cs), torch.cat(ys)


def named_gt(y, spec):
    """For each named concept j: (labels in {0,1}, row-indices) over rule-masked digits."""
    ctgt, cmask = spec.ctgt.cpu(), spec.cmask.cpu()
    cols = []
    for j in range(len(NAMED)):
        rows = torch.nonzero(cmask[y, j] > 0.5, as_tuple=True)[0]
        cols.append((ctgt[y[rows], j].long(), rows) if len(rows) else None)
    return cols


def nmi_binary(pred_bin, tgt):
    a, b, n = pred_bin.long(), tgt.long(), pred_bin.numel()
    if n == 0:
        return float("nan")

    def H(v):
        p = v.float().mean().item()
        return -sum(q * math.log(q) for q in (p, 1 - p) if q > 0)
    Ha, Hb = H(a), H(b)
    if Ha == 0 or Hb == 0:
        return 0.0
    mi = 0.0
    for u in (0, 1):
        for v in (0, 1):
            puv = ((a == u) & (b == v)).float().mean().item()
            pu = (a == u).float().mean().item()
            pv = (b == v).float().mean().item()
            if puv > 0 and pu > 0 and pv > 0:
                mi += puv * math.log(puv / (pu * pv))
    return mi / math.sqrt(Ha * Hb)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concepts", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--weight-mode", type=str, default="trainable",
                    choices=["trainable", "fixed"])
    ap.add_argument("--save", type=str, default=None,
                    help="path to save the trained model (for reproducibility)")
    ap.add_argument("--load", type=str, default=None,
                    help="load a saved model instead of training")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--data", type=str,
                    default=os.path.join(_REPO_ROOT, "benchmarks", "mnist-addition", "data"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ld, test_ld = make_loaders(args.data, args.batch_size)
    if args.load:
        ckpt = torch.load(args.load, map_location=device, weights_only=False)
        K = ckpt["K"]
        model = MultiTreeBaconCBM(K, weight_mode=ckpt.get("weight_mode", "trainable"),
                                  no_negation=ckpt.get("no_negation", False),
                                  device=device).to(device)
        if ckpt.get("frozen"):
            model.prepare_frozen_structure()    # rebuild frozenInputToLeaf before load
            model.load_state_dict(ckpt["state_dict"])
        else:
            model.load_state_dict(ckpt["state_dict"])
            model.anneal(1.0)                   # restore sharp routing (temp not in state_dict)
        model.eval()
        print(f"loaded {args.load}  K={K}  weight_mode={ckpt.get('weight_mode')}  "
              f"{'HARD-FROZEN  ' if ckpt.get('frozen') else ''}"
              f"saved-acc={ckpt.get('acc', float('nan')) * 100:.2f}%\n")
    else:
        K = args.concepts
        print(f"Training emergent model: K={K} concepts, {args.epochs} epochs, "
              f"weights={args.weight_mode}...")
        acc, model, _ = train_one(K, train_ld, test_ld, device, epochs=args.epochs,
                                  seed=args.seed, weight_mode=args.weight_mode,
                                  verbose=False)
        print(f"test accuracy: {acc * 100:.2f}%\n")
        if args.save:
            torch.save({"state_dict": model.state_dict(), "K": K,
                        "weight_mode": args.weight_mode, "seed": args.seed,
                        "acc": acc}, args.save)
            print(f"saved model -> {args.save}\n")

    c, y = collect(model, test_ld, device)                     # (N,K), (N,)
    spec = _bench.mnist_concept_spec("cpu")
    gt = named_gt(y, spec)

    # emerged (K) x named (9) identification AUC
    A = torch.full((K, len(NAMED)), float("nan"))
    for j, col in enumerate(gt):
        if col is None:
            continue
        labels, rows = col
        if labels.sum() == 0 or labels.sum() == len(labels):
            continue
        for i in range(K):
            A[i, j] = roc_auc(c[rows, i], labels)

    print("=" * 78)
    print("SEMANTIC NAMING  (identification AUC of each emerged concept vs the 9 "
          "human strokes)")
    print("=" * 78)
    for i in range(K):
        row = A[i]
        strength = torch.where(torch.isnan(row), torch.zeros_like(row),
                               (row - 0.5).abs())
        order = torch.argsort(strength, descending=True)
        parts = []
        for j in order[:3].tolist():
            if math.isnan(row[j].item()):
                continue
            auc = row[j].item()
            sign = "" if auc >= 0.5 else "NOT "
            parts.append(f"{sign}{NAMED[j]} (AUC {auc:.2f})")
        print(f"  c{i}:  " + " ,  ".join(parts))

    # per-digit signature (mean emerged activation) + binary code
    per = torch.zeros(10, K)
    n = torch.zeros(10)
    for d in range(10):
        m = y == d
        per[d] = c[m].mean(0)
        n[d] = m.sum()
    print("\n" + "=" * 78)
    print("DIGIT SIGNATURES  (mean emerged-concept activation; [x]=ON >0.5)")
    print("=" * 78)
    print("       " + "  ".join(f"c{i}" for i in range(K)))
    for d in range(10):
        vals = "  ".join(f"{per[d, i]:.2f}" for i in range(K))
        code = "".join("1" if per[d, i] > 0.5 else "0" for i in range(K))
        print(f"  {d}:   {vals}   -> code {code}")

    # which digits switch each concept ON (its extensional meaning)
    print("\n" + "=" * 78)
    print("EMERGED CONCEPT = subset of digits it fires on")
    print("=" * 78)
    for i in range(K):
        on = [str(d) for d in range(10) if per[d, i] > 0.5]
        off = [str(d) for d in range(10) if per[d, i] <= 0.5]
        nm = nmi_binary((c[:, i] > 0.5).long(),
                        torch.tensor([1 if per[yy, i] > 0.5 else 0 for yy in y]))
        print(f"  c{i}:  ON for digits {{{','.join(on)}}}   "
              f"OFF {{{','.join(off)}}}")

    # distinctness of the digit codes (are all 10 digits uniquely coded?)
    codes = ["".join("1" if per[d, i] > 0.5 else "0" for i in range(K))
             for d in range(10)]
    uniq = len(set(codes))
    print(f"\ndistinct digit codes: {uniq}/10  "
          f"({'all unique' if uniq == 10 else 'some collisions'})")

    probe_shapes(model, device, K)

    print("\n" + "=" * 78)
    print("LEARNED LOGIC TREES  (softmax weights: genuine graded aggregation; "
          "a in [-1,2]: 1=AND, 0=OR)")
    print("=" * 78)
    csample = c[:1000].to(device)
    for d in range(10):
        describe_tree(model.trees[d], d, K, csample)


@torch.no_grad()
def probe_shapes(model, device, K, n=400):
    """DIRECT visual probe: no digit-label GT.  Feed the trained model synthetic
    axis-aligned shapes and read each emerged concept's response, to test whether
    any concept is actually a 'roundness/circle' detector (as a human would hope).
    """
    SH = ["circle", "ellipse", "line", "cross", "corner", "vee", "zigzag",
          "square", "rectangle", "triangle"]
    imgs, names = shapes_mod.generate(n, seed=123, shapes=SH, rotate=False)
    c = model.concept_probs(imgs.to(device)).cpu()
    idx = {s: [i for i, nm in enumerate(names) if nm == s] for s in SH}

    print("\n" + "=" * 78)
    print("DIRECT SHAPE PROBE  (mean emerged-concept activation per shape; "
          "no digit labels)")
    print("=" * 78)
    print("  shape       " + "  ".join(f"c{i}" for i in range(K)))
    for s in SH:
        rows = idx[s]
        vals = "  ".join(f"{c[rows, i].mean():.2f}" for i in range(K))
        tag = " (round)" if s in ("circle", "ellipse") else ""
        print(f"  {s:10s}  {vals}{tag}")

    # roundness detection strength per concept: AUC(c_i, round vs non-round)
    round_lab = torch.tensor([1 if nm in ("circle", "ellipse") else 0
                              for nm in names])
    print("\n  roundness detection per concept  (AUC circle/ellipse vs rest; "
          "0.5 = none, >0.5 fires ON round, <0.5 fires on lines/edges):")
    aucs = []
    for i in range(K):
        auc = roc_auc(c[:, i], round_lab)
        aucs.append(auc)
        # which shape each concept prefers (argmax mean activation)
        pref = max(SH, key=lambda s: c[idx[s], i].mean().item())
        print(f"    c{i}:  round-AUC {auc:.2f}   prefers '{pref}'")
    best_i = max(range(K), key=lambda i: aucs[i])          # strongest POSITIVE
    line_i = min(range(K), key=lambda i: aucs[i])          # strongest anti-round
    print(f"\n  => most round-selective concept: c{best_i} (round-AUC "
          f"{aucs[best_i]:.2f}); most line/edge-selective: c{line_i} "
          f"(round-AUC {aucs[line_i]:.2f})")
    # does it fire on round SPECIFICALLY, or on any closed shape?
    closed = [c[idx[s], best_i].mean().item() for s in ("square", "triangle", "rectangle")]
    rnd = [c[idx[s], best_i].mean().item() for s in ("circle", "ellipse")]
    if aucs[best_i] < 0.7:
        print("     (weak: no concept cleanly detects circles -- roundness did "
              "NOT emerge as its own concept)")
    elif max(closed) > 0.5:
        print(f"     (NOTE: c{best_i} also fires on non-round closed shapes "
              f"(square/triangle/rect ~{max(closed):.2f}) -- it is a "
              f"'closed-shape vs thin-line' detector, not a pure circle concept)")


def _andness_label(a):
    if a >= 1.3:
        return "hard-AND"
    if a >= 0.7:
        return "AND"
    if a >= 0.55:
        return "soft-AND"
    if a > 0.45:
        return "MEAN"
    if a > 0.3:
        return "soft-OR"
    if a >= -0.3:
        return "OR"
    return "hard-OR"


def _norm2(raw):
    """Reproduce binaryTreeLogicNet's minmax weight normalization for a 2-vec."""
    w = raw.detach().float()
    lo, hi = w.min(), w.max()
    if (hi - lo).item() == 0:
        return [0.5, 0.5]
    w = (w - lo) / (hi - lo)
    s = w.sum()
    w = w / s if s.item() != 0 else torch.tensor([0.5, 0.5])
    return [round(v, 2) for v in w.tolist()]


@torch.no_grad()
def describe_tree(tree, digit, K, csample):
    """Decode one digit's learned left-fold BACON tree (softmax weights: genuine
    2-input graded aggregation).  Shows routing, transforms, per-node andness +
    soft weights, and verifies concept usage with per-concept gradient sensitivity.
    """
    il = tree.input_to_leaf
    if hasattr(il, "P_hard"):                                  # hard-frozen tree
        leaf_concept = il.P_hard.argmax(1).tolist()            # exact bijection
    else:
        P = il.sinkhorn(il.logits.detach(), temperature=float(il.temperature),
                        n_iters=il.sinkhorn_iters)             # (leaves, inputs)
        leaf_concept = P.argmax(1).tolist()
    tl = tree.transformation_layer
    tnames = [type(t).__name__.replace("Transformation", "") for t in tl.transformations]
    trans = tl.logits.argmax(1).tolist()

    def leaf_expr(j):
        cpt, t = f"c{leaf_concept[j]}", tnames[trans[j]]
        return cpt if t == "Identity" else (
            f"NOT {cpt}" if t == "Negation" else f"{t}({cpt})")

    print(f"\n  --- digit {digit} tree ---")
    print("    leaves L0..L{}: ".format(K - 1)
          + ", ".join(f"L{j}={leaf_expr(j)}" for j in range(K)))
    acc = "L0"
    for i in range(tree.num_layers):
        a = (torch.sigmoid(tree.biases[i]) * 3 - 1).item()
        w = torch.softmax(tree.weights[i].detach(), dim=0).tolist()   # [w_acc, w_leaf]
        acc = (f"{_andness_label(a)}(a={a:+.2f})[{acc} x{w[0]:.2f}, "
               f"L{i + 1} x{w[1]:.2f}]")
    print(f"    formula: {acc}")

    # empirical sensitivity: |d(tree output)/d c_i| averaged over samples
    with torch.enable_grad():
        cc = csample.clone().requires_grad_(True)
        g = torch.autograd.grad(tree(cc).sum(), cc)[0].abs().mean(0)
    g = g / (g.max() + 1e-9)
    used = [f"c{i}={g[i]:.2f}" for i in range(K) if g[i] >= 0.15]
    print(f"    concepts actually used (|grad|, norm): {'  '.join(used)}")






if __name__ == "__main__":
    main()
