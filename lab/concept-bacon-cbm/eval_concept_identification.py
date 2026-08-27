"""
Quantitative, theory-grounded measure of CONCEPT TRANSFER / IDENTIFICATION.

The "align-gap" heuristic is replaced by concept *identifiability* measures from
the disentanglement / nonlinear-ICA literature (Eastwood & Williams DCI;
Hyvarinen identifiability up to permutation).  For a concept encoder we ask, for
every named concept k, whether its predicted activation c_k actually tracks the
ground-truth presence t_k of that concept:

  * Identification AUC = ROC-AUC(c_k, t_k) = P(c_k(x+) > c_k(x-)) for a
    true-positive x+ and true-negative x-.  This is the Mann-Whitney U / Somers'
    D rank statistic: it is INVARIANT to any monotone re-scaling of the
    activation (the right invariance for "does this axis encode the concept",
    without assuming the activation is calibrated).  0.5 = chance, 1.0 = perfect.
  * Normalized MI = NMI(1[c_k>0.5], t_k): the information-theoretic complement
    (fraction of the concept's entropy captured).

Two views (both reported per model):
  1. NAMED identification  : mean_k AUC(c_k, t_k)  -- does axis k detect concept
     k under its OWN name (alignment to the human concept).
  2. BEST-PERMUTATION      : Hungarian assignment on the K x K AUC matrix (using
     max(AUC, 1-AUC), since identifiability is only ever up to permutation & sign)
     -- can the concepts be recovered at all, under optimal relabeling?
  3. IDENTITY RATE (DCI)   : fraction of concepts whose single best detector IS
     the correctly-named axis -- a disentanglement/completeness score.

If NAMED ~ BEST-PERM ~ high, the encoder identifies the concepts AND keeps their
identity (interpretable).  If BEST-PERM is high but NAMED is low, the concepts
exist but are scrambled across axes.  If BEST-PERM is low, the concepts are not
linearly present at all.

Domains:
  * MNIST test   -- in-distribution, ground truth = per-digit rule targets
                    (mnist_concept_spec ctgt/cmask): does the encoder LEARN the
                    named concepts (no concept labels were used in training)?
  * shapes (axis-aligned) -- zero-shot TRANSFER, ground truth = hand-specified
                    geometry table below.  Rotation is disabled so orientation
                    concepts (vertical/horizontal/diagonal) have a stable GT.

    python eval_concept_identification.py
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

import config as cfg                                            # noqa: E402
import shapes as shapes_mod                                     # noqa: E402
from eval_shapes import load_model, roc_auc                     # noqa: E402
from eval_cream_zeroshot import CREAMDigit, train_cream         # noqa: E402
from eval_zeroshot_compare import CBMDigit                      # noqa: E402
from train import make_loaders                                  # noqa: E402

CONCEPTS = cfg.CONCEPTS
K = len(CONCEPTS)
# concepts actually used by at least one digit rule (right_curve is reserved /
# never constrained, so no model can be expected to learn it -> excluded).
from bacon_logic import parse_formula, collect_concepts         # noqa: E402
_USED = set()
for f in cfg.DIGIT_RULES.values():
    _USED |= collect_concepts(parse_formula(f), set())
ACTIVE = [i for i, n in enumerate(CONCEPTS) if n in _USED]

# --------------------------------------------------------------------------- #
# ground-truth concept table for AXIS-ALIGNED shapes.
#   1 = concept present, 0 = absent, None = ambiguous (excluded).
# Rendering (shapes.py, rotate=False): circle/ellipse = round closed outline;
# line = vertical stroke; cross = vertical + horizontal-through-middle;
# square/rectangle = axis box (top+bottom bars, vertical sides; loops ambiguous);
# triangle = apex-top, horizontal base, two slanted sides.
# --------------------------------------------------------------------------- #
#            loopU loopL vert  hTop  hMid  hBot  lCurv rCurv diag
GT_SHAPES = {
    "circle":    [1,    1,    0,    0,    0,    0,    1,    None, 0],
    "ellipse":   [1,    1,    0,    0,    0,    0,    1,    None, 0],
    "line":      [0,    0,    1,    0,    0,    0,    0,    None, 0],
    "cross":     [0,    0,    1,    0,    1,    0,    0,    None, 0],
    "square":    [None, None, 1,    1,    0,    1,    0,    None, 0],
    "rectangle": [None, None, 1,    1,    0,    1,    0,    None, 0],
    "triangle":  [None, None, 0,    0,    0,    1,    0,    None, 1],
}


# --------------------------------------------------------------------------- #
def nmi_binary(pred_bin: torch.Tensor, tgt: torch.Tensor) -> float:
    """Normalized mutual information between two binary vectors (MI / sqrt(HxHy))."""
    a = pred_bin.long(); b = tgt.long(); n = a.numel()
    if n == 0:
        return float("nan")

    def H(v):
        p1 = v.float().mean().item()
        return -sum(p * math.log(p) for p in (p1, 1 - p1) if p > 0)
    Ha, Hb = H(a), H(b)
    if Ha == 0 or Hb == 0:
        return float("nan")
    mi = 0.0
    for u in (0, 1):
        for v in (0, 1):
            puv = ((a == u) & (b == v)).float().mean().item()
            pu = (a == u).float().mean().item()
            pv = (b == v).float().mean().item()
            if puv > 0 and pu > 0 and pv > 0:
                mi += puv * math.log(puv / (pu * pv))
    return mi / math.sqrt(Ha * Hb)


@torch.no_grad()
def concept_probs(model, x, kind):
    out = model(x)
    return out[1] if kind == "ocbm" else out[1]      # both: index 1 = concept probs


def auc_matrix(cprobs, gt_cols):
    """K x K matrix M[i,j] = AUC(pred_i, gt_j) over the samples valid for j.

    ``gt_cols[j]`` = (values in {0,1}, row-indices used) or None if concept j has
    no defined GT.  Returns (M, valid_js).
    """
    M = torch.full((K, K), float("nan"))
    valid = []
    for j, col in enumerate(gt_cols):
        if col is None:
            continue
        labels, rows = col
        if labels.sum() == 0 or labels.sum() == len(labels):
            continue
        valid.append(j)
        for i in range(K):
            M[i, j] = roc_auc(cprobs[rows, i].cpu(), labels)
    return M, valid


def summarize_model(tag, cprobs, gt_cols):
    M, valid = auc_matrix(cprobs, gt_cols)
    active = [j for j in valid if j in ACTIVE]
    named = [M[j, j].item() for j in active]
    # NMI (named) at 0.5 threshold
    nmis = []
    for j in active:
        labels, rows = gt_cols[j]
        nmis.append(nmi_binary((cprobs[rows, j] > 0.5).long().cpu(), labels))
    # best permutation over ACTIVE predicted axes vs ACTIVE true concepts
    strength = torch.stack([torch.stack([torch.maximum(M[i, j], 1 - M[i, j])
                                         for j in active]) for i in ACTIVE])  # (|ACTIVE|, |active|)
    matched, identity = _assign(strength, ACTIVE, active)
    mean_named = sum(named) / len(named)
    mean_nmi = sum(v for v in nmis if v == v) / max(sum(1 for v in nmis if v == v), 1)
    print(f"{tag:8s}  named-AUC {mean_named:5.3f}   named-NMI {mean_nmi:5.3f}   "
          f"best-perm-AUC {matched:5.3f}   identity-rate {identity:4.2f}")
    return M, active


def _assign(strength, active_pred, active_true):
    """Optimal 1:1 assignment (Hungarian if scipy, else greedy).  Returns
    (mean matched strength, identity rate = fraction matched to own name)."""
    S = strength.numpy()
    try:
        from scipy.optimize import linear_sum_assignment
        r, c = linear_sum_assignment(-S)
    except Exception:  # greedy fallback
        import numpy as np
        S2 = S.copy(); r, c = [], []
        for _ in range(min(S.shape)):
            i, j = divmod(int(np.nanargmax(S2)), S2.shape[1])
            r.append(i); c.append(j); S2[i, :] = -1; S2[:, j] = -1
        r, c = np.array(r), np.array(c)
    matched = float(S[r, c].mean())
    identity = float(sum(active_pred[i] == active_true[j] for i, j in zip(r, c)) / len(r))
    return matched, identity


# --------------------------------------------------------------------------- #
def build_gt_mnist(y, spec):
    """gt_cols for MNIST: concept j GT = ctgt[y,j] over images with cmask[y,j]=1."""
    ctgt, cmask = spec.ctgt.cpu(), spec.cmask.cpu()
    cols = []
    for j in range(K):
        m = cmask[y, j] > 0.5
        rows = torch.nonzero(m, as_tuple=True)[0]
        cols.append((ctgt[y[rows], j].long(), rows) if len(rows) else None)
    return cols


def build_gt_shapes(names):
    """gt_cols for shapes from GT_SHAPES table."""
    cols = []
    for j in range(K):
        labels, rows = [], []
        for i, nm in enumerate(names):
            v = GT_SHAPES[nm][j]
            if v is not None:
                labels.append(v); rows.append(i)
        cols.append((torch.tensor(labels), torch.tensor(rows)) if rows else None)
    return cols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--retrain", action="store_true")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data = os.path.join(_REPO_ROOT, "benchmarks", "mnist-addition", "data")
    tl, vl = make_loaders(data, 256)
    rules = {int(k): v for k, v in cfg.DIGIT_RULES.items()}

    ocbm = load_model(os.path.join(_HERE, "checkpoint.pt"), device); ocbm.eval()

    def get_model(build, path, tag):
        m = build(); p = os.path.join(_HERE, path)
        if os.path.exists(p) and not args.retrain:
            m.load_state_dict(torch.load(p, map_location=device)); m.to(device)
        else:
            print(f"training {tag}..."); train_cream(m, tl, device, args.epochs)
            torch.save(m.state_dict(), p)
        m.eval(); return m

    cbm = get_model(lambda: CBMDigit(cfg.CONCEPTS), "cbm_digit_zs.pt", "CBM")
    cream = get_model(lambda: CREAMDigit(cfg.CONCEPTS, rules, d_y=20),
                      "cream_digit_zs.pt", "CREAM")
    models = [("OCBM", ocbm, "ocbm"), ("CBM", cbm, "cbm"), ("CREAM", cream, "cream")]

    print(f"\nActive concepts (used by >=1 rule, {len(ACTIVE)}/{K}): "
          + ", ".join(CONCEPTS[i] for i in ACTIVE))

    # ---------- in-distribution: MNIST test ---------- #
    sys.path.insert(0, os.path.join(_HERE, "table"))
    import _bench
    spec = _bench.mnist_concept_spec("cpu")
    xs, ys, cps = [], [], {t: [] for t, _, _ in models}
    for x, y in vl:
        xs.append(x); ys.append(y)
        for tag, m, kind in models:
            cps[tag].append(concept_probs(m, x.to(device), kind).cpu())
    y_all = torch.cat(ys)
    gt = build_gt_mnist(y_all, spec)
    print("\n=== Concept identification on MNIST test (in-distribution) ===")
    for tag, _, _ in models:
        summarize_model(tag, torch.cat(cps[tag]), gt)

    # ---------- zero-shot transfer: axis-aligned shapes ---------- #
    imgs, names = shapes_mod.generate(args.n, seed=123,
                                      shapes=list(GT_SHAPES.keys()), rotate=False)
    imgs = imgs.to(device)
    gt = build_gt_shapes(names)
    print("\n=== Concept identification on shapes (zero-shot transfer) ===")
    for tag, m, kind in models:
        summarize_model(tag, concept_probs(m, imgs, kind), gt)

    print("\nnamed-AUC = mean_k AUC(c_k, t_k) [alignment to the human concept]; "
          "\nbest-perm-AUC = Hungarian over the K x K AUC matrix [identifiability "
          "up to relabeling]; \nidentity-rate = fraction of concepts whose best "
          "detector is its own named axis [DCI disentanglement].")


if __name__ == "__main__":
    main()
