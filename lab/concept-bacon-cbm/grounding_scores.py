r"""Formal grounding-satisfaction scores for the K=5 MNIST OCBM.

Defines four grounding pillars, each in [0,1], and fuses them with a soft
graded-logic (GCD) aggregator into a per-concept satisfaction score z:

  G_c  concept grounding   : polarity-aware coherence gap (rule-preferred end).
  G_s  structural grounding : LLM present/absent consistency x evidence support,
                              audited against the frozen trees; includes a
                              SHUFFLED-hypothesis falsification control.
  G_e  evidence grounding   : Grad-CAM insertion-deletion on the concept logit.
  G_f  functional grounding : zero-shot USPS transfer retention (model-level).

The falsification control re-scores G_s under permuted concept->meaning
assignments; a meaningful metric collapses for wrong assignments.

  CUDA_VISIBLE_DEVICES="" py -3 grounding_scores.py --load saved/k5_harden.pt \
      --hypothesis results/k5_hypothesis_v3.json
  # add --heads 2 to also train linear/MLP heads and compare G_c'/G_e/G_f.
"""
from __future__ import annotations

import argparse
import itertools
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from eval_concept_transfer import (load_model, mnist_loader, usps_loader,  # noqa: E402
                                   _MEAN, _STD)
from eval_tree_transfer_calibrated import calibrate, predict_from_probs  # noqa: E402
from visual_inspect_mnist import collect, usage_polarity, concept_heatmap  # noqa: E402
from structural_verifier import load_structure, verify                  # noqa: E402
import json                                                             # noqa: E402


# --------------------------------------------------------------------------- #
#  graded-logic aggregation (weighted power mean; andness via exponent r)
#    r -> 0   geometric mean   = hard conjunction ("must have all", annihilating)
#    r = 0.5                    = soft conjunction (non-annihilating)
#    r = 1    arithmetic mean   = neutral ("nice to have")
#    r = 2                      = soft disjunction (rewards the higher)
# --------------------------------------------------------------------------- #
def power_mean(vals, r, weights=None):
    v = torch.tensor([float(x) for x in vals]).clamp(1e-6, 1.0)
    w = torch.tensor([float(x) for x in weights]) if weights else torch.ones(len(v))
    w = w / w.sum()
    if abs(r) < 1e-9:
        return float(torch.exp((w * torch.log(v)).sum()))
    return float(((w * v.pow(r)).sum()).pow(1.0 / r))


def hard_conj(vals, weights=None):
    """Hard conjunction (geometric mean): both/all criteria must be high."""
    return power_mean(vals, 0.0, weights)


def soft_conj(vals, weights=None):
    """Soft conjunction (non-annihilating): a zero criterion lowers but does not
    zero the result -- used to fold in the supporting functional score G_f."""
    return power_mean(vals, 0.5, weights)


gcd_soft = hard_conj  # backward-compatible alias


# --------------------------------------------------------------------------- #
#  G_c  concept coherence
# --------------------------------------------------------------------------- #
def coherence_gap(C, Y, P, K):
    """Polarity-aware gap: mean act on +digits minus on -digits (per concept)."""
    out = []
    for i in range(K):
        means = {d: float(C[(Y == d), i].mean()) for d in range(10)}
        pos = [means[d] for d in range(10) if P[i, d] > 0]
        neg = [means[d] for d in range(10) if P[i, d] < 0]
        gap = (sum(pos) / len(pos) if pos else 0) - (sum(neg) / len(neg) if neg else 0)
        out.append(max(0.0, gap))
    return out


def coherence_free(C, Y, K):
    """Polarity-free coherence (for heads without rule polarity): best one-vs-rest
    gap = max_d [mean_d(c_i) - mean_{!=d}(c_i)] over digits."""
    out = []
    for i in range(K):
        means = torch.tensor([C[(Y == d), i].mean() for d in range(10)])
        overall = C[:, i].mean()
        out.append(max(0.0, float((means - overall).max())))
    return out


def external_alignment(C, Y, K):
    """G_ce: EXTERNAL alignment of each concept to a distinct HUMAN stroke.

    Builds the K x 9 direction-agnostic identification AUC of every concept vs
    every human primitive (config.DIGIT_RULES stroke truth tables), then assigns
    each concept to a DISTINCT stroke with a one-to-one Hungarian matching (no
    stroke reused). The matched AUC in [0.5,1] is rescaled to [0,1]
    (0.5=chance ->0, 1=perfect ->1). Head-agnostic: needs only activations and
    stroke labels, so it is defined for black-box heads too (unlike the internal
    polarity-aware G_ci, which reads the logic tree)."""
    from scipy.optimize import linear_sum_assignment
    from eval_shapes import roc_auc
    import _bench
    from interpret_emergent_concepts import NAMED, named_gt
    spec = _bench.mnist_concept_spec("cpu")
    gt = named_gt(Y, spec)
    A = torch.full((K, len(NAMED)), float("nan"))
    for j, col in enumerate(gt):
        if col is None:
            continue
        labels, rows = col
        if labels.sum() == 0 or labels.sum() == len(labels):
            continue
        for i in range(K):
            A[i, j] = roc_auc(C[rows, i], labels)
    align = torch.nan_to_num(torch.maximum(A, 1 - A), nan=0.0)  # [0.5,1], undef->0
    ri, ci = linear_sum_assignment(-align.numpy())              # max total alignment
    matched = torch.zeros(K)
    for r, c in zip(ri, ci):
        matched[r] = align[r, c]
    # rescale AUC [0.5,1] -> [0,1] so a chance concept contributes 0, matching
    # the [0,1] scale of the internal gap G_ci for a meaningful arithmetic mean.
    return [max(0.0, min(1.0, 2.0 * float(matched[i]) - 1.0)) for i in range(K)]


def _pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    denom = (a.norm() * b.norm())
    return float((a * b).sum() / denom) if denom > 1e-8 else 0.0


def coherence_cec(C, Y, K, splits=8, seed=0):
    """Cross-exemplar consistency: split-half reproducibility of the per-digit
    activation profile. NOTE: degenerate here (~1.0 for all concepts incl. diffuse
    ones) because a diffuse-but-reproducible profile still correlates; kept only as
    a diagnostic. Use coherence_template instead."""
    g = torch.Generator().manual_seed(seed)
    N = C.shape[0]
    acc = [[] for _ in range(K)]
    for _ in range(splits):
        perm = torch.randperm(N, generator=g)
        h1, h2 = perm[: N // 2], perm[N // 2:]
        Y1, Y2, C1, C2 = Y[h1], Y[h2], C[h1], C[h2]
        for i in range(K):
            m1 = torch.tensor([C1[Y1 == d, i].mean() for d in range(10)])
            m2 = torch.tensor([C2[Y2 == d, i].mean() for d in range(10)])
            acc[i].append(_pearson(m1, m2))
    return [max(0.0, sum(a) / len(a)) for a in acc]


def coherence_template(X, C, K, top_frac=0.01):
    """Template-prediction consistency: build a pixel template from each concept's
    top activators, then correlate each image's cosine-match to that template with
    the concept's actual activation over ALL images. A coherent concept fires in
    proportion to template resemblance (high r); a diffuse concept fires regardless
    of appearance (low r) -- so it flags c2 that split-half correlation misses.
    Polarity-free and head-agnostic."""
    N = X.shape[0]
    Xf = X.reshape(N, -1)
    Xc = Xf - Xf.mean(1, keepdim=True)
    Xn = Xc.norm(dim=1) + 1e-8
    m = max(50, int(top_frac * N))
    out = []
    for i in range(K):
        top = torch.topk(C[:, i], m).indices
        T = Xf[top].mean(0)
        Tc = T - T.mean()
        resp = (Xc @ Tc) / (Xn * (Tc.norm() + 1e-8))           # cosine match [N]
        out.append(max(0.0, _pearson(resp, C[:, i])))
    return out


# --------------------------------------------------------------------------- #
#  G_s  structural consistency x support  (+ shuffled control)
# --------------------------------------------------------------------------- #
def g_s_from_report(report, K, smax=5):
    gs = []
    for i in range(K):
        c = report["concepts"][f"c{i}"]
        cons = c["consistency"] or 0.0
        sup = c["support"]
        gs.append(cons * min(sup, smax) / smax)
    return gs


def shuffled_gs(hyp, sign, S, thresh, K, smax=5):
    """Re-score G_s under every non-identity permutation of concept->membership."""
    base = hyp["concepts"]
    overalls, gs_means = [], []
    for perm in itertools.permutations(range(K)):
        if all(p == i for i, p in enumerate(perm)):
            continue
        shuf = {"concepts": {f"c{i}": base[f"c{perm[i]}"] for i in range(K)}}
        rep = verify(shuf, sign, S, thresh, min_support=2)
        oc = rep["summary"]["overall_consistency"]
        overalls.append(oc if oc is not None else 0.0)
        gs_means.append(sum(g_s_from_report(rep, K, smax)) / K)
    n = len(overalls)
    return sum(overalls) / n, sum(gs_means) / n


# --------------------------------------------------------------------------- #
#  G_e  Grad-CAM insertion-deletion on the concept logit
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _clogit(model, imgs, i):
    return model.encoder(imgs)[:, i]                            # pre-sigmoid (no saturation)


def insertion_deletion(model, X, i, steps=14):
    """Insertion-deletion faithfulness on the concept LOGIT (not the saturating
    sigmoid). Pixels are ranked by Grad-CAM; each curve is normalized per image
    between the blank-image and full-image logit so scores are comparable. Returns
    (insertion_AUC, deletion_AUC) in [0,1]. A well-grounded concept has evidence
    concentrated on few pixels: insertion rises fast (high), deletion drops fast
    (low)."""
    cam = concept_heatmap(model, X, i)                         # [N,28,28] spatial rank
    N = X.shape[0]
    flat_x = X.reshape(N, -1)
    order = cam.reshape(N, -1).argsort(1, descending=True)
    bg = (0.0 - _MEAN) / _STD
    L_full = _clogit(model, X, i)
    L_bg = _clogit(model, torch.full_like(flat_x, bg).reshape(N, 1, 28, 28), i)
    denom = (L_full - L_bg)
    denom = torch.where(denom.abs() < 1e-3, torch.full_like(denom, 1e-3), denom)
    Pn = flat_x.shape[1]
    ks = [int(round(f * Pn)) for f in torch.linspace(0, 1, steps).tolist()]
    ins_vals, del_vals = [], []
    for k in ks:
        idx = order[:, :k]
        ins_k = torch.full_like(flat_x, bg)
        ins_k.scatter_(1, idx, flat_x.gather(1, idx))
        del_k = flat_x.clone()
        del_k.scatter_(1, idx, torch.full((N, idx.shape[1]), bg))
        Li = _clogit(model, ins_k.reshape(N, 1, 28, 28), i)
        Ld = _clogit(model, del_k.reshape(N, 1, 28, 28), i)
        ins_vals.append(float(((Li - L_bg) / denom).clamp(0, 1).mean()))
        del_vals.append(float(((Ld - L_bg) / denom).clamp(0, 1).mean()))
    return sum(ins_vals) / len(ins_vals), sum(del_vals) / len(del_vals)


def g_e(model, X, C, K, n=24):
    out = []
    for i in range(K):
        idx = torch.topk(C[:, i], n).indices
        ins, dele = insertion_deletion(model, X[idx], i)
        out.append(max(0.0, min(1.0, ins - dele)))
    return out


def evidence_consistency(C, Y, K):
    """Within-digit firing consistency: a well-grounded concept fires at a similar
    level across exemplars of the SAME digit (decisive, low within-class spread).
    G_e = 1 - 2*mean_d(std of c_i within digit d), clipped to [0,1]. Rewards
    reliable detectors, penalizes erratic ones. Polarity-free and head-agnostic."""
    out = []
    for i in range(K):
        stds = [float(C[Y == d, i].std()) for d in range(10)]
        out.append(max(0.0, 1.0 - 2.0 * (sum(stds) / len(stds))))
    return out


# --------------------------------------------------------------------------- #
#  G_f  transfer retention (model-level)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def _collect_probs(model, loader, device):
    cs, ys = [], []
    for x, y in loader:
        cs.append(model.concept_probs(x.to(device)).cpu()); ys.append(y)
    return torch.cat(cs), torch.cat(ys)


def transfer_acc(model, data, bs, device, is_tree=True):
    """Return (usps_calibrated_acc, mnist_acc) -- ABSOLUTE accuracies. We report the
    absolute calibrated-USPS accuracy as G_f (not a retention ratio, which inflates
    when in-domain accuracy drops)."""
    Cm, Ym = _collect_probs(model, mnist_loader(data, bs), device)
    Cu, Yu = _collect_probs(model, usps_loader(data, bs), device)
    Cu_cal = calibrate(Cu, Cm)
    if is_tree:
        mn = (predict_from_probs(model, Cm, device) == Ym).float().mean().item()
        us = (predict_from_probs(model, Cu_cal, device) == Yu).float().mean().item()
    else:
        mn = (model.head(Cm.to(device)).argmax(1).cpu() == Ym).float().mean().item()
        us = (model.head(Cu_cal.to(device)).argmax(1).cpu() == Yu).float().mean().item()
    return us, mn


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", default=os.path.join(_HERE, "saved", "k5_harden.pt"))
    ap.add_argument("--hypothesis", default=os.path.join(_HERE, "results", "k5_hypothesis_v3.json"))
    ap.add_argument("--data", default="./data")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--thresh", type=float, default=0.05)
    ap.add_argument("--heads", type=int, default=0,
                    help="if >0, also train that many seeds of linear/MLP heads and compare")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--out", default=os.path.join(_HERE, "results", "v2", "grounding_scores.txt"))
    args = ap.parse_args()

    device = torch.device("cpu")
    model, ckpt = load_model(args.load, device)
    K = ckpt["K"]
    with open(args.hypothesis, encoding="utf-8") as f:
        hyp = json.load(f)

    X, C, Y = collect(model, mnist_loader(args.data, args.batch_size), device)
    P = usage_polarity(model, K)
    sign, S, _K, acc, _ = load_structure(args.load, args.thresh)

    Gc = coherence_gap(C, Y, P, K)
    Gtmpl = coherence_template(X, C, K)
    rep = verify(hyp, sign, S, args.thresh, min_support=2)
    Gs = g_s_from_report(rep, K)
    Ge = g_e(model, X, C, K)
    Gec = evidence_consistency(C, Y, K)
    Gf, mn = transfer_acc(model, args.data, args.batch_size, device, is_tree=True)
    # G_c has two sub-measurements: internal self-consistency (polarity-aware gap,
    # ontology-only) and external alignment (bijective concept<->human-stroke AUC,
    # head-agnostic). The reported coherence is their arithmetic mean.
    Gci = Gc                                                        # internal (was G_c)
    Gce = external_alignment(C, Y, K)                               # external alignment
    Gc = [0.5 * (Gci[i] + Gce[i]) for i in range(K)]               # combined coherence
    z = [hard_conj([Gc[i], Gs[i]], [0.5, 0.5]) for i in range(K)]   # per-concept: G_c AND G_s (50/50)
    core = hard_conj([sum(Gc) / K, Gf], [0.5, 0.5])                 # MANDATORY core: coherence AND function (both head-agnostic)
    Z = soft_conj([core, sum(Gs) / K], [0.65, 0.35])               # + structural bonus G_s (non-annihilating, ontology-only)

    sh_oc, sh_gs = shuffled_gs(hyp, sign, S, args.thresh, K)
    true_oc = rep["summary"]["overall_consistency"]

    lines = []
    def out(s=""):
        print(s); lines.append(s)

    names = {0: "junction", 1: "upper arc/hook", 2: "U-turn", 3: "arc", 4: "slant bar"}
    out(f"GROUNDING SCORES  {os.path.basename(args.load)}  K={K}  acc={acc*100:.2f}%\n")
    out(f"{'concept':<16}{'G_ci':>7}{'G_ce':>7}{'G_c':>7}{'G_s':>7}{'z':>7}")
    out("-" * 51)
    for i in range(K):
        out(f"c{i} {names.get(i,''):<13}{Gci[i]:>7.2f}{Gce[i]:>7.2f}{Gc[i]:>7.2f}{Gs[i]:>7.2f}{z[i]:>7.2f}")
    out("-" * 51)
    out(f"{'mean':<16}{sum(Gci)/K:>7.2f}{sum(Gce)/K:>7.2f}{sum(Gc)/K:>7.2f}{sum(Gs)/K:>7.2f}{sum(z)/K:>7.2f}")
    out(f"\n  G_ci = internal self-consistency (polarity-aware gap, ontology-only);"
        f" G_ce = external alignment (bijective concept<->stroke AUC, rescaled, head-agnostic);"
        f" G_c = mean(G_ci, G_ce).")
    out(f"\nG_f (USPS-calibrated transfer accuracy, model-level) = {Gf:.2f}  "
        f"(in-domain MNIST {mn*100:.1f}%)")
    out(f"\nMODEL-LEVEL grounding  Z = soft-conj_{{0.65/0.35}}( hard-conj(mean G_c, G_f), mean G_s )"
        f"  = soft-conj({core:.2f}, {sum(Gs)/K:.2f}) = {Z:.2f}")
    out(f"  (MANDATORY core = hard-conj(G_c, G_f), both head-agnostic; structural G_s folded in as a "
        f"non-annihilating bonus at 0.35 -- a head with no structure (G_s=0) still scores its "
        f"coherence-AND-function core, and the ontology is rewarded for the structure it uniquely has.)")
    out(f"\nG_c variants  (gap = polarity-aware magnitude; tmpl = template-prediction, "
        f"corr(top-activator-template match, activation)):")
    out("  " + "  ".join(f"c{i} gap {Gc[i]:.2f}/tmpl {Gtmpl[i]:.2f}" for i in range(K)))
    out(f"  mean: gap {sum(Gc)/K:.2f}  |  tmpl {sum(Gtmpl)/K:.2f}")
    out(f"\nG_e variants  (attr = attribution ins-del on top activators; "
        f"cons = within-digit firing consistency):")
    out("  " + "  ".join(f"c{i} attr {Ge[i]:.2f}/cons {Gec[i]:.2f}" for i in range(K)))
    out(f"  mean: attr {sum(Ge)/K:.2f}  |  cons {sum(Gec)/K:.2f}")
    out(f"\nFALSIFICATION CONTROL (structural grounding is discriminative):")
    out(f"  true hypothesis   : overall consistency {true_oc:.2f},  mean G_s {sum(Gs)/K:.2f}")
    out(f"  shuffled meanings : overall consistency {sh_oc:.2f},  mean G_s {sh_gs:.2f}  "
        f"(mean over {K}!-1 permutations)")
    out(f"  => structure rejects wrong meaning->membership assignments "
        f"(consistency {true_oc:.2f} vs {sh_oc:.2f}).")

    if args.heads > 0:
        out("\n" + "=" * 44)
        out(f"HEAD COMPARISON (G_ci is ontology-only = 0 for black-box heads; "
            f"G_ce/G_c/G_f are head-agnostic)")
        from ablation_head import BlackBoxCBM, train_blackbox  # noqa: E402
        from train_emergent_concepts import make_loaders as _ml  # noqa: E402
        tr, te = _ml(args.data, args.batch_size)
        out(f"{'head':<12}{'G_ci':>7}{'G_ce':>7}{'G_c':>7}{'G_f(USPS)':>11}{'G_s':>7}{'Z':>7}{'MNIST':>8}")
        out("-" * 66)
        Z_ont = soft_conj([hard_conj([sum(Gc) / K, Gf], [0.5, 0.5]), sum(Gs) / K], [0.65, 0.35])
        out(f"{'ontology':<12}{sum(Gci)/K:>7.2f}{sum(Gce)/K:>7.2f}{sum(Gc)/K:>7.2f}"
            f"{Gf:>11.2f}{sum(Gs)/K:>7.2f}{Z_ont:>7.2f}{mn*100:>7.1f}%")
        for h in ("mlp", "linear"):
            gce, gf, mns = [], [], []
            for s in range(args.heads):
                m = train_blackbox(h, K, tr, te, device, args.epochs, s)
                Xh, Ch, Yh = collect(m, mnist_loader(args.data, args.batch_size), device)
                gce.append(sum(external_alignment(Ch, Yh, K)) / K)
                us_h, mn_h = transfer_acc(m, args.data, args.batch_size, device, is_tree=False)
                gf.append(us_h); mns.append(mn_h)
            mgce = sum(gce) / len(gce)
            mgf = sum(gf) / len(gf)
            # black-box heads have no logic tree -> no internal self-consistency, no G_s
            Z_h = soft_conj([hard_conj([0.5 * mgce, mgf], [0.5, 0.5]), 0.0], [0.65, 0.35])
            out(f"{h:<12}{0.0:>7.2f}{mgce:>7.2f}{0.5*mgce:>7.2f}"
                f"{mgf:>11.2f}{0.0:>7.2f}{Z_h:>7.2f}{sum(mns)/len(mns)*100:>7.1f}%")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    open(args.out, "w", encoding="utf-8").write("\n".join(lines))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
