"""Concept redundancy / distinctness analysis for the emergent K-concept model.

Loads a (frozen) MultiTreeBaconCBM checkpoint, runs the MNIST test set through
the encoder, and reports how *distinct* the emerged concepts are:

  1. Pearson correlation matrix  R[i,j]  over the K concept activations.
  2. |R| off-diagonal summary (max / mean per concept).
  3. Multicollinearity: for each concept c_i, R^2 of predicting c_i from the
     other K-1 concepts (linear least squares) and its VIF = 1/(1-R^2).
     A concept with high R^2 (say > 0.8) is largely a linear blend of the
     others -> redundant.  A concept with low R^2 is carrying unique variance.

    python concept_correlations.py --load saved/k5_harden.pt

Interpretation guide:
  * High pairwise |R| + high VIF  => concepts overlap; K may be too LARGE
    (spare capacity spent duplicating a factor).
  * Low pairwise |R| but one concept is a *nonlinear* mix (fires as A and B
    but never predicted linearly) => the factor is genuinely entangled and
    more capacity (larger K) could let it split.  Correlation alone can't see
    this; we also report the sign-split (fraction of mass shared with each
    partner) to disambiguate.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, _REPO_ROOT)

from train_emergent_concepts import MultiTreeBaconCBM, make_loaders  # noqa: E402


def load_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    m = MultiTreeBaconCBM(ckpt["K"], weight_mode=ckpt.get("weight_mode", "trainable"),
                          no_negation=ckpt.get("no_negation", False),
                          device=device).to(device)
    if ckpt.get("frozen"):
        m.prepare_frozen_structure(); m.load_state_dict(ckpt["state_dict"])
    else:
        m.load_state_dict(ckpt["state_dict"]); m.anneal(1.0)
    m.eval()
    return m, ckpt


@torch.no_grad()
def collect_concepts(model, loader, device):
    model.eval()
    cs, ys = [], []
    for x, y in loader:
        cs.append(model.concept_probs(x.to(device)).cpu())
        ys.append(y)
    return torch.cat(cs), torch.cat(ys)


def corr_matrix(C):
    """Pearson correlation over columns of C (N,K)."""
    Cc = C - C.mean(0, keepdim=True)
    std = Cc.std(0, unbiased=False, keepdim=True).clamp_min(1e-8)
    Z = Cc / std
    return (Z.t() @ Z) / C.shape[0]


def collinearity(C):
    """For each column i: R^2 of least-squares predicting col i from the rest.
    Returns (r2, vif) tensors of length K."""
    N, K = C.shape
    Cc = C - C.mean(0, keepdim=True)
    r2 = torch.zeros(K)
    for i in range(K):
        y = Cc[:, i]
        idx = [j for j in range(K) if j != i]
        X = Cc[:, idx]
        # add intercept column (already centered, but keep for safety)
        sol, *_ = torch.linalg.lstsq(X, y.unsqueeze(1))
        pred = (X @ sol).squeeze(1)
        ss_res = ((y - pred) ** 2).sum()
        ss_tot = (y ** 2).sum().clamp_min(1e-12)
        r2[i] = (1 - ss_res / ss_tot).clamp(0, 1)
    vif = 1.0 / (1.0 - r2).clamp_min(1e-6)
    return r2, vif


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", required=True)
    ap.add_argument("--data", default="./data")
    ap.add_argument("--batch-size", type=int, default=512)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_model(args.load, device)
    K = ckpt["K"]
    _, test_ld = make_loaders(args.data, args.batch_size)
    C, y = collect_concepts(model, test_ld, device)  # (N,K)

    R = corr_matrix(C)
    r2, vif = collinearity(C)

    print(f"\nCONCEPT DISTINCTNESS  {os.path.basename(args.load)}  K={K}  N={C.shape[0]}")
    print(f"mean concept activation: " +
          "  ".join(f"c{i}={C[:,i].mean():.2f}" for i in range(K)))

    # correlation matrix
    print("\nPearson correlation R[i,j]:")
    head = "      " + "".join(f"  c{j:<4d}" for j in range(K))
    print(head)
    for i in range(K):
        row = "".join(f"{R[i,j]:+.2f} " for j in range(K))
        print(f"  c{i}  {row}")

    # off-diagonal summary
    absR = R.abs().clone()
    absR.fill_diagonal_(0.0)
    print("\nper-concept |R| to others   (max partner, mean):")
    for i in range(K):
        mx, arg = absR[i].max(0)
        print(f"  c{i}:  max |R|={mx:.2f} (with c{arg.item()})   mean |R|={absR[i].mean():.2f}")

    # multicollinearity
    print("\nmulticollinearity  (R^2 = predictability from the OTHER concepts):")
    for i in range(K):
        flag = "  <-- redundant" if r2[i] > 0.8 else ("  <-- unique" if r2[i] < 0.4 else "")
        print(f"  c{i}:  R^2={r2[i]:.2f}   VIF={vif[i]:6.2f}{flag}")

    # global verdict
    maxoff = absR.max().item()
    meanoff = absR.mean().item()
    print(f"\nsummary:  max off-diag |R|={maxoff:.2f}   mean off-diag |R|={meanoff:.2f}"
          f"   max R^2={r2.max():.2f}")
    if r2.max() > 0.8:
        print("  => at least one concept is a near-linear blend of the others (redundant).")
        print("     Redundancy argues K is too LARGE, not too small; try a SMALLER K.")
    elif maxoff > 0.6:
        print("  => concepts share substantial variance but none is fully predictable;")
        print("     partial overlap -- inspect the sign split before changing K.")
    else:
        print("  => concepts are largely distinct; K looks appropriate.")


if __name__ == "__main__":
    main()
