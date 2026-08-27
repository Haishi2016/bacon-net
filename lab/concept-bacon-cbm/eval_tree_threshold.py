"""Per-tree decision thresholds for the MNIST->USPS whole-model transfer.

The whole model decides argmax_d T_d(c), where T_d in (0,1) is the truth of the
d-th digit tree.  A per-tree threshold tau_d (default 0.5) shifts each tree's
operating point.  In the argmax competition a threshold is a per-tree BIAS:

    pred = argmax_d [ logit(T_d) - logit(tau_d) ]           (tau_d=0.5 -> unchanged)

Raising tau_d suppresses an over-firing tree.  This is the OUTPUT-side analogue
of the (input-side) per-concept quantile calibration in
eval_tree_transfer_calibrated.py.  We compare, on the frozen K=5 MNIST OBM:

  raw                 : default 0.5 threshold (the 62.5% / 8-attractor result)
  priormatch (0-shot) : label-free biases so predicted class shares match the
                        MNIST class prior (no USPS labels)
  logit-qmatch (0-shot): label-free -- match each USPS per-tree logit distn onto
                        its MNIST reference distn (output-side quantile match)
  maxF1 (SUPERVISED)  : per-tree one-vs-rest max-F1 threshold, fit on a USPS
                        split and reported on the held-out half (few-label
                        adaptation upper bound -- NOT zero-shot)

and stacks the best zero-shot output-side fix on top of the concept fix.

    py -3 eval_tree_threshold.py --load saved/k5_harden.pt
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

from concept_receptive_fields import load_model                    # noqa: E402
from eval_concept_transfer import usps_loader, _MEAN, _STD         # noqa: E402
from eval_tree_transfer import per_digit_recall, per_digit_precision  # noqa: E402
from eval_tree_transfer_calibrated import calibrate                # noqa: E402


def mnist_loader(root, bs):
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((_MEAN,), (_STD,))])
    ds = datasets.MNIST(root, train=False, download=True, transform=tf)
    return DataLoader(ds, batch_size=bs)


@torch.no_grad()
def collect_concepts(model, loader, device):
    cs, ys = [], []
    for x, y in loader:
        cs.append(model.concept_probs(x.to(device)).cpu())
        ys.append(y)
    return torch.cat(cs), torch.cat(ys)


@torch.no_grad()
def tree_logits(model, C, device, bs=1024):
    """Return per-tree truth-logits L[:,d] = logit(T_d(c)) for all images."""
    outs = []
    for i in range(0, C.shape[0], bs):
        p = C[i:i + bs].to(device)
        t = torch.cat([tr(p) for tr in model.trees], dim=1).clamp(1e-6, 1 - 1e-6)
        outs.append((torch.log(t) - torch.log1p(-t)).cpu())
    return torch.cat(outs)                                          # [N,10]


def report(tag, pred, y):
    acc = (pred == y).float().mean().item()
    prec = per_digit_precision(pred, y)
    rec = per_digit_recall(pred, y)
    share8 = (pred == 8).float().mean().item()
    print(f"[{tag:>24}]  acc={acc*100:5.2f}%   "
          f"8-share={share8*100:4.1f}%  8-prec={prec[8]*100:4.1f}%   "
          f"d9-rec={rec[9]*100:4.1f}%  d3-rec={rec[3]*100:4.1f}%  d4-rec={rec[4]*100:4.1f}%")
    return acc


# ---------------------------------------------------------------- strategies
def bias_priormatch(L, target_prior, iters=500, lr=0.3):
    """Label-free logit adjustment: per-tree bias b so the SOFT predicted class
    marginal softmax(L+b).mean(0) matches target_prior.  Soft shares keep the
    update smooth/convergent (hard argmax bincount oscillates)."""
    b = torch.zeros(L.shape[1])
    logq = torch.log(target_prior.clamp_min(1e-6))
    for _ in range(iters):
        share = torch.softmax(L + b, dim=1).mean(0).clamp_min(1e-9)
        b = b + lr * (logq - torch.log(share))
    return b


def logit_qmatch(Lu, Lm):
    """Label-free: per-tree quantile-match USPS logits Lu onto MNIST ref Lm."""
    out = Lu.clone()
    for d in range(Lu.shape[1]):
        ref = Lm[:, d].sort().values
        ranks = Lu[:, d].argsort().argsort().float() / max(len(Lu) - 1, 1)
        idx = (ranks * (len(ref) - 1)).round().long()
        out[:, d] = ref[idx]
    return out


def thresholds_maxf1(L_fit, y_fit, n_grid=200):
    """SUPERVISED: per-tree one-vs-rest max-F1 threshold in logit space.

    Returns bias b_d = -t_d so the combined rule is argmax(L - t)."""
    t = torch.zeros(L_fit.shape[1])
    for d in range(L_fit.shape[1]):
        col = L_fit[:, d]
        pos = (y_fit == d)
        npos = pos.sum().item()
        if npos == 0:
            t[d] = col.max() + 1.0                                  # never fire
            continue
        cand = torch.quantile(col, torch.linspace(0.0, 1.0, n_grid))
        best_f1, best_t = -1.0, 0.0
        for c in cand:
            fire = col >= c
            tp = (fire & pos).sum().item()
            fp = (fire & ~pos).sum().item()
            fn = (~fire & pos).sum().item()
            f1 = tp / (tp + 0.5 * (fp + fn) + 1e-9)
            if f1 > best_f1:
                best_f1, best_t = f1, c.item()
        t[d] = best_t
    return -t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", required=True)
    ap.add_argument("--data", default="./data")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--device", default="cpu",
                    help="cpu (default, avoids contending with GPU jobs)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = args.device
    model, ck = load_model(args.load, device)
    K = ck["K"]

    Cm, ym = collect_concepts(model, mnist_loader(args.data, args.batch_size), device)
    Cu, yu = collect_concepts(model, usps_loader(args.data, args.batch_size), device)

    Lm = tree_logits(model, Cm, device)
    Lu = tree_logits(model, Cu, device)

    print(f"\nPER-TREE THRESHOLD TEST  {os.path.basename(args.load)}  K={K}  "
          f"MNIST N={len(ym)}  USPS N={len(yu)}\n")

    # --- reference: raw argmax (should reproduce the 62.5% transfer number) ---
    report("raw (tau=0.5)", Lu.argmax(1), yu)

    # --- zero-shot: prior matching (target = MNIST empirical class prior) ------
    prior = torch.bincount(ym, minlength=10).float()
    prior = prior / prior.sum()
    b_pm = bias_priormatch(Lu, prior)
    report("priormatch [0-shot]", (Lu + b_pm).argmax(1), yu)

    # --- zero-shot: output-side logit quantile match --------------------------
    Lu_qm = logit_qmatch(Lu, Lm)
    report("logit-qmatch [0-shot]", Lu_qm.argmax(1), yu)

    # --- supervised oracle: per-tree max-F1 on a held-out USPS split ----------
    n = len(yu)
    perm = torch.randperm(n)
    fit, tst = perm[: n // 2], perm[n // 2:]
    b_f1 = thresholds_maxf1(Lu[fit], yu[fit])
    report("maxF1 fit-half [sup]", (Lu[fit] + b_f1).argmax(1), yu[fit])
    report("maxF1 test-half [sup]", (Lu[tst] + b_f1).argmax(1), yu[tst])

    # --- reference input-side concept fix (from the paper) --------------------
    Cu_cal = calibrate(Cu, Cm)
    Lu_cc = tree_logits(model, Cu_cal, device)
    report("concept-qmatch [0-shot]", Lu_cc.argmax(1), yu)

    # --- stack: best zero-shot output-side fix ON TOP of the concept fix ------
    b_pm_cc = bias_priormatch(Lu_cc, prior)
    report("concept+priormatch", (Lu_cc + b_pm_cc).argmax(1), yu)

    print("\n(zero-shot = no USPS labels; [sup] uses USPS labels on the fit half only)")


if __name__ == "__main__":
    main()
