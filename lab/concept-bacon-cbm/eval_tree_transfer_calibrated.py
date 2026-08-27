"""Test the concept-inflation hypothesis behind the USPS '8-attractor'.

eval_tree_transfer.py showed the whole OBM collapses toward digit 8 on USPS
(8 predicted for 42% of images, precision 19.7%). Hypothesis: USPS's thicker/
blurrier strokes INFLATE every concept activation, so the 8-tree (round AND
junction = the highest-conjunction digit) over-fires.

Fix (unsupervised, keeps concepts in [0,1], NO USPS labels): per-concept QUANTILE
MATCHING -- remap each USPS concept's empirical distribution onto the MNIST
reference concept distribution. This removes any marginal shift/inflation while
preserving each image's relative concept ranking. If the 8-attractor dissolves
and accuracy jumps, concept inflation is confirmed as the mechanism.

    python eval_tree_transfer_calibrated.py --load saved/k5_harden.pt
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

from concept_receptive_fields import load_model                 # noqa: E402
from eval_concept_transfer import usps_loader, _MEAN, _STD      # noqa: E402
from eval_tree_transfer import (per_digit_recall,               # noqa: E402
                                per_digit_precision, top_confusion)


def mnist_loader(root, bs):
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((_MEAN,), (_STD,))])
    ds = datasets.MNIST(root, train=False, download=True, transform=tf)
    return DataLoader(ds, batch_size=bs)


@torch.no_grad()
def collect_probs(model, loader, device):
    cs, ys = [], []
    for x, y in loader:
        cs.append(model.concept_probs(x.to(device)).cpu())
        ys.append(y)
    return torch.cat(cs), torch.cat(ys)


@torch.no_grad()
def predict_from_probs(model, probs, device, bs=1024):
    """Run the graded-logic trees on given concept probabilities -> digit preds."""
    preds = []
    lt = model.log_temp.exp()
    for i in range(0, probs.shape[0], bs):
        p = probs[i:i + bs].to(device)
        truths = torch.cat([t(p) for t in model.trees], dim=1).clamp(1e-6, 1 - 1e-6)
        logits = lt * (torch.log(truths) - torch.log1p(-truths))
        preds.append(logits.argmax(1).cpu())
    return torch.cat(preds)


def quantile_match(u, ref):
    """Map values u onto the distribution of ref (per-concept, monotone)."""
    ref_sorted = ref.sort().values
    ranks = u.argsort().argsort().float() / max(len(u) - 1, 1)   # empirical CDF
    idx = (ranks * (len(ref_sorted) - 1)).round().long()
    return ref_sorted[idx]


def calibrate(Cu, Cm):
    """Per-concept quantile-match USPS concepts Cu onto MNIST reference Cm."""
    out = Cu.clone()
    for i in range(Cu.shape[1]):
        out[:, i] = quantile_match(Cu[:, i], Cm[:, i])
    return out


def report(tag, pred, y, ref_recall=None):
    acc = (pred == y).float().mean().item()
    rec, prec = per_digit_recall(pred, y), per_digit_precision(pred, y)
    share8 = (pred == 8).float().mean().item()
    print(f"\n[{tag}]  USPS acc = {acc*100:.2f}%   "
          f"8 pred-share = {share8*100:.1f}%  (8 precision {prec[8]*100:.1f}%)")
    print(f"  {'d':>2} {'recall':>7} {'prec':>7}")
    for d in range(10):
        print(f"  {d:>2} {rec[d]*100:6.1f}% {prec[d]*100:6.1f}%")
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", required=True)
    ap.add_argument("--data", default="./data")
    ap.add_argument("--batch-size", type=int, default=512)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ck = load_model(args.load, device)

    Cm, _ = collect_probs(model, mnist_loader(args.data, args.batch_size), device)
    Cu, yu = collect_probs(model, usps_loader(args.data, args.batch_size), device)

    print(f"\nCONCEPT-INFLATION TEST  {os.path.basename(args.load)}  K={ck['K']}")
    print("per-concept mean activation  (MNIST ref -> USPS raw):")
    for i in range(ck["K"]):
        print(f"  c{i}: {Cm[:,i].mean():.2f} -> {Cu[:,i].mean():.2f}"
              f"  (+{(Cu[:,i].mean()-Cm[:,i].mean()):.2f})")

    pred_raw = predict_from_probs(model, Cu, device)
    acc_raw = report("RAW USPS concepts", pred_raw, yu)

    Cu_cal = calibrate(Cu, Cm)
    pred_cal = predict_from_probs(model, Cu_cal, device)
    acc_cal = report("QUANTILE-CALIBRATED concepts", pred_cal, yu)

    print(f"\n=> calibration lifted USPS acc {acc_raw*100:.2f}% -> {acc_cal*100:.2f}% "
          f"({(acc_cal-acc_raw)*100:+.2f} pts).  8 pred-share "
          f"{(pred_raw==8).float().mean()*100:.1f}% -> {(pred_cal==8).float().mean()*100:.1f}%.")
    if acc_cal - acc_raw > 0.1:
        print("   Concept INFLATION confirmed as the 8-attractor mechanism.")


if __name__ == "__main__":
    main()
