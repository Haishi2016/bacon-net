"""Zero-shot transfer of the WHOLE OBM (encoder + graded-logic trees) to USPS.

Concept-level transfer (eval_concept_transfer.py) showed individual concepts
transfer unevenly (c2 collapses). Here we transfer the entire learned symbolic
classifier -- encoder -> K concepts -> 10 graded-logic digit trees -> argmax --
to external USPS digits with NO retraining, and ask:

  * Does the composed model transfer better than its weakest concept, because
    each digit tree already routes through its most DISCRIMINATIVE concepts
    (e.g. the "4" tree uses c0/c4, not the flaky c2)?
  * Which digit TREES are robust vs fragile under the domain shift?

Reports overall MNIST vs USPS accuracy, per-digit recall + gap, and the top
confusion for each digit on USPS.

    python eval_tree_transfer.py --load saved/k5_harden.pt
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


def mnist_loader(root, bs):
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((_MEAN,), (_STD,))])
    ds = datasets.MNIST(root, train=False, download=True, transform=tf)
    return DataLoader(ds, batch_size=bs)


@torch.no_grad()
def predict(model, loader, device):
    preds, ys = [], []
    for x, y in loader:
        logits, _, _ = model(x.to(device))
        preds.append(logits.argmax(1).cpu())
        ys.append(y)
    return torch.cat(preds), torch.cat(ys)


def per_digit_recall(pred, y):
    return {d: (pred[y == d] == d).float().mean().item() for d in range(10)}


def per_digit_precision(pred, y):
    out = {}
    for d in range(10):
        sel = pred == d
        out[d] = (y[sel] == d).float().mean().item() if sel.any() else float("nan")
    return out


def top_confusion(pred, y, d):
    wrong = pred[(y == d) & (pred != d)]
    if len(wrong) == 0:
        return None, 0.0
    vals, counts = wrong.unique(return_counts=True)
    j = counts.argmax()
    return int(vals[j]), counts[j].item() / (y == d).sum().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", required=True)
    ap.add_argument("--data", default="./data")
    ap.add_argument("--batch-size", type=int, default=512)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ck = load_model(args.load, device)
    K = ck["K"]

    pm, ym = predict(model, mnist_loader(args.data, args.batch_size), device)
    pu, yu = predict(model, usps_loader(args.data, args.batch_size), device)
    acc_m = (pm == ym).float().mean().item()
    acc_u = (pu == yu).float().mean().item()

    print(f"\nWHOLE-TREE TRANSFER  {os.path.basename(args.load)}  K={K}")
    print(f"  MNIST test acc = {acc_m*100:.2f}%   (N={len(ym)})")
    print(f"  USPS  test acc = {acc_u*100:.2f}%   (N={len(yu)})   "
          f"zero-shot, NO retraining")

    rm, ru = per_digit_recall(pm, ym), per_digit_recall(pu, yu)
    prec_u = per_digit_precision(pu, yu)
    pred_share = {d: (pu == d).float().mean().item() for d in range(10)}
    print("\nper-digit USPS: recall (MNIST->USPS), precision, pred-share, top confusion:")
    print(f"  {'d':>2} {'M-rec':>6} {'U-rec':>6} {'U-prec':>7} {'pred%':>6}   confusion")
    for d in range(10):
        cd, frac = top_confusion(pu, yu, d)
        conf = f"-> {cd} ({frac*100:.0f}%)" if cd is not None else ""
        print(f"  {d:>2} {rm[d]*100:5.1f}% {ru[d]*100:5.1f}% {prec_u[d]*100:6.1f}% "
              f"{pred_share[d]*100:5.1f}%   {conf}")

    print(f"\n8-attractor check: true 8s are {(yu==8).float().mean()*100:.1f}% of USPS "
          f"but the model predicts 8 for {pred_share[8]*100:.1f}% of it "
          f"(precision {prec_u[8]*100:.1f}%).")
    gaps = sorted(range(10), key=lambda d: rm[d] - ru[d])
    print(f"genuinely robust trees (high recall AND precision): "
          f"{[d for d in range(10) if ru[d]>0.7 and prec_u[d]>0.7]}")
    print(f"digit-4 tree (uses c0/c4, NOT the flaky c2): USPS recall "
          f"{ru[4]*100:.1f}% prec {prec_u[4]*100:.1f}% (gap {(rm[4]-ru[4])*100:+.1f})")


if __name__ == "__main__":
    main()
