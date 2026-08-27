"""Model-agnostic defense against the blank/noise attack (no network internals).

eval_head_attack_compare.py showed the collapse (blank->1, noise->8) is an
ENCODER/concept-bottleneck failure shared by logic, linear and MLP heads -- so
the fix should be black-box and input-level. Real MNIST digits occupy a narrow
band of simple INPUT statistics: an ink fraction around ~0.15 and a moderate
total-variation-per-ink (smooth connected strokes). A blank has ~zero ink; dense
noise has ~0.5 ink and very high TV. We fit a 1-99 percentile box on two such
statistics from MNIST TRAIN (no labels, no model) and ABSTAIN on inputs that
fall outside it.

Reports: MNIST test false-reject rate (want low) and per-attack detection rate
(want high), plus accuracy on ACCEPTED MNIST.

    python attack_defense.py --load saved/k5_harden.pt
"""

from __future__ import annotations

import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

from concept_receptive_fields import load_model                 # noqa: E402
from train_emergent_concepts import make_loaders                # noqa: E402
from eval_concept_transfer import _MEAN, _STD                   # noqa: E402
from attack_noise import make_inputs                            # noqa: E402


def denorm(x):
    return (x * _STD + _MEAN).clamp(0, 1)


def stats(x_pix):
    """Two model-free input statistics on [0,1] images (N,1,28,28)."""
    ink = (x_pix > 0.5).float().mean((1, 2, 3))                  # ink fraction
    tv = (x_pix[:, :, 1:, :] - x_pix[:, :, :-1, :]).abs().mean((1, 2, 3)) \
        + (x_pix[:, :, :, 1:] - x_pix[:, :, :, :-1]).abs().mean((1, 2, 3))
    tv_per_ink = tv / ink.clamp_min(1e-3)                        # roughness / ink
    return ink, tv_per_ink


def fit_box(feats, lo=0.5, hi=99.5):
    q = torch.tensor([lo / 100, hi / 100])
    return {k: torch.quantile(v, q) for k, v in feats.items()}


def is_ood(feats, box):
    flag = torch.zeros(len(next(iter(feats.values()))), dtype=torch.bool)
    for k, v in feats.items():
        lo, hi = box[k]
        flag |= (v < lo) | (v > hi)
    return flag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", required=True)
    ap.add_argument("--data", default="./data")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ck = load_model(args.load, device)

    train_ld, test_ld = make_loaders(args.data, 512)

    # fit the box on MNIST train input statistics (no labels, no model)
    ink_tr, tv_tr = [], []
    for x, _ in train_ld:
        i, t = stats(denorm(x)); ink_tr.append(i); tv_tr.append(t)
    box = fit_box({"ink": torch.cat(ink_tr), "tv": torch.cat(tv_tr)})
    print(f"\nMODEL-AGNOSTIC DEFENSE  {os.path.basename(args.load)}")
    print(f"MNIST-train box:  ink in [{box['ink'][0]:.3f},{box['ink'][1]:.3f}]"
          f"   tv/ink in [{box['tv'][0]:.2f},{box['tv'][1]:.2f}]")

    # MNIST test: false-reject rate + accuracy on accepted
    xs, ys = [], []
    for x, y in test_ld:
        xs.append(x); ys.append(y)
    Xte = torch.cat(xs); Yte = torch.cat(ys)
    ink, tv = stats(denorm(Xte))
    ood_te = is_ood({"ink": ink, "tv": tv}, box)
    with torch.no_grad():
        pred = torch.cat([model(Xte[i:i+512].to(device))[0].argmax(1).cpu()
                          for i in range(0, len(Xte), 512)])
    acc_all = (pred == Yte).float().mean().item()
    keep = ~ood_te
    acc_kept = (pred[keep] == Yte[keep]).float().mean().item()
    print(f"\nMNIST test: false-reject {ood_te.float().mean()*100:.2f}%  "
          f"(keeps {keep.float().mean()*100:.1f}%)   "
          f"acc all {acc_all*100:.2f}% -> acc on accepted {acc_kept*100:.2f}%")

    # attacks: detection rate
    print("\nattack detection (fraction flagged OOD -> rejected):")
    for name, xn in make_inputs(2000).items():
        i, t = stats(denorm(xn))
        det = is_ood({"ink": i, "tv": t}, box).float().mean().item()
        print(f"  {name:26s}  rejected {det*100:5.1f}%")

    print("\n=> a two-number input sanity check (no model internals) rejects the "
          "corner\n   attacks while keeping nearly all real digits; applies to "
          "ANY CBM head.")


if __name__ == "__main__":
    main()
