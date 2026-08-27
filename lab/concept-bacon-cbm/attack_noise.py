"""Adversarial probe: do negative-only digit detectors accept 'nothing'?

Some emergent digit trees fire on ABSENCE of concepts, e.g. the pruned
digit-1 rule is A[not c1, not c0] (not-loop AND not-junction) and digit-6 is
A[not c2, not c3].  A rule made only of negations is satisfied by an input that
activates NO concept -- so a blank or random-noise image (which is "not
anything") may fall through to the all-negative detector and be confidently
classified as that digit.

We feed several out-of-distribution inputs (blank background, uniform noise,
Gaussian noise, sparse salt, full ink) through the frozen OBM and report the
predicted-digit distribution, mean confidence, and mean concept activations.
If noise collapses onto 1/6/3, the negative-detector vulnerability is confirmed.

    python attack_noise.py --load saved/k5_harden.pt
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
from eval_concept_transfer import _MEAN, _STD                   # noqa: E402


def norm(x_pix):
    return (x_pix - _MEAN) / _STD


def make_inputs(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    z = torch.zeros(n, 1, 28, 28)
    return {
        "blank (all background)": norm(z),
        "uniform noise U(0,1)":   norm(torch.rand(n, 1, 28, 28, generator=g)),
        "gaussian pix~N(.13,.3)": norm(torch.randn(n, 1, 28, 28, generator=g)
                                       .mul(0.3).add(0.13).clamp(0, 1)),
        "gaussian raw N(0,1)":    torch.randn(n, 1, 28, 28, generator=g),
        "sparse salt (10% ink)":  norm(torch.bernoulli(
                                       torch.full((n, 1, 28, 28), 0.1), generator=g)),
        "full ink (all 1)":       norm(torch.ones(n, 1, 28, 28)),
    }


@torch.no_grad()
def run(model, x, device):
    logits, probs, _ = model(x.to(device))
    conf = torch.softmax(logits, 1).max(1).values
    return logits.argmax(1).cpu(), conf.cpu(), probs.cpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", required=True)
    ap.add_argument("--n", type=int, default=2000)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ck = load_model(args.load, device)
    K = ck["K"]

    print(f"\nNOISE ATTACK  {os.path.basename(args.load)}  K={K}  n={args.n}/type")
    print("If a negative-only detector (e.g. 1='not loop & not junction') is "
          "vulnerable,\nnoise -> that digit at high confidence.\n")

    for name, x in make_inputs(args.n).items():
        pred, conf, probs = run(model, x, device)
        hist = torch.bincount(pred, minlength=10).tolist()
        top = max(range(10), key=lambda d: hist[d])
        frac = hist[top] / len(pred)
        cmean = " ".join(f"c{i}={probs[:,i].mean():.2f}" for i in range(K))
        print(f"{name:26s} -> digit {top} ({frac*100:.0f}%, conf {conf.mean():.2f})"
              f"  | dist {hist}")
        print(f"{'':26s}    concept means: {cmean}")

    print("\n(dist = count of predictions per digit 0..9)")


if __name__ == "__main__":
    main()
