r"""Structural observation packet for LLM-assessed concept grounding (K=5 MNIST).

For each digit tree, probe the FROZEN tree function directly: hold all concepts at
0.5 (neutral) and measure how the tree's output changes as concept c_i goes 0->1.
This signed influence is read purely from the learned STRUCTURE (topology, andness,
polarity, weights) -- no activations, no labels. The output is the puzzle an LLM
solves: 10 rules over 5 UNLABELED predicates c0..c4 (each tagged with the digit it
describes); infer what c0..c4 must mean so every rule reads as a coherent digit.

  py -3 ground_structure_mnist.py --load saved/k5_harden.pt
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "table"))

from auto_ground_mnist import load_model                        # noqa: E402


@torch.no_grad()
def influence(model, K):
    """S[K,10]: signed effect of c_i on digit d's tree output, probed at the
    neutral point (all other concepts = 0.5). >0 = c_i supports digit d."""
    S = torch.zeros(K, 10)
    base = torch.full((1, K), 0.5)
    for d in range(10):
        t = model.trees[d]
        for i in range(K):
            hi = base.clone(); hi[0, i] = 1.0
            lo = base.clone(); lo[0, i] = 0.0
            S[i, d] = float((t(hi) - t(lo)).reshape(-1)[0])
    return S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", default=os.path.join(_HERE, "saved", "k5_harden.pt"))
    ap.add_argument("--thresh", type=float, default=0.05)
    args = ap.parse_args()

    model, K, acc = load_model(args.load, torch.device("cpu"))
    print(f"loaded {os.path.basename(args.load)}  K={K}  acc={acc*100:.2f}%\n")
    S = influence(model, K)

    print("=" * 68)
    print("PER-DIGIT RULE  (signed concept influence, structure only)")
    print("  +c_i = digit needs c_i present;  -c_i = digit needs c_i absent")
    print("=" * 68)
    for d in range(10):
        order = sorted(range(K), key=lambda i: -abs(S[i, d]))
        parts = [f"{'+' if S[i, d] > 0 else '-'}c{i} ({S[i, d]:+.2f})"
                 for i in order if abs(S[i, d]) >= args.thresh]
        print(f"  digit {d}:  " + "   ".join(parts))

    print("\n" + "=" * 68)
    print("SIGNED USAGE MATRIX  (rows = concepts, cols = digits 0-9)")
    print("=" * 68)
    print("       " + "  ".join(str(d) for d in range(10)))
    for i in range(K):
        cells = []
        for d in range(10):
            v = S[i, d]
            cells.append(" +" if v > args.thresh else (" -" if v < -args.thresh else " ."))
        print(f"  c{i}:  " + " ".join(cells))
    print("\n(each c_i: which digits want it present [+] vs absent [-] vs "
          "indifferent [.])")


if __name__ == "__main__":
    main()
