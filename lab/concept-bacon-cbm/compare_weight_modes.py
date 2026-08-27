"""
Test the hypothesis: do trainable node weights cause concept ENTANGLEMENT that
fixed (equal) weights avoid?  Weight-based aggregation can trade one concept off
against another, so the encoder need not make concepts independent; fixing the
weights removes that freedom.

For each weight mode (trainable / fixed) and several seeds we train a K-concept
emergent model and measure:
  * ACC        : test accuracy.
  * corr       : mean |pairwise Pearson correlation| of the concept ACTIVATIONS
                 over the test set.  Independent (disentangled) concepts -> ~0;
                 entangled / redundant concepts -> high.
  * profile-cos: mean pairwise COSINE similarity of the concepts' shape-response
                 vectors (each concept's mean activation over the shape probe).
                 Distinct semantic axes -> low; redundant concepts -> high.

    python compare_weight_modes.py --concepts 3 --seeds 5 --epochs 20
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, _REPO_ROOT)

import shapes as shapes_mod                                     # noqa: E402
from train import make_loaders                                  # noqa: E402
from train_emergent_concepts import train_one                   # noqa: E402

SHAPES = ["circle", "ellipse", "line", "cross", "corner", "vee", "zigzag",
          "square", "rectangle", "triangle"]


@torch.no_grad()
def concept_acts(model, loader, device):
    cs = []
    for x, _ in loader:
        cs.append(model.concept_probs(x.to(device)).cpu())
    return torch.cat(cs)                                        # (N, K)


def mean_abs_corr(C):
    """Mean |off-diagonal Pearson correlation| of concept activations."""
    Cc = C - C.mean(0)
    Z = Cc / (Cc.std(0) + 1e-8)
    corr = (Z.T @ Z) / C.shape[0]                              # (K, K)
    K = C.shape[1]
    mask = ~torch.eye(K, dtype=torch.bool)
    return corr[mask].abs().mean().item()


@torch.no_grad()
def shape_profile_cos(model, device, n=400):
    """Mean pairwise cosine similarity of the concepts' shape-response vectors."""
    imgs, names = shapes_mod.generate(n, seed=123, shapes=SHAPES, rotate=False)
    c = model.concept_probs(imgs.to(device)).cpu()
    prof = torch.stack([c[[i for i, nm in enumerate(names) if nm == s]].mean(0)
                        for s in SHAPES])                       # (S, K)
    P = prof.T                                                  # (K, S) per concept
    Pn = P / (P.norm(dim=1, keepdim=True) + 1e-8)
    sim = Pn @ Pn.T
    K = P.shape[0]
    mask = ~torch.eye(K, dtype=torch.bool)
    return sim[mask].mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concepts", type=int, default=3)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--data", type=str,
                    default=os.path.join(_REPO_ROOT, "benchmarks", "mnist-addition", "data"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ld, test_ld = make_loaders(args.data, args.batch_size)
    K = args.concepts
    print(f"K={K}  seeds={args.seeds}  epochs={args.epochs}  device={device}\n")

    summary = {}
    for mode in ("trainable", "fixed"):
        accs, corrs, coss = [], [], []
        for s in range(args.seeds):
            acc, model, _ = train_one(K, train_ld, test_ld, device,
                                      epochs=args.epochs, seed=s,
                                      weight_mode=mode, verbose=False)
            C = concept_acts(model, test_ld, device)
            corr = mean_abs_corr(C)
            cos = shape_profile_cos(model, device)
            accs.append(acc); corrs.append(corr); coss.append(cos)
            print(f"  {mode:9s} seed {s}:  acc {acc * 100:.2f}   "
                  f"corr {corr:.3f}   profile-cos {cos:.3f}")
        summary[mode] = (accs, corrs, coss)
        print()

    def ms(v):
        return statistics.mean(v), (statistics.pstdev(v) if len(v) > 1 else 0.0)

    print("=" * 64)
    print(f"{'mode':10s} {'ACC':>16s} {'corr (entangl.)':>18s} {'profile-cos':>16s}")
    for mode in ("trainable", "fixed"):
        a, c, k = summary[mode]
        am, asd = ms(a); cm, csd = ms(c); km, ksd = ms(k)
        print(f"{mode:10s} {am * 100:6.2f}+/-{asd * 100:4.2f}   "
              f"{cm:6.3f}+/-{csd:4.3f}      {km:6.3f}+/-{ksd:4.3f}")
    print("\nlower corr / profile-cos = purer, less entangled concepts.")


if __name__ == "__main__":
    main()
