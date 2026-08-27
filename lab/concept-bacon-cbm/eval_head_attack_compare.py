"""Is the blank/noise attack OBM-specific, or does a standard CBM head share it?

Freezes the emergent K concept encoder and trains two conventional decision
heads on the SAME concepts -- a linear layer and a 1-hidden-layer MLP (i.e. a
standard CBM head) -- then runs the identical blank/noise attack on all three
heads (BACON logic tree vs linear vs MLP). Tests whether over-confident OOD
collapse is unique to the all-negative logic rules or a general concept-
bottleneck phenomenon.

    python eval_head_attack_compare.py --load saved/k5_harden.pt
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

from concept_receptive_fields import load_model                 # noqa: E402
from train_emergent_concepts import make_loaders                # noqa: E402
from attack_noise import make_inputs                            # noqa: E402


@torch.no_grad()
def encode(model, loader, device):
    cs, ys = [], []
    for x, y in loader:
        cs.append(model.concept_probs(x.to(device)).cpu())
        ys.append(y)
    return torch.cat(cs), torch.cat(ys)


def train_head(head, C, Y, device, epochs=8, lr=1e-2):
    head = head.to(device)
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    C, Y = C.to(device), Y.to(device)
    for _ in range(epochs):
        for i in range(0, len(C), 512):
            opt.zero_grad()
            loss = nn.functional.cross_entropy(head(C[i:i+512]), Y[i:i+512])
            loss.backward(); opt.step()
    head.eval()
    return head


@torch.no_grad()
def acc_of(head, C, Y, device):
    return (head(C.to(device)).argmax(1).cpu() == Y).float().mean().item()


@torch.no_grad()
def attack_head(predict_fn, device):
    rows = {}
    for name, x in make_inputs(2000).items():
        logits = predict_fn(x.to(device))
        conf = torch.softmax(logits, 1).max(1).values.mean().item()
        pred = logits.argmax(1).cpu()
        hist = torch.bincount(pred, minlength=10)
        top = int(hist.argmax()); frac = hist[top].item() / len(pred)
        rows[name] = (top, frac, conf)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", required=True)
    ap.add_argument("--data", default="./data")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ck = load_model(args.load, device)
    K = ck["K"]

    train_ld, test_ld = make_loaders(args.data, 512)
    Ctr, Ytr = encode(model, train_ld, device)
    Cte, Yte = encode(model, test_ld, device)

    linear = train_head(nn.Linear(K, 10), Ctr, Ytr, device)
    mlp = train_head(nn.Sequential(nn.Linear(K, 64), nn.ReLU(), nn.Linear(64, 10)),
                     Ctr, Ytr, device)

    @torch.no_grad()
    def logic_predict(x):
        return model(x)[0]

    def linear_predict(x):
        return linear(model.concept_probs(x))

    def mlp_predict(x):
        return mlp(model.concept_probs(x))

    # clean test accuracy per head
    acc_logic = (torch.cat([model(x.to(device))[0].argmax(1).cpu()
                            for x, _ in test_ld]) == Yte).float().mean().item()
    acc_lin = acc_of(linear, Cte, Yte, device)
    acc_mlp = acc_of(mlp, Cte, Yte, device)

    print(f"\nHEAD-ATTACK COMPARISON  {os.path.basename(args.load)}  K={K}")
    print(f"clean MNIST test acc:  logic {acc_logic*100:.2f}%   "
          f"linear {acc_lin*100:.2f}%   MLP {acc_mlp*100:.2f}%")

    heads = {"BACON logic": logic_predict, "linear (CBM)": linear_predict,
             "MLP (CBM)": mlp_predict}
    results = {name: attack_head(fn, device) for name, fn in heads.items()}

    inputs = list(next(iter(results.values())).keys())
    print("\nattack -> (predicted digit, %, mean confidence) per head:")
    print(f"  {'input':26s} {'BACON logic':>18} {'linear (CBM)':>18} {'MLP (CBM)':>18}")
    for inp in inputs:
        cells = []
        for name in heads:
            d, frac, conf = results[name][inp]
            cells.append(f"{d} ({frac*100:.0f}%,c{conf:.2f})")
        print(f"  {inp:26s} {cells[0]:>18} {cells[1]:>18} {cells[2]:>18}")

    print("\n=> if linear/MLP also collapse to one digit at high confidence, the "
          "OOD\n   over-confidence is a general concept-bottleneck issue; the "
          "logic head\n   merely makes the WHICH and WHY predictable.")


if __name__ == "__main__":
    main()
