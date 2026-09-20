r"""MoM (Mixture of Models) confidence-routing ensemble over faithful GL trees.

Given several hardened recttree checkpoints (e.g. one seed's W64/W8/W4 heads),
runs each on the CUB test set and reports: per-member accuracy, MoM = route each
image to the single most-confident tree (stays faithful -- one tree explains each
prediction), majority-vote (realizable), and the ORACLE (>=1 correct) upper bound.

    py -3 route_ensemble.py --ckpts saved/B_s0.pt saved/C_s0.pt saved/D_s0.pt
    py -3 route_ensemble.py --ckpts <...> --labels W64 W8 W4
"""
import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "table"))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

import _cub                                                     # noqa: E402
from interpret_recttree_cub import load_model                  # noqa: E402


@torch.no_grad()
def evaluate(ckpts, labels, workers=4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    models, names = {}, []
    attr312 = True
    for path, lab in zip(ckpts, labels):
        if not os.path.exists(path):
            print(f"[skip] {lab}: {path} not found")
            continue
        m, K, a312, *_ = load_model(path, device)
        models[lab] = m
        attr312 = a312
        names.append(lab)
    image_size = 299
    loader = DataLoader(_cub._CUBImages("test", False, attr312=attr312, image_size=image_size),
                        batch_size=32, shuffle=False, num_workers=workers, pin_memory=True)
    P, C, labels_ = {n: [] for n in names}, {n: [] for n in names}, []
    for img, c, y in loader:
        img = img.to(device)
        labels_.append(y)
        for n in names:
            logits = models[n](img)[0]
            P[n].append(logits.argmax(1).cpu())
            C[n].append(torch.softmax(logits, 1).max(1).values.cpu())
    y = torch.cat(labels_)
    P = {n: torch.cat(P[n]) for n in names}
    C = {n: torch.cat(C[n]) for n in names}
    N = y.numel()

    acc = {n: 100.0 * (P[n] == y).float().mean().item() for n in names}
    stackP = torch.stack([P[n] for n in names], 1)             # (N, M)
    stackC = torch.stack([C[n] for n in names], 1)
    correct = torch.stack([(P[n] == y) for n in names], 1)
    oracle = 100.0 * (correct.sum(1) >= 1).float().mean().item()
    maj = 100.0 * (torch.mode(stackP, 1).values == y).float().mean().item()
    pick = stackC.argmax(1)
    mom = 100.0 * (stackP[torch.arange(N), pick] == y).float().mean().item()
    return names, acc, mom, maj, oracle, N


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", nargs="+", required=True)
    ap.add_argument("--labels", nargs="+", default=None)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    labels = args.labels or [os.path.basename(p)[:20] for p in args.ckpts]
    names, acc, mom, maj, oracle, N = evaluate(args.ckpts, labels, args.workers)
    print(f"\nN test = {N}   members = {len(names)}")
    for n in names:
        print(f"  {n:<20} {acc[n]:.2f}%")
    best = max(acc.values())
    print(f"\n  majority-vote        {maj:.2f}%")
    print(f"  MoM (confidence)     {mom:.2f}%   (+{mom - best:.2f} over best member)")
    print(f"  ORACLE (>=1 correct) {oracle:.2f}%   (ceiling)")


if __name__ == "__main__":
    main()
