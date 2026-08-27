"""Locate the emergent-CUB ceiling by decomposing where accuracy is lost.

Freeze a trained CUBEmergent encoder and compare, on the SAME frozen features:
  - logic-tree head        (the graded-logic BACON head we trained; from ckpt acc)
  - linear probe on the K concepts
  - MLP probe on the K concepts        -> concept-INFORMATION ceiling
  - linear probe on the 512-d backbone -> cost of the K-concept bottleneck

Gaps decompose the loss:
  logic-head -> MLP-on-concepts   = head cost (is the tree head leaving info on the table?)
  MLP-on-concepts -> 512-probe    = bottleneck cost (do K concepts throw info away?)
  512-probe   -> blackbox (~75)   = feature cost

    python cub_probe_ceiling.py --load saved/cub_tree_k24_400ep_hybrid.pt --K 24
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "table"))
sys.path.insert(0, os.path.abspath(os.path.join(_HERE, "..", "..")))

import _cub                                                     # noqa: E402
from cub_emergent import CUBEmergent                            # noqa: E402


@torch.no_grad()
def extract(model, loader, device):
    """Return frozen (concepts K, backbone-feats 512, labels) for a split."""
    model.eval()
    C, Z, Y = [], [], []
    for img, _c, y in loader:
        img = img.to(device)
        z = model.backbone(img)                    # (B, 512)
        c = torch.sigmoid(model.concept(z))        # (B, K)
        Z.append(z.cpu()); C.append(c.cpu()); Y.append(y)
    return torch.cat(C), torch.cat(Z), torch.cat(Y)


def train_probe(Xtr, Ytr, Xte, Yte, n_cls, device, hidden=0, epochs=300, lr=1e-2):
    """Train a linear (hidden=0) or 1-hidden-layer MLP probe; return test acc."""
    d = Xtr.shape[1]
    if hidden > 0:
        probe = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Linear(hidden, n_cls))
    else:
        probe = nn.Linear(d, n_cls)
    probe = probe.to(device)
    Xtr, Ytr = Xtr.to(device), Ytr.to(device)
    Xte, Yte = Xte.to(device), Yte.to(device)
    opt = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = Xtr.shape[0]
    for _ in range(epochs):
        probe.train()
        perm = torch.randperm(n, device=device)
        for i in range(0, n, 1024):
            idx = perm[i:i + 1024]
            loss = F.cross_entropy(probe(Xtr[idx]), Ytr[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        sched.step()
    probe.eval()
    with torch.no_grad():
        acc = (probe(Xte).argmax(1) == Yte).float().mean().item()
    return acc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", type=str, required=True)
    ap.add_argument("--K", type=int, required=True)
    ap.add_argument("--head", type=str, default="tree")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CUBEmergent(args.K, head=args.head).to(device)
    ckpt = torch.load(args.load, map_location=device)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    logic_acc = ckpt.get("acc", None)

    tl = DataLoader(_cub._CUBImages("train", False), batch_size=128,
                    shuffle=False, num_workers=args.workers, pin_memory=True)
    vl = DataLoader(_cub._CUBImages("test", False), batch_size=128,
                    shuffle=False, num_workers=args.workers, pin_memory=True)

    print(f"Extracting frozen features (K={args.K}) from {os.path.basename(args.load)} ...")
    Ctr, Ztr, Ytr = extract(model, tl, device)
    Cte, Zte, Yte = extract(model, vl, device)
    n_cls = int(max(Ytr.max(), Yte.max()).item()) + 1

    lin_c = train_probe(Ctr, Ytr, Cte, Yte, n_cls, device, hidden=0)
    mlp_c = train_probe(Ctr, Ytr, Cte, Yte, n_cls, device, hidden=256)
    lin_z = train_probe(Ztr, Ytr, Zte, Yte, n_cls, device, hidden=0)

    print("\n=================== EMERGENT-CUB CEILING DECOMPOSITION ===================")
    if logic_acc is not None:
        print(f"  logic-tree head (trained)      : {logic_acc * 100:5.2f}%   <- what we have")
    print(f"  linear probe on {args.K:>2d} concepts    : {lin_c * 100:5.2f}%")
    print(f"  MLP probe on {args.K:>2d} concepts       : {mlp_c * 100:5.2f}%   <- concept-INFO ceiling")
    print(f"  linear probe on 512 backbone   : {lin_z * 100:5.2f}%   <- bottleneck-free features")
    print("  (blackbox ResNet18 reference   : ~74.9% from OCBM table)")
    print("--------------------------------------------------------------------------")
    if logic_acc is not None:
        print(f"  head cost   (MLP-concepts - logic-head) : {(mlp_c - logic_acc) * 100:+5.2f} pts")
    print(f"  bottleneck  (512-probe - MLP-concepts)  : {(lin_z - mlp_c) * 100:+5.2f} pts")
    print("==========================================================================")


if __name__ == "__main__":
    main()
