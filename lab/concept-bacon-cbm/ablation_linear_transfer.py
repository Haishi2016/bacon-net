"""Head-swap ablation on FIXED concepts: does the graded-logic ontology head
transfer to USPS better than a plain linear head built on the SAME trained
concepts?

We take the frozen MNIST OCBM (encoder + K sigmoid concepts + logic trees),
hold the encoder/concepts FIXED, and:
  * measure the OCBM (ontology head) zero-shot USPS accuracy, and
  * fit a LINEAR head (and, for reference, a small MLP head) on the frozen
    MNIST-train concepts only, then evaluate it zero-shot on USPS.

Both heads consume the identical K-dim concept vectors, so any transfer
difference is attributable to the reasoning layer, not the representation.

    CUDA_VISIBLE_DEVICES="" py -3 ablation_linear_transfer.py --load saved/k5_harden.pt
"""
from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from eval_concept_transfer import (load_model, mnist_loader, usps_loader,  # noqa: E402
                                   collect, _MEAN, _STD)
from eval_tree_transfer_calibrated import calibrate, predict_from_probs  # noqa: E402


def mnist_train_loader(root, bs):
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((_MEAN,), (_STD,))])
    ds = datasets.MNIST(root, train=True, download=True, transform=tf)
    return DataLoader(ds, batch_size=bs)


def fit_head(head, Ctr, Ytr, K, epochs, seed, device):
    """Train a head on FROZEN concept vectors (MNIST train)."""
    torch.manual_seed(seed)
    if head == "linear":
        net = nn.Linear(K, 10)
    else:
        net = nn.Sequential(nn.Linear(K, 64), nn.ReLU(), nn.Linear(64, 10))
    net = net.to(device)
    opt = torch.optim.Adam(net.parameters(), lr=1e-2)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    n = len(Ytr)
    idx = torch.randperm(n)
    Ctr, Ytr = Ctr[idx], Ytr[idx]
    for _ in range(epochs):
        net.train()
        for s in range(0, n, 1024):
            xb, yb = Ctr[s:s + 1024].to(device), Ytr[s:s + 1024].to(device)
            opt.zero_grad()
            nn.functional.cross_entropy(net(xb), yb).backward()
            opt.step()
        sched.step()
    net.eval()
    return net


@torch.no_grad()
def head_acc(net, C, Y, device):
    return (net(C.to(device)).argmax(1).cpu() == Y).float().mean().item()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", default=os.path.join(_HERE, "saved", "k5_harden.pt"))
    ap.add_argument("--data", default="./data")
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, ckpt = load_model(args.load, device)
    K = ckpt["K"]

    # --- frozen concepts for every head (MNIST train/test + USPS test) ---
    mnist_te = mnist_loader(args.data, args.batch_size)
    usps_te = usps_loader(args.data, args.batch_size)
    Ctr, Ytr = collect(model, mnist_train_loader(args.data, args.batch_size), device)
    Cm, Ym = collect(model, mnist_te, device)
    Cu, Yu = collect(model, usps_te, device)
    # unsupervised per-concept quantile calibration: USPS -> MNIST reference.
    # Applied to the concept vectors BEFORE any head, so it is head-agnostic.
    Cu_cal = calibrate(Cu, Cm)

    # --- OCBM (ontology head) zero-shot, raw + calibrated concepts ---
    ocbm_mnist = (predict_from_probs(model, Cm, device) == Ym).float().mean().item()
    ocbm_usps = (predict_from_probs(model, Cu, device) == Yu).float().mean().item()
    ocbm_usps_cal = (predict_from_probs(model, Cu_cal, device) == Yu).float().mean().item()

    print(f"\nHEAD-SWAP ABLATION on FIXED concepts  {os.path.basename(args.load)}  "
          f"K={K}\n  MNIST test N={len(Ym)}  USPS test N={len(Yu)}\n")
    print(f"{'head':<24}{'MNIST-test':>11}{'USPS raw':>11}{'USPS calib':>13}")
    print("-" * 59)
    print(f"{'OCBM (ontology)':<24}{ocbm_mnist*100:>10.2f}%"
          f"{ocbm_usps*100:>10.2f}%{ocbm_usps_cal*100:>12.2f}%")

    for head in ("linear", "mlp"):
        m_acc, u_acc, uc_acc = [], [], []
        for s in range(args.seeds):
            net = fit_head(head, Ctr, Ytr, K, args.epochs, s, device)
            m_acc.append(head_acc(net, Cm, Ym, device))
            u_acc.append(head_acc(net, Cu, Yu, device))
            uc_acc.append(head_acc(net, Cu_cal, Yu, device))
        mm, uu, uc = torch.tensor(m_acc), torch.tensor(u_acc), torch.tensor(uc_acc)
        print(f"{head + ' (frozen)':<24}{mm.mean()*100:>10.2f}%"
              f"{uu.mean()*100:>10.2f}%{uc.mean()*100:>12.2f}%"
              f"  (calib +/- {uc.std(unbiased=False)*100:.2f})")

    print("\nNote: all heads consume the identical frozen K-dim concept vectors; "
          "the linear/MLP heads are trained on MNIST-train concepts only and never "
          "see USPS (zero-shot). Calibration (per-concept quantile match to the MNIST "
          "reference) is head-agnostic. Differences isolate the reasoning layer.")


if __name__ == "__main__":
    main()
