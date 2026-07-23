"""
Reproduce CREAM's FashionMNIST results and compare against our BACON-logic CBM.

For each setting (iFMNIST = incomplete concepts, cFMNIST = complete concepts) we
train and evaluate:
  BlackBox, SoftCBM, CREAM (paper method), BaconCBM (ours), BaconCBM+side-channel
plus the C_true->Y linear reference used to quantify concept leakage.

Reported per model:
  Task(full)      task accuracy of the full model (side-channel on if present)
  Task(concepts)  task accuracy of the concept-only pathway (side-channel off)
  Concept         mean concept accuracy (softmax-argmax over mutex groups)
  Leakage         max(Task(concepts) - (C_true->Y accuracy), 0)

    python run_cream.py --epochs 20
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", "..", ".."))
sys.path.insert(0, _HERE)

import fmnist_concepts as fc                                       # noqa: E402
from models import (BlackBox, CtrueY, SoftCBM, CREAM, BaconCBM)    # noqa: E402


def load_fmnist(data_root, batch_size):
    tf = transforms.Compose([transforms.ToTensor(),
                             transforms.Normalize((0.2860,), (0.3530,))])
    tr = datasets.FashionMNIST(data_root, train=True, download=True, transform=tf)
    te = datasets.FashionMNIST(data_root, train=False, download=True, transform=tf)
    # preload to tensors for speed
    Xtr = torch.stack([tr[i][0] for i in range(len(tr))])
    ytr = torch.tensor(tr.targets)
    Xte = torch.stack([te[i][0] for i in range(len(te))])
    yte = torch.tensor(te.targets)
    tl = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch_size, shuffle=True)
    vl = DataLoader(TensorDataset(Xte, yte), batch_size=512, shuffle=False)
    return tl, vl


def concept_accuracy(cprobs, ctrue, spec):
    """Mutex groups via argmax + independent binary concepts via 0.5 threshold."""
    correct = total = 0
    for g in spec.mutex_groups:
        correct += (cprobs[:, g].argmax(1) == ctrue[:, g].argmax(1)).sum().item()
        total += cprobs.shape[0]
    for k in spec.binary_concepts:
        correct += ((cprobs[:, k] > 0.5).float() == ctrue[:, k]).sum().item()
        total += cprobs.shape[0]
    return correct / total if total else float("nan")


@torch.no_grad()
def evaluate(model, loader, spec, device, is_bacon_or_cream):
    model.eval()
    full_c = full_n = 0
    cpath_c = 0
    cacc_sum = cacc_n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        ctrue = spec.concept_targets(y)
        logits, cprobs = model(x)                       # side on
        full_c += (logits.argmax(1) == y).sum().item()
        full_n += y.numel()
        if is_bacon_or_cream:
            logits_cp, _ = model(x, use_side=False)     # concept-only path
        else:
            logits_cp = logits
        cpath_c += (logits_cp.argmax(1) == y).sum().item()
        if cprobs is not None:
            cacc_sum += concept_accuracy(cprobs, ctrue, spec) * y.numel()
            cacc_n += y.numel()
    concept = (cacc_sum / cacc_n) if cacc_n else float("nan")
    return full_c / full_n, cpath_c / full_n, concept


def train_model(model, tl, spec, device, epochs, lam, has_concepts):
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            logits, cprobs = model(x)
            loss = F.cross_entropy(logits, y)
            if has_concepts and cprobs is not None:
                ctrue = spec.concept_targets(y)
                loss = loss + lam * F.binary_cross_entropy(cprobs, ctrue)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
    return model


def train_ctruey(spec, tl, device, epochs=30):
    """Linear on ground-truth concepts -> class (the leakage reference)."""
    model = CtrueY(spec).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(epochs):
        for _, y in tl:
            y = y.to(device)
            c = spec.concept_targets(y)
            logits, _ = model(c)
            loss = F.cross_entropy(logits, y)
            opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def eval_ctruey(model, vl, spec, device):
    model.eval(); correct = total = 0
    for _, y in vl:
        y = y.to(device)
        logits, _ = model(spec.concept_targets(y))
        correct += (logits.argmax(1) == y).sum().item(); total += y.numel()
    return correct / total


def run_setting(spec, tl, vl, device, epochs, lam):
    print(f"\n{'='*74}\n{spec.name}  (K={spec.K} concepts, "
          f"{len(spec.groups)} mutex groups)\n{'='*74}")

    ctruey = train_ctruey(spec, tl, device)
    ctruey_acc = eval_ctruey(ctruey, vl, spec, device)
    print(f"C_true -> Y  (concept-only ceiling): {ctruey_acc*100:.2f}%")

    configs = [
        ("BlackBox",       BlackBox(),                         False, False),
        ("SoftCBM",        SoftCBM(spec),                      True,  False),
        ("CREAM",          CREAM(spec, dropout_p=0.9),         True,  True),
        ("BaconCBM (ours)", BaconCBM(spec),                    True,  True),
        ("BaconCBM+SC (ours)", BaconCBM(spec, d_y=20, dropout_p=0.9), True, True),
    ]

    print(f"\n{'Model':22s} {'Task(full)':>11s} {'Task(cpt)':>10s} "
          f"{'Concept':>9s} {'Leakage':>8s}")
    rows = []
    for name, model, has_c, has_side in configs:
        train_model(model, tl, spec, device, epochs, lam, has_c)
        if name.startswith("BaconCBM (ours)"):
            os.makedirs(os.path.join(_HERE, "saved"), exist_ok=True)
            model.save(os.path.join(_HERE, "saved", f"bacon_{spec.name}.pt"))
        accepts_side = name != "BlackBox"
        full, cpath, cacc = evaluate(model, vl, spec, device, accepts_side)
        # concept-path only meaningful for concept models
        if name == "BlackBox":
            leak = float("nan"); cpath_disp = full
        else:
            cpath_disp = cpath
            leak = max(cpath - ctruey_acc, 0.0)
        cstr = f"{cacc*100:8.2f}" if has_c else "       -"
        lstr = f"{leak*100:7.2f}" if name != "BlackBox" else "      -"
        print(f"{name:22s} {full*100:10.2f} {cpath_disp*100:9.2f} {cstr} {lstr}")
        rows.append((name, full, cpath_disp, cacc if has_c else None, None if name == "BlackBox" else leak))
    return ctruey_acc, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lam", type=float, default=1.0)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--data", type=str,
                    default=os.path.join(_REPO_ROOT, "benchmarks", "mnist-addition", "data"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device={device}  epochs={args.epochs}  lambda={args.lam}")

    tl, vl = load_fmnist(args.data, args.batch_size)
    for spec in (fc.IFMNIST, fc.SFMNIST, fc.CFMNIST):
        run_setting(spec, tl, vl, device, args.epochs, args.lam)


if __name__ == "__main__":
    main()
