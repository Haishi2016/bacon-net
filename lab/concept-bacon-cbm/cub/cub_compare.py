"""
CUB-200-2011 comparison: fixed manually-generated BACON trees vs baselines.

Uses FROZEN ResNet-18 (ImageNet) features (extracted once and cached) so every
head trains in seconds, making the comparison fast.  Models:

  BlackBox  : features -> Linear(200)                       (task-only reference)
  SoftCBM   : features -> sigmoid concepts(112) -> Linear(200)  (leakage-prone)
  BaconCBM  : features -> sigmoid concepts(112) -> 200 FIXED signed-AND trees
              generated from per-class concept prototypes (cub_trees.py).

The 200 BACON trees are NOT trained (manually generated); only the concept
extractor trains.  We report task accuracy, concept accuracy, and leakage
Λ = max(task(via concepts) − C_true→Y ceiling, 0)  (ceiling = 100% for CUB).

    python cub_compare.py
"""

from __future__ import annotations

import os
import pickle
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from PIL import Image
from torchvision import models, transforms

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from cub_trees import CUB, load_split, build_prototypes, graded_and, load_frequencies  # noqa: E402

_CACHE = os.path.join(_HERE, "cache")
_EPS = 1e-6


# --------------------------------------------------------------------------- #
# Frozen feature extraction (run once, cached)
# --------------------------------------------------------------------------- #
def _local_path(img_path: str) -> str:
    i = img_path.replace("\\", "/").find("images/")
    return os.path.join(CUB, img_path.replace("\\", "/")[i:])


class _ImgDataset(torch.utils.data.Dataset):
    _tf = transforms.Compose([
        transforms.Resize(256), transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def __init__(self, entries):
        self.entries = entries

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, i):
        e = self.entries[i]
        img = Image.open(_local_path(e["img_path"])).convert("RGB")
        return self._tf(img), i


@torch.no_grad()
def extract_features(split: str, device: str) -> str:
    out = os.path.join(_CACHE, f"feat_{split}.pt")
    if os.path.exists(out):
        return out
    os.makedirs(_CACHE, exist_ok=True)
    with open(os.path.join(CUB, f"{split}.pkl"), "rb") as f:
        entries = pickle.load(f)
    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = nn.Identity()
    net.eval().to(device)
    loader = DataLoader(_ImgDataset(entries), batch_size=128, num_workers=4)
    feats = torch.zeros(len(entries), 512)
    print(f"  extracting {split} features ({len(entries)} images)...")
    for x, idx in loader:
        feats[idx] = net(x.to(device)).cpu()
    C = torch.tensor([e["attribute_label"] for e in entries], dtype=torch.float32)
    y = torch.tensor([e["class_label"] for e in entries], dtype=torch.long)
    torch.save({"feat": feats, "concept": C, "label": y}, out)
    return out


def load_features(split, device):
    d = torch.load(os.path.join(_CACHE, f"feat_{split}.pt"), map_location="cpu")
    return d["feat"], d["concept"], d["label"]


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #
class BlackBox(nn.Module):
    def __init__(self, K=112, L=200):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, L))

    def forward(self, x):
        return self.head(x), None


class SoftCBM(nn.Module):
    def __init__(self, K=112, L=200):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, K))
        self.head = nn.Linear(K, L)

    def forward(self, x):
        c = torch.sigmoid(self.enc(x))
        return self.head(c), c


class BaconFixed(nn.Module):
    """Frozen 200 prototype trees; only the concept encoder trains.

    design="signed"   : AND over all 112 signed literals (strict template match).
    design="positive" : AND over only the ~23 ON concepts (sparser, more robust).
    design="weighted" : statistical -- majority-direction literal weighted by
                        confidence |2*freq-1| (optional ~0.5 features ~ ignored).
    """

    def __init__(self, P: torch.Tensor, design="signed", F: torch.Tensor = None,
                 gamma=1.0, K=112, L=200, temp=8.0):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, K))
        self.register_buffer("P", P)                 # (L, K) fixed prototypes
        self.design = design
        self.temp = temp
        if design in ("weighted", "soft", "distinct"):
            self.register_buffer("sign", (F >= 0.5).float())
            if design == "distinct":
                # emphasise features where this class deviates from the average
                # class (discriminative), not just within-class confidence.
                self.register_buffer("w", (F - F.mean(0, keepdim=True)).abs().pow(gamma))
            else:
                self.register_buffer("w", (2.0 * F - 1.0).abs().pow(gamma))

    def forward(self, x):
        c = torch.sigmoid(self.enc(x))               # (B,K)
        cc = c.unsqueeze(1)                          # (B,1,K)
        if self.design in ("weighted", "soft", "distinct"):
            s = self.sign.unsqueeze(0)               # (1,L,K)
            match = s * cc + (1 - s) * (1 - cc)       # (B,L,K)
            w = self.w.unsqueeze(0)
            if self.design == "weighted":            # graded AND (geo-mean, andness~1)
                logm = torch.log(match.clamp_min(_EPS))
                logtruth = (logm * w).sum(-1) / self.w.sum(-1).clamp_min(_EPS)
                return self.temp * logtruth, c
            # soft / distinct: weighted ARITHMETIC mean (andness ~0.5), robust
            truth = (match * w).sum(-1) / self.w.sum(-1).clamp_min(_EPS)   # (B,L) in [0,1]
            truth = truth.clamp(_EPS, 1 - _EPS)
            return self.temp * (torch.log(truth) - torch.log1p(-truth)), c
        p = self.P.unsqueeze(0)                      # (1,L,K)
        if self.design == "signed":
            match = p * cc + (1 - p) * (1 - cc)       # agreement (B,L,K)
            logtruth = torch.log(match.clamp_min(_EPS)).mean(-1)         # (B,L)
        else:                                         # positive-only AND
            logc = torch.log(cc.clamp_min(_EPS))      # (B,1,K)
            logtruth = (logc * p).sum(-1) / p.sum(-1).clamp_min(1.0)     # (B,L)
        return self.temp * logtruth, c


# --------------------------------------------------------------------------- #
# Train / eval
# --------------------------------------------------------------------------- #
def train(model, tl, device, epochs, lam, has_concept):
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(epochs):
        for x, c, y in tl:
            x, c, y = x.to(device), c.to(device), y.to(device)
            logits, cp = model(x)
            loss = F.cross_entropy(logits, y)
            if has_concept and cp is not None:
                loss = loss + lam * F.binary_cross_entropy(cp, c)
            opt.zero_grad(); loss.backward(); opt.step()
    return model


@torch.no_grad()
def evaluate(model, vl, device):
    model.eval()
    correct = total = 0
    cacc_sum = cn = 0
    for x, c, y in vl:
        x, c, y = x.to(device), c.to(device), y.to(device)
        logits, cp = model(x)
        correct += (logits.argmax(1) == y).sum().item(); total += y.numel()
        if cp is not None:
            cacc_sum += ((cp > 0.5).float() == c).float().mean().item() * y.numel(); cn += y.numel()
    return correct / total, (cacc_sum / cn if cn else float("nan"))


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    for split in ("train", "test"):
        extract_features(split, device)

    Ftr, Ctr, ytr = load_features("train", device)
    Fte, Cte, yte = load_features("test", device)
    P = build_prototypes(Ctr, ytr)                  # (200,112)
    F, _ = load_frequencies(P)                       # (200,112) freqs

    tl = DataLoader(TensorDataset(Ftr, Ctr, ytr), batch_size=128, shuffle=True)
    vl = DataLoader(TensorDataset(Fte, Cte, yte), batch_size=512)

    print(f"\nCUB frozen-ResNet18 features | train {len(ytr)} test {len(yte)} "
          f"| device {device}")
    print("C_true->Y ceiling (fixed trees on true concepts) = 100%\n")

    configs = [
        ("BlackBox", BlackBox(), False),
        ("SoftCBM", SoftCBM(), True),
        ("BaconCBM (signed-AND)", BaconFixed(P, design="signed"), True),
        ("BaconCBM (positive-AND)", BaconFixed(P, design="positive"), True),
        ("BaconCBM (weighted/stat)", BaconFixed(P, design="weighted", F=F), True),
        ("BaconCBM (soft weighted)", BaconFixed(P, design="soft", F=F), True),
        ("BaconCBM (distinct/tfidf)", BaconFixed(P, design="distinct", F=F), True),
    ]
    print(f"{'model':24s} {'task acc':>9s} {'concept acc':>12s} {'leakage':>8s}")
    for name, model, has_c in configs:
        train(model, tl, device, epochs=30, lam=5.0, has_concept=has_c)
        task, cacc = evaluate(model, vl, device)
        leak = max(task - 1.0, 0.0)                  # ceiling is 1.0
        cstr = f"{cacc*100:11.2f}" if has_c else "          -"
        print(f"{name:24s} {task*100:8.2f} {cstr} {leak*100:7.2f}")


if __name__ == "__main__":
    main()
