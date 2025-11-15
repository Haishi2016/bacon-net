#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
LTN-style MNIST Addition

- Two CNN digit classifiers (one per image)
- Logic Tensor Network–style satisfaction loss enforcing:
    digit_1(x1) + digit_2(x2) = sum_label

Training objective:
    maximize sat_y = Σ_{i+j=y} p1(i|x1) * p2(j|x2)
via loss = - E[log sat_y].
"""

import argparse, math, random, os
from typing import Tuple, Iterator
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, IterableDataset, DataLoader
from torchvision import datasets, transforms


# ------------------------------------------------------------
# Streaming Dataset: re-sample MNIST pairs every epoch
# ------------------------------------------------------------
class MnistAdditionStream(IterableDataset):
    """
    Iterable dataset: yields epoch_len pairs per epoch.
    Each sample: ((img1, img2), sum) where sum in {0..18}.
    """
    def __init__(self, root: str, train: bool, epoch_len: int, seed: int = 42):
        super().__init__()
        self.seed = seed
        self.epoch_len = epoch_len
        self.train = train
        aug = [
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
        ] if train else []
        self.mnist = datasets.MNIST(
            root=root, train=train, download=True,
            transform=transforms.Compose(
                aug + [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ]
            )
        )

    def __iter__(self) -> Iterator:
        # Different RNG per worker; epoch variation handled by caller via seed param
        worker_info = torch.utils.data.get_worker_info()
        base = self.seed + (0 if worker_info is None else worker_info.id)
        rng = random.Random(base)
        n = len(self.mnist)
        for _ in range(self.epoch_len):
            i1 = rng.randrange(n)
            i2 = rng.randrange(n)
            x1, d1 = self.mnist[i1]
            x2, d2 = self.mnist[i2]
            yield (x1, x2), int(d1 + d2)


# ------------------------------------------------------------
# Fixed test set (so accuracy is stable across epochs)
# ------------------------------------------------------------
class MnistAdditionFixed(Dataset):
    def __init__(self, root: str, train: bool, size_pairs: int, seed: int = 999):
        super().__init__()
        rng = random.Random(seed)
        self.mnist = datasets.MNIST(
            root=root, train=train, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ])
        )
        n = len(self.mnist)
        self.pairs = []
        for _ in range(size_pairs):
            i1 = rng.randrange(n)
            i2 = rng.randrange(n)
            _, d1 = self.mnist[i1]
            _, d2 = self.mnist[i2]
            self.pairs.append((i1, i2, int(d1 + d2)))

    def __len__(self): return len(self.pairs)

    def __getitem__(self, idx):
        i1, i2, s = self.pairs[idx]
        x1, _ = self.mnist[i1]
        x2, _ = self.mnist[i2]
        return (x1, x2), s


# ------------------------------------------------------------
# Small CNN feature tower (with BatchNorm)
# ------------------------------------------------------------
class SmallCnn(nn.Module):
    def __init__(self, out_features=128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),  # 14x14
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),  # 7x7
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, out_features), nn.ReLU(),
        )

    def forward(self, x):
        return self.fc(self.conv(x))


# ------------------------------------------------------------
# Digit nets + LTN-style addition reasoning
# ------------------------------------------------------------
class DigitNet(nn.Module):
    """CNN -> 10-way digit classifier."""
    def __init__(self, feat_dim=128, n_digits=10):
        super().__init__()
        self.cnn = SmallCnn(out_features=feat_dim)
        self.fc_digit = nn.Linear(feat_dim, n_digits)

    def forward(self, x):
        feat = self.cnn(x)                  # [B, feat_dim]
        logits = self.fc_digit(feat)        # [B, 10]
        probs = F.softmax(logits, dim=1)    # [B, 10]
        return logits, probs


class LTNAddition(nn.Module):
    """
    LTN-style MNIST-addition model:
    - Two DigitNet towers (one per image)
    - LTN satisfaction for the constraint sum = d1 + d2
    """
    def __init__(self, feat_dim=128):
        super().__init__()
        self.d1 = DigitNet(feat_dim=feat_dim)  # x1 -> digit dist
        self.d2 = DigitNet(feat_dim=feat_dim)  # x2 -> digit dist

        # Precompute all digit pair sums: sum_idx[i,j] = i+j
        sum_idx = torch.arange(10).view(10, 1) + torch.arange(10).view(1, 10)
        self.register_buffer("sum_idx", sum_idx)  # shape [10,10]

    def forward(self, x1, x2):
        """
        Returns:
          logits1, logits2: [B,10] raw digit logits
          p1, p2: [B,10] digit probabilities
        """
        logits1, p1 = self.d1(x1)
        logits2, p2 = self.d2(x2)
        return logits1, logits2, p1, p2

    def satisfaction_per_sum(self, p1, p2):
        """
        Compute, for each sample, satisfaction degree for each possible sum k in [0..18].

        p1, p2: [B,10] digit distributions
        Returns:
          sat: [B, 19], where sat[b,k] = Σ_{i+j=k} p1[b,i]*p2[b,j]
        """
        # T[b,i,j] = p1[b,i] * p2[b,j]  (product t-norm)
        T = p1.unsqueeze(2) * p2.unsqueeze(1)      # [B,10,10]

        sats = []
        for k in range(19):
            mask_k = (self.sum_idx == k)          # [10,10] bool
            # Flatten masked entries for each batch element and sum
            s_k = (T[:, mask_k]).sum(dim=1)       # [B]
            sats.append(s_k)
        sat = torch.stack(sats, dim=1)            # [B,19]
        return sat

    def predict_sum(self, p1, p2):
        """
        Turn digit distributions into a predicted sum label via LTN-style reasoning:
          y_hat = argmax_k sat_k
        """
        sat = self.satisfaction_per_sum(p1, p2)   # [B,19]
        return sat.argmax(dim=1)                  # [B]


# ------------------------------------------------------------
# LTN satisfaction loss
# ------------------------------------------------------------
def ltn_satisfaction_loss(sat: torch.Tensor, y: torch.Tensor, eps: float = 1e-9):
    """
    sat: [B,19] satisfaction per possible sum (from model.satisfaction_per_sum)
    y:   [B] ground-truth sum labels (0..18)

    Loss = - mean log sat_y  (maximize truth of the formula for each example)
    """
    # sat_y[b] = sat[b, y[b]]
    sat_y = sat.gather(1, y.view(-1, 1)).squeeze(1)  # [B]
    loss = -torch.log(sat_y + eps).mean()
    return loss


# ------------------------------------------------------------
# Pairwise rule printer (clean OR-of-ANDs per sum) — device-safe
# ------------------------------------------------------------
def print_pair_rules_ltn(model, test_loader, device="cpu"):
    """
    Mimics your DCR rule printer, but using predicted digit distributions.

    For each sum class k, we estimate contributions of digit pairs (i,j)
    and print only the valid ones with i + j = k.
    """
    model.eval()
    pair_sum = torch.zeros(19, 10, 10, device=device)
    count_k  = torch.zeros(19, device=device)

    with torch.no_grad():
        for (x1, x2), _ in test_loader:
            x1, x2 = x1.to(device), x2.to(device)
            _, _, p1, p2 = model(x1, x2)                # [B,10] each

            sat = model.satisfaction_per_sum(p1, p2)    # [B,19]
            k = sat.argmax(1)                           # [B]
            outer = p1.unsqueeze(2) * p2.unsqueeze(1)   # [B,10,10]

            for kk in range(19):
                mask = (k == kk).float().view(-1, 1, 1)
                if mask.sum() == 0:
                    continue
                pair_sum[kk] += (outer * mask).sum(dim=0)
                count_k[kk]  += mask.sum()

    print("\n=== LTN Pairwise rules (valid digit pairs i∧j with i+j=k) ===")
    for kk in range(19):
        if count_k[kk] == 0:
            print(f"y_{kk}: (no samples)")
            continue
        contrib = pair_sum[kk] / count_k[kk]
        pairs = [(i, j, contrib[i, j].item()) for i in range(10) for j in range(10)]
        pairs.sort(key=lambda t: t[2], reverse=True)
        valid = [(i, j, v) for (i, j, v) in pairs if i + j == kk]
        human = " ∨ ".join([f"({i}∧{j})" for (i, j, _) in valid]) or "(none)"
        print(f"y_{kk}: {human}")
    print("=============================================================\n")


# ------------------------------------------------------------
# Train / Eval
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data", type=str)
    ap.add_argument("--train-pairs", default=80000, type=int,
                    help="pairs per epoch (streamed)")
    ap.add_argument("--test-pairs", default=5000, type=int)
    ap.add_argument("--batch-size", default=128, type=int)
    ap.add_argument("--epochs", default=80, type=int)
    ap.add_argument("--lr", default=1e-3, type=float)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--patience", default=15, type=int)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_ds = MnistAdditionStream(args.data, train=True,
                                   epoch_len=args.train_pairs,
                                   seed=args.seed)
    train_loader = DataLoader(train_ds,
                              batch_size=args.batch_size,
                              num_workers=2)

    test_ds = MnistAdditionFixed(args.data, train=False,
                                 size_pairs=args.test_pairs,
                                 seed=999)
    test_loader = DataLoader(test_ds,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=2,
                             pin_memory=True)

    model = LTNAddition(feat_dim=128).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        # ---- Train ----
        model.train()
        running_loss = 0.0
        seen = 0

        for (x1, x2), y in train_loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)

            _, _, p1, p2 = model(x1, x2)              # [B,10]
            sat = model.satisfaction_per_sum(p1, p2)  # [B,19]

            loss = ltn_satisfaction_loss(sat, y)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bs = x1.size(0)
            running_loss += float(loss.item()) * bs
            seen += bs
            if seen >= args.train_pairs:
                break

        train_loss = running_loss / max(1, seen)

        # ---- Eval ----
        model.eval()
        with torch.no_grad():
            tot, correct = 0, 0
            for (x1, x2), y in test_loader:
                x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                _, _, p1, p2 = model(x1, x2)
                y_hat = model.predict_sum(p1, p2)
                correct += (y_hat == y).sum().item()
                tot += y.size(0)
        acc = correct / max(1, tot)

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
              f"test_acc={acc*100:.2f}%")

        # Early stopping
        if acc > best_acc:
            best_acc = acc
            bad_epochs = 0
            torch.save(model.state_dict(), "best_ltn_mnist_addition.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping (no improvement for {args.patience} epochs).")
                break

    print(f"\nBest LTN-style test_acc: {best_acc*100:.2f}% "
          f"(checkpoint: best_ltn_mnist_addition.pt)")

    # Pairwise rule view
    print_pair_rules_ltn(model, test_loader, device=device)


if __name__ == "__main__":
    main()
