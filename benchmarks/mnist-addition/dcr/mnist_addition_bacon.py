#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, math, random, os
from typing import Tuple, Iterator
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, IterableDataset, DataLoader
from torchvision import datasets, transforms

from bacon.binaryTreeLogicNet import binaryTreeLogicNet
from bacon.aggregators.bool import MinMaxAggregator


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
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, out_features), nn.ReLU(),
        )

    def forward(self, x):
        return self.fc(self.conv(x))


# ------------------------------------------------------------
# Concept head per image: simple logits + Gumbel-Softmax to 10-way probs
# ------------------------------------------------------------
class ImageConceptHead(nn.Module):
    def __init__(self, feat_dim: int, n_concepts=10):
        super().__init__()
        self.logit_head = nn.Linear(feat_dim, n_concepts)

    def forward(self, features, tau: float, hard: bool = False) -> torch.Tensor:
        logits = self.logit_head(features)               # [B,10]
        probs = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=1)  # [B,10], sum=1
        return probs


# ------------------------------------------------------------
# BACON-top model for sum==10 (binary)
# ------------------------------------------------------------
class BaconAddition(nn.Module):
    def __init__(self):
        super().__init__()
        self.tower1 = SmallCnn(128)
        self.tower2 = SmallCnn(128)
        self.head1 = ImageConceptHead(128, 10)
        self.head2 = ImageConceptHead(128, 10)
        self.bacon = binaryTreeLogicNet(
            input_size=20,
            is_frozen=False,
            weight_mode="fixed",
            weight_normalization="minmax",
            aggregator=MinMaxAggregator(),
            normalize_andness=False,
            tree_layout="paired",
            loss_amplifier=1.0,
        )

    def forward(self, x1, x2, tau: float, hard: bool = False):
        z1 = self.tower1(x1)
        z2 = self.tower2(x2)
        p1 = self.head1(z1, tau=tau, hard=hard)  # [B,10]
        p2 = self.head2(z2, tau=tau, hard=hard)  # [B,10]
        c_prob = torch.cat([p1, p2], dim=1)      # [B,20]
        y_score = self.bacon(c_prob)             # [B,1] in [0,1]
        return y_score, c_prob


# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------
def group_entropy_loss(c_prob):
    eps = 1e-8
    g1 = c_prob[:, :10]
    g2 = c_prob[:, 10:]
    ent = -(g1 * (g1 + eps).log()).sum(dim=1) - (g2 * (g2 + eps).log()).sum(dim=1)
    return (ent / math.log(10)).mean()


def print_pair_rules_sum10(model, test_loader, tau_eval=0.8, device='cpu'):
    model.eval()
    pair_sum = torch.zeros(10, 10, device=device)
    count_pos = 0.0
    with torch.no_grad():
        for (x1, x2), _ in test_loader:
            x1, x2 = x1.to(device), x2.to(device)
            z1 = model.tower1(x1); z2 = model.tower2(x2)
            p1 = model.head1(z1, tau=tau_eval, hard=True)  # [B,10]
            p2 = model.head2(z2, tau=tau_eval, hard=True)  # [B,10]
            y_score, _ = model(x1, x2, tau=tau_eval, hard=True)
            pos_mask = (y_score.squeeze(1) >= 0.5).float().view(-1, 1, 1)
            outer = p1.unsqueeze(2) * p2.unsqueeze(1)
            pair_sum += (outer * pos_mask).sum(dim=0)
            count_pos += pos_mask.sum().item()

    print("\n=== Pairwise rules for sum==10 (valid i∧j pairs where i+j=10) ===")
    if count_pos == 0:
        print("No positive predictions.")
        print("===============================================\n")
        return
    contrib = pair_sum / max(1.0, count_pos)
    pairs = [(i, j, contrib[i, j].item()) for i in range(10) for j in range(10) if i + j == 10]
    pairs.sort(key=lambda t: t[2], reverse=True)
    human = " ∨ ".join([f"({i}∧{j})" for (i, j, _) in pairs]) or "(none)"
    print(f"sum=10: {human}")
    print("===============================================\n")


# ------------------------------------------------------------
# Train / Eval
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data", type=str)
    ap.add_argument("--train-pairs", default=80000, type=int, help="pairs per epoch (streamed)")
    ap.add_argument("--test-pairs", default=5000, type=int)
    ap.add_argument("--batch-size", default=128, type=int)
    ap.add_argument("--epochs", default=120, type=int)
    ap.add_argument("--lr", default=1e-3, type=float)
    ap.add_argument("--entropy", default=0.0005, type=float, help="tiny entropy kept always")
    ap.add_argument("--tau-start", default=3.0, type=float)
    ap.add_argument("--tau-end", default=0.6, type=float)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--patience", default=15, type=int)
    ap.add_argument("--pretrained", default=None, type=str, help="path to a pretrained model state_dict (.pt)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = MnistAdditionStream(args.data, train=True, epoch_len=args.train_pairs, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=2)

    test_ds = MnistAdditionFixed(args.data, train=False, size_pairs=args.test_pairs, seed=999)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = BaconAddition().to(device)

    # Optionally load pretrained weights
    if args.pretrained is not None:
        if not os.path.isfile(args.pretrained):
            raise FileNotFoundError(f"Checkpoint not found: {args.pretrained}")
        try:
            ckpt = torch.load(args.pretrained, map_location=device, weights_only=True)
        except TypeError:
            ckpt = torch.load(args.pretrained, map_location=device)
        except Exception as e:
            print(f"Safe load (weights_only=True) failed: {e}. Falling back to standard torch.load (unsafe).")
            ckpt = torch.load(args.pretrained, map_location=device)
        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            state_dict = ckpt
        elif hasattr(ckpt, "state_dict"):
            state_dict = ckpt.state_dict()
        else:
            raise RuntimeError("Unrecognized checkpoint format; expected a state_dict or a dict with 'state_dict'.")
        cleaned_state_dict = {}
        for k, v in state_dict.items():
            nk = k[7:] if k.startswith("module.") else k
            cleaned_state_dict[nk] = v
        missing, unexpected = model.load_state_dict(cleaned_state_dict, strict=False)
        if missing:
            print(f"Warning: missing keys when loading pretrained weights: {sorted(missing)}")
        if unexpected:
            print(f"Warning: unexpected keys when loading pretrained weights: {sorted(unexpected)}")
        print(f"Loaded pretrained weights from {args.pretrained}")

        # Evaluate only, skip training
        model.eval()
        with torch.no_grad():
            tot, correct = 0, 0
            for (x1, x2), y in test_loader:
                x1, x2 = x1.to(device), x2.to(device)
                y = y.to(device)
                y10 = (y == 10).float().view(-1, 1)
                y_score, _ = model(x1, x2, tau=args.tau_end, hard=True)
                pred = (y_score >= 0.5).float()
                correct += (pred == y10).sum().item()
                tot += y10.numel()
            acc = correct / max(1, tot)
        print(f"Pretrained eval | binary sum==10 acc={acc*100:.2f}% | tau={args.tau_end:.3f}")
        print_pair_rules_sum10(model, test_loader, tau_eval=args.tau_end, device=device)
        return

    # Optimizer and scheduler
    bce = nn.BCELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-4)

    best_acc = 0.0
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        # τ linear annealing
        t = (epoch - 1) / max(1, args.epochs - 1)
        tau = args.tau_start + (args.tau_end - args.tau_start) * t

        # ---- Train ----
        model.train()
        running_loss = 0.0
        seen = 0

        for (x1, x2), y in train_loader:
            x1, x2 = x1.to(device), x2.to(device)
            y = y.to(device)
            y10 = (y == 10).float().view(-1, 1)

            y_score, c_prob = model(x1, x2, tau=tau, hard=False)
            loss_main = bce(y_score.clamp(1e-6, 1 - 1e-6), y10)
            loss_ent = group_entropy_loss(c_prob) * args.entropy
            loss = loss_main + loss_ent

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bs = x1.size(0)
            running_loss += float(loss.item()) * bs
            seen += bs
            if seen >= args.train_pairs:
                break

        sched.step()
        train_loss = running_loss / max(1, seen)

        # ---- Eval ----
        model.eval()
        with torch.no_grad():
            tot, correct = 0, 0
            for (x1, x2), y in test_loader:
                x1, x2 = x1.to(device), x2.to(device)
                y = y.to(device)
                y10 = (y == 10).float().view(-1, 1)
                y_score, _ = model(x1, x2, tau=args.tau_end, hard=True)
                pred = (y_score >= 0.5).float()
                correct += (pred == y10).sum().item()
                tot += y10.numel()
            acc = correct / max(1, tot)

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | bin_acc={acc*100:.2f}% | tau={tau:.3f}")

        # Early stopping on best test acc
        if acc > best_acc:
            best_acc = acc
            bad_epochs = 0
            torch.save(model.state_dict(), "best_bacon_mnist_addition.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping (no improvement for {args.patience} epochs).")
                break

    print(f"\nBest bin_acc: {best_acc*100:.2f}% (checkpoint: best_bacon_mnist_addition.pt)")
    print_pair_rules_sum10(model, test_loader, tau_eval=args.tau_end, device=device)


if __name__ == "__main__":
    main()


