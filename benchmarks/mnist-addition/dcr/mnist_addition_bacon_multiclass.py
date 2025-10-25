#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, math, random, os, sys
from typing import Iterator
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, IterableDataset, DataLoader
from torchvision import datasets, transforms

# Ensure local bacon package is used instead of any installed site-package
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))

from bacon.binaryTreeLogicNet import binaryTreeLogicNet
from bacon.aggregators.bool import MinMaxAggregator


# ------------------------------------------------------------
# Streaming Dataset
# ------------------------------------------------------------
class MnistAdditionStream(IterableDataset):
    def __init__(self, root: str, train: bool, epoch_len: int, seed: int = 42):
        super().__init__()
        self.seed = seed
        self.epoch_len = epoch_len
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
# Fixed Dataset
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
# Small CNN tower
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
# 10-way concept head per image
# ------------------------------------------------------------
class ImageConceptHead(nn.Module):
    def __init__(self, feat_dim: int, n_concepts=10):
        super().__init__()
        self.logit_head = nn.Linear(feat_dim, n_concepts)

    def forward(self, features, tau: float, hard: bool = False) -> torch.Tensor:
        logits = self.logit_head(features)
        probs = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=1)  # [B,10], sum=1
        return probs


# ------------------------------------------------------------
# Multiclass BACON head (19 classes)
# ------------------------------------------------------------
class BaconMultiHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.tower1 = SmallCnn(128)
        self.tower2 = SmallCnn(128)
        self.head1 = ImageConceptHead(128, 10)
        self.head2 = ImageConceptHead(128, 10)
        # Two small BACON trees (10 leaves each) to learn permutations for p1 and p2
        self.perm1 = binaryTreeLogicNet(
            input_size=10,
            is_frozen=False,
            weight_mode="fixed",
            weight_normalization="minmax",
            aggregator=MinMaxAggregator(),
            normalize_andness=False,
            tree_layout="paired",
            loss_amplifier=1.0,
        )
        self.perm2 = binaryTreeLogicNet(
            input_size=10,
            is_frozen=False,
            weight_mode="fixed",
            weight_normalization="minmax",
            aggregator=MinMaxAggregator(),
            normalize_andness=False,
            tree_layout="paired",
            loss_amplifier=1.0,
        )
        self.agg = MinMaxAggregator()

    def forward(self, x1, x2, tau: float, hard: bool = False):
        z1 = self.tower1(x1)
        z2 = self.tower2(x2)
        p1 = self.head1(z1, tau=tau, hard=hard)  # [B,10]
        p2 = self.head2(z2, tau=tau, hard=hard)  # [B,10]
        # Learn soft/hard permutations via BACON trees' input_to_leaf
        p1p = self.perm1.input_to_leaf(p1)  # [B,10]
        p2p = self.perm2.input_to_leaf(p2)  # [B,10]
        # Pairwise AND via min (a=1.0) using MinMaxAggregator (vectorized)
        B = p1p.size(0)
        x = p1p.unsqueeze(2).expand(B, 10, 10)
        y = p2p.unsqueeze(1).expand(B, 10, 10)
        a_and = torch.tensor(1.0, device=x.device)
        s = self.agg.aggregate_tensor(x, y, a_and, w0=0.5, w1=0.5)  # [B,10,10]
        # Smooth OR per class k: 1 - prod(1 - s_ij) for i+j=k
        mask = s.new_zeros(19, 10, 10)
        for k in range(19):
            for i in range(10):
                j = k - i
                if 0 <= j <= 9:
                    mask[k, i, j] = 1.0
        one_minus = (1.0 - s.clamp(1e-6, 1 - 1e-6))
        log_one_minus = (one_minus + 1e-12).log()
        y_list = []
        for k in range(19):
            mk = mask[k].unsqueeze(0).expand(B, 10, 10)
            log_prod_k = (log_one_minus * mk).sum(dim=(1, 2))
            yk = 1.0 - torch.exp(log_prod_k)
            y_list.append(yk.unsqueeze(1))
        y_prob = torch.cat(y_list, dim=1)  # [B,19]
        c_prob = torch.cat([p1, p2], dim=1)
        return y_prob, c_prob


# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------
def group_entropy_loss(c_prob):
    eps = 1e-8
    g1 = c_prob[:, :10]
    g2 = c_prob[:, 10:]
    ent = -(g1 * (g1 + eps).log()).sum(dim=1) - (g2 * (g2 + eps).log()).sum(dim=1)
    return (ent / math.log(10)).mean()


def print_pair_rules_all(model, test_loader, tau_eval=0.8, device='cpu'):
    model.eval()
    pair_sum = torch.zeros(19, 10, 10, device=device)
    count_k = torch.zeros(19, device=device)
    with torch.no_grad():
        for (x1, x2), _ in test_loader:
            x1, x2 = x1.to(device), x2.to(device)
            z1 = model.tower1(x1); z2 = model.tower2(x2)
            p1 = model.head1(z1, tau=tau_eval, hard=True)
            p2 = model.head2(z2, tau=tau_eval, hard=True)
            y_prob, _ = model(x1, x2, tau=tau_eval, hard=True)
            k = y_prob.argmax(1)
            outer = p1.unsqueeze(2) * p2.unsqueeze(1)
            for kk in range(19):
                mask = (k == kk).float().view(-1, 1, 1)
                pair_sum[kk] += (outer * mask).sum(dim=0)
                count_k[kk] += mask.sum()

    print("\n=== Pairwise rules (valid pairs i∧j with i+j=k) ===")
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
    print("===============================================\n")


# ------------------------------------------------------------
# Train / Eval
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data", type=str)
    ap.add_argument("--train-pairs", default=80000, type=int)
    ap.add_argument("--test-pairs", default=5000, type=int)
    ap.add_argument("--batch-size", default=128, type=int)
    ap.add_argument("--epochs", default=120, type=int)
    ap.add_argument("--lr", default=1e-3, type=float)
    ap.add_argument("--entropy", default=0.0005, type=float)
    ap.add_argument("--tau-start", default=3.0, type=float)
    ap.add_argument("--tau-end", default=0.6, type=float)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--patience", default=15, type=int)
    ap.add_argument("--pretrained", default=None, type=str)
    ap.add_argument("--auto-refine", action="store_true", help="enable gated hard concepts (optional)")
    ap.add_argument("--refine-tau-gate", default=None, type=float, help="enable hard concepts only when tau <= this value")
    ap.add_argument("--refine-acc-gate", default=None, type=float, help="enable hard concepts only when eval acc >= this value (0..1)")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = MnistAdditionStream(args.data, train=True, epoch_len=args.train_pairs, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=2)

    test_ds = MnistAdditionFixed(args.data, train=False, size_pairs=args.test_pairs, seed=999)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = BaconMultiHead().to(device)

    # Pretrained (optional)
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
        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else (ckpt.state_dict() if hasattr(ckpt, "state_dict") else ckpt)
        cleaned = { (k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items() }
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if missing: print(f"Warning: missing keys: {sorted(missing)}")
        if unexpected: print(f"Warning: unexpected keys: {sorted(unexpected)}")
        model.eval()
        with torch.no_grad():
            tot, correct = 0, 0
            for (x1, x2), y in test_loader:
                x1, x2 = x1.to(device), x2.to(device)
                y = y.to(device)
                y_prob, _ = model(x1, x2, tau=args.tau_end, hard=True)
                pred = y_prob.argmax(1)
                correct += (pred == y).sum().item()
                tot += y.size(0)
            acc = correct / max(1, tot)
        print(f"Pretrained eval | multiclass acc={acc*100:.2f}% | tau={args.tau_end:.3f}")
        print_pair_rules_all(model, test_loader, tau_eval=args.tau_end, device=device)
        return

    # Loss/optim
    ce = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-4)

    best_acc = 0.0
    bad_epochs = 0
    last_eval_acc = None

    for epoch in range(1, args.epochs + 1):
        t = (epoch - 1) / max(1, args.epochs - 1)
        tau = args.tau_start + (args.tau_end - args.tau_start) * t

        # Train
        model.train()
        running_loss, seen = 0.0, 0
        for (x1, x2), y in train_loader:
            x1, x2 = x1.to(device), x2.to(device)
            y = y.to(device)
            # Optional external gating: switch to hard concepts when gate conditions are met
            use_hard = False
            if args.auto_refine:
                cond = True
                if args.refine_tau_gate is not None:
                    cond = cond and (tau <= args.refine_tau_gate)
                if args.refine_acc_gate is not None and last_eval_acc is not None:
                    cond = cond and (last_eval_acc >= float(args.refine_acc_gate))
                use_hard = cond
            y_prob, c_prob = model(x1, x2, tau=tau, hard=use_hard)
            logits = torch.logit(y_prob.clamp(1e-6, 1 - 1e-6))
            loss_main = ce(logits, y)
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

        # Eval
        model.eval()
        with torch.no_grad():
            tot, correct = 0, 0
            for (x1, x2), y in test_loader:
                x1, x2 = x1.to(device), x2.to(device)
                y = y.to(device)
                y_prob, _ = model(x1, x2, tau=args.tau_end, hard=True)
                pred = y_prob.argmax(1)
                correct += (pred == y).sum().item()
                tot += y.size(0)
            acc = correct / max(1, tot)

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | acc={acc*100:.2f}% | tau={tau:.3f}")
        last_eval_acc = acc

        if acc > best_acc:
            best_acc = acc
            bad_epochs = 0
            torch.save(model.state_dict(), "best_bacon_mnist_addition_multiclass.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping (no improvement for {args.patience} epochs).")
                break

    print(f"\nBest acc: {best_acc*100:.2f}% (checkpoint: best_bacon_mnist_addition_multiclass.pt)")
    print_pair_rules_all(model, test_loader, tau_eval=args.tau_end, device=device)


if __name__ == "__main__":
    main()


