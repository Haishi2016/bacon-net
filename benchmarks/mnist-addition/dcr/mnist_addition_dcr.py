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

import torch_explain as te
from torch_explain.nn.concepts import ConceptReasoningLayer


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
        # Different RNG per worker + epoch if available
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
# Concept head per image: ConceptEmbedding + Gumbel-Softmax
# ------------------------------------------------------------
class ImageConceptHead(nn.Module):
    def __init__(self, feat_dim: int, n_concepts=10, emb_size=30):
        super().__init__()
        self.embed = te.nn.ConceptEmbedding(feat_dim, n_concepts, emb_size)
        self.logit_head = nn.Linear(feat_dim, n_concepts)

    def forward(self, features, tau: float, hard: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        c_emb, _ = self.embed(features)                  # [B,10,emb]
        logits = self.logit_head(features)               # [B,10]
        probs = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=1)  # [B,10], sum=1
        return c_emb, probs


# ------------------------------------------------------------
# Full DCR model
# ------------------------------------------------------------
class DCRAddition(nn.Module):
    def __init__(self, emb_size=30):
        super().__init__()
        self.tower1 = SmallCnn(128)
        self.tower2 = SmallCnn(128)
        self.head1 = ImageConceptHead(128, 10, emb_size)
        self.head2 = ImageConceptHead(128, 10, emb_size)
        self.core = ConceptReasoningLayer(emb_size, n_classes=19)  # outputs probs in [0,1]
        # Auxiliary head on concept probs (logits for CE)
        self.aux = nn.Sequential(
            nn.Linear(20, 30), nn.ReLU(),
            nn.Linear(30, 19)  # logits
        )

    def forward(self, x1, x2, tau: float, hard: bool = False):
        z1 = self.tower1(x1)
        z2 = self.tower2(x2)
        emb1, p1 = self.head1(z1, tau=tau, hard=hard)   # [B,10,emb], [B,10]
        emb2, p2 = self.head2(z2, tau=tau, hard=hard)   # [B,10,emb], [B,10]
        c_emb = torch.cat([emb1, emb2], dim=1)          # [B,20,emb]
        c_prob = torch.cat([p1, p2], dim=1)             # [B,20]
        y_pred = self.core(c_emb, c_prob)               # [B,19] probs
        y_aux = self.aux(c_prob)                        # [B,19] logits
        return y_pred, y_aux, c_emb, c_prob


# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------
def group_entropy_loss(c_prob):
    """Entropy penalty per 10-way group (normalize by log(10))."""
    eps = 1e-8
    g1 = c_prob[:, :10]
    g2 = c_prob[:, 10:]
    ent = -(g1 * (g1 + eps).log()).sum(dim=1) - (g2 * (g2 + eps).log()).sum(dim=1)
    return (ent / math.log(10)).mean()


def triangle_class_weights():
    """Analytic counts for sums: #{(i,j): i+j=k} = min(k,18-k)+1, normalized inverse."""
    counts = np.array([min(k, 18 - k) + 1 for k in range(19)], dtype=np.float32)
    inv = 1.0 / counts
    w = inv / inv.mean()
    return torch.tensor(w, dtype=torch.float32)


# ------------------------------------------------------------
# Pairwise rule printer (clean OR-of-ANDs per sum)
# ------------------------------------------------------------
def print_pair_rules(model, test_loader, tau_eval=0.8, device='cpu'):
    model.eval()
    pair_sum = torch.zeros(19, 10, 10)
    count_k = torch.zeros(19)
    with torch.no_grad():
        for (x1, x2), _ in test_loader:
            x1, x2 = x1.to(device), x2.to(device)
            # hard one-hot concepts at eval
            z1 = model.tower1(x1); z2 = model.tower2(x2)
            _, p1 = model.head1(z1, tau=tau_eval, hard=True)  # [B,10]
            _, p2 = model.head2(z2, tau=tau_eval, hard=True)  # [B,10]
            # choose predicted class to attribute pairs
            y_pred, _, _, _ = model(x1, x2, tau=tau_eval, hard=True)
            k = y_pred.argmax(1)  # [B]
            outer = p1.unsqueeze(2) * p2.unsqueeze(1)  # [B,10,10]
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
    ap.add_argument("--train-pairs", default=80000, type=int, help="pairs per epoch (streamed)")
    ap.add_argument("--test-pairs", default=5000, type=int)
    ap.add_argument("--batch-size", default=128, type=int)
    ap.add_argument("--epochs", default=120, type=int)
    ap.add_argument("--lr", default=1e-3, type=float)
    ap.add_argument("--emb", default=30, type=int)
    ap.add_argument("--aux", default=1.0, type=float, help="aux CE weight")
    ap.add_argument("--entropy", default=0.0005, type=float, help="tiny entropy kept always")
    ap.add_argument("--tau-start", default=2.5, type=float)
    ap.add_argument("--tau-end", default=0.8, type=float)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--patience", default=15, type=int)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Streaming train, fixed test
    train_ds = MnistAdditionStream(args.data, train=True, epoch_len=args.train_pairs, seed=args.seed)
    test_ds = MnistAdditionFixed(args.data, train=False, size_pairs=args.test_pairs, seed=999)

    # For IterableDataset, don't set shuffle=True
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = DCRAddition(emb_size=args.emb).to(device)

    # Triangular class weights
    class_weights = triangle_class_weights().to(device)
    ce = nn.CrossEntropyLoss(weight=class_weights)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-4)

    best_acc = 0.0
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        # τ linear annealing
        t = (epoch - 1) / max(1, args.epochs - 1)
        tau = args.tau_start + (args.tau_end - args.tau_start) * t

        # ---- Train one epoch over ~train_pairs samples ----
        model.train()
        running_loss = 0.0
        seen = 0

        for (x1, x2), y in train_loader:
            x1, x2 = x1.to(device), x2.to(device)
            y = y.to(device)

            y_pred, y_aux, _, c_prob = model(x1, x2, tau=tau, hard=False)

            # main CE on y_pred (convert probs->logits)
            main_logits = torch.logit(y_pred.clamp(1e-6, 1 - 1e-6))
            loss_main = ce(main_logits, y)

            # aux CE on logits
            loss_aux = ce(y_aux, y)

            # constant tiny entropy to prevent brittle collapse
            loss_ent = group_entropy_loss(c_prob) * args.entropy

            loss = loss_main + args.aux * loss_aux + loss_ent

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bs = x1.size(0)
            running_loss += float(loss.item()) * bs
            seen += bs
            # Stop epoch once we've produced epoch_len samples (IterableDataset keeps going otherwise)
            if seen >= args.train_pairs:
                break

        sched.step()

        train_loss = running_loss / max(1, seen)

        # ---- Eval (hard one-hots, slightly warm tau) ----
        model.eval()
        with torch.no_grad():
            tot, correct = 0, 0
            for (x1, x2), y in test_loader:
                x1, x2 = x1.to(device), x2.to(device)
                y = y.to(device)
                y_pred, _, _, _ = model(x1, x2, tau=args.tau_end, hard=True)
                correct += (y_pred.argmax(1) == y).sum().item()
                tot += y.size(0)
            acc = correct / tot

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | test_acc={acc*100:.2f}% | tau={tau:.3f}")

        # Early stopping on best test acc
        if acc > best_acc:
            best_acc = acc
            bad_epochs = 0
            torch.save(model.state_dict(), "best_dcr_mnist_addition.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping (no improvement for {args.patience} epochs).")
                break

    print(f"\nBest test_acc: {best_acc*100:.2f}% (checkpoint: best_dcr_mnist_addition.pt)")

    # Pairwise rule view (clean)
    print_pair_rules(model, test_loader, tau_eval=args.tau_end, device=device)


if __name__ == "__main__":
    main()
