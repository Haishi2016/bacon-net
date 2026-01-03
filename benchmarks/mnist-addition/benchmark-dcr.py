#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, math, random, os, sys
from typing import Iterator, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, IterableDataset, DataLoader
from torchvision import datasets, transforms

import torch_explain as te
from torch_explain.nn.concepts import ConceptReasoningLayer


# ------------------------------------------------------------
# Streaming Dataset (same as BACON v1)
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
# Fixed Dataset (same as BACON v1)
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
# Small CNN tower (same as BACON v1)
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
# DCR concept head per image: embeddings + Gumbel probs
# ------------------------------------------------------------
class ImageConceptHeadDCR(nn.Module):
    def __init__(self, feat_dim: int, n_concepts=10, emb_size=30):
        super().__init__()
        self.embed = te.nn.ConceptEmbedding(feat_dim, n_concepts, emb_size)
        self.logit_head = nn.Linear(feat_dim, n_concepts)

    def forward(self, features, tau: float, hard: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        # c_emb: [B,10,emb_size], _ are concept activations we don't use explicitly
        c_emb, _ = self.embed(features)
        logits = self.logit_head(features)  # [B,10]
        probs = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=1)  # [B,10]
        return c_emb, probs


# ------------------------------------------------------------
# DCR head (19-class addition)
# ------------------------------------------------------------
class DCRMultiHead(nn.Module):
    def __init__(self, emb_size=30):
        super().__init__()
        self.tower1 = SmallCnn(128)
        self.tower2 = SmallCnn(128)
        self.head1 = ImageConceptHeadDCR(128, 10, emb_size)
        self.head2 = ImageConceptHeadDCR(128, 10, emb_size)

        # Concept reasoning over 20 concepts (10+10) into 19 classes (0..18)
        self.core = ConceptReasoningLayer(emb_size, n_classes=19)

        # Auxiliary head on concept probabilities (like v1 DCR)
        self.aux = nn.Sequential(
            nn.Linear(20, 30), nn.ReLU(),
            nn.Linear(30, 19)
        )

    def forward(self, x1, x2, tau: float, hard: bool = False):
        z1 = self.tower1(x1)
        z2 = self.tower2(x2)

        emb1, p1 = self.head1(z1, tau=tau, hard=hard)  # [B,10,emb], [B,10]
        emb2, p2 = self.head2(z2, tau=tau, hard=hard)  # [B,10,emb], [B,10]

        c_emb = torch.cat([emb1, emb2], dim=1)         # [B,20,emb]
        c_prob = torch.cat([p1, p2], dim=1)            # [B,20]

        y_pred = self.core(c_emb, c_prob)              # [B,19] probabilities
        y_aux = self.aux(c_prob)                       # [B,19] logits

        return y_pred, y_aux, c_emb, c_prob


# ------------------------------------------------------------
# Utils
# ------------------------------------------------------------
def group_entropy_loss(c_prob):
    eps = 1e-8
    g1 = c_prob[:, :10]
    g2 = c_prob[:, 10:]
    ent = -(g1 * (g1 + eps).log()).sum(dim=1) - (g2 * (g2 + eps).log()).sum(dim=1)
    return (ent / math.log(10)).mean()


def triangle_class_weights():
    """
    Analytic counts for sums: #{(i,j): i+j=k} = min(k,18-k)+1, normalized inverse.
    """
    counts = np.array([min(k, 18 - k) + 1 for k in range(19)], dtype=np.float32)
    inv = 1.0 / counts
    w = inv / inv.mean()
    return torch.tensor(w, dtype=torch.float32)


def print_pair_rules_dcr(model, test_loader, tau_eval=0.8, device='cpu'):
    """
    Optional: compute pairwise digit pairs (i,j) contributing to each sum k,
    using hard concepts at eval, like in your DCR v1 script.
    """
    model.eval()
    pair_sum = torch.zeros(19, 10, 10, device=device)
    count_k = torch.zeros(19, device=device)
    with torch.no_grad():
        for (x1, x2), _ in test_loader:
            x1, x2 = x1.to(device), x2.to(device)
            # Get hard one-hot concepts
            z1 = model.tower1(x1); z2 = model.tower2(x2)
            _, p1 = model.head1(z1, tau=tau_eval, hard=True)  # [B,10]
            _, p2 = model.head2(z2, tau=tau_eval, hard=True)  # [B,10]

            # Predict sums
            y_pred, _, _, _ = model(x1, x2, tau=tau_eval, hard=True)
            k = y_pred.argmax(1)  # [B]

            outer = p1.unsqueeze(2) * p2.unsqueeze(1)  # [B,10,10]
            for kk in range(19):
                mask = (k == kk).float().view(-1, 1, 1)
                pair_sum[kk] += (outer * mask).sum(dim=0)
                count_k[kk]  += mask.sum()

    print("\n=== DCR Pairwise rules (valid pairs i∧j with i+j=k) ===")
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
# Train / Eval (DCR)
# ------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data", type=str)
    ap.add_argument("--train-pairs", default=80000, type=int)
    ap.add_argument("--test-pairs", default=5000, type=int)
    ap.add_argument("--batch-size", default=128, type=int)
    ap.add_argument("--epochs", default=120, type=int)
    ap.add_argument("--lr", default=1e-3, type=float)
    ap.add_argument("--emb", default=30, type=int)
    ap.add_argument("--aux", default=1.0, type=float, help="aux CE weight")
    ap.add_argument("--entropy", default=0.0005, type=float)
    ap.add_argument("--tau-start", default=3.0, type=float)
    ap.add_argument("--tau-end", default=0.6, type=float)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--patience", default=15, type=int)
    ap.add_argument("--pretrained", default=None, type=str)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = MnistAdditionStream(args.data, train=True, epoch_len=args.train_pairs, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=2)

    test_ds = MnistAdditionFixed(args.data, train=False, size_pairs=args.test_pairs, seed=999)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = DCRMultiHead(emb_size=args.emb).to(device)

    # Optional: load pretrained DCR weights
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
        elif hasattr(ckpt, "state_dict"):
            state_dict = ckpt.state_dict()
        else:
            state_dict = ckpt

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
                y_pred, _, _, _ = model(x1, x2, tau=args.tau_end, hard=True)
                pred = y_pred.argmax(1)
                correct += (pred == y).sum().item()
                tot += y.size(0)
            acc = correct / max(1, tot)
        print(f"Pretrained eval | DCR acc={acc*100:.2f}% | tau={args.tau_end:.3f}")
        print_pair_rules_dcr(model, test_loader, tau_eval=args.tau_end, device=device)
        return

    # Class weights for imbalanced sums
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

        # ---- Train ----
        model.train()
        running_loss, seen = 0.0, 0

        for (x1, x2), y in train_loader:
            x1, x2 = x1.to(device), x2.to(device)
            y = y.to(device)

            y_pred, y_aux, _, c_prob = model(x1, x2, tau=tau, hard=False)

            # main CE on y_pred (probabilities -> logits)
            main_logits = torch.logit(y_pred.clamp(1e-6, 1 - 1e-6))
            loss_main = ce(main_logits, y)

            # auxiliary CE on y_aux logits
            loss_aux = ce(y_aux, y)

            # entropy regularizer over concepts
            loss_ent = group_entropy_loss(c_prob) * args.entropy

            loss = loss_main + args.aux * loss_aux + loss_ent

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
                y_pred, _, _, _ = model(x1, x2, tau=args.tau_end, hard=True)
                pred = y_pred.argmax(1)
                correct += (pred == y).sum().item()
                tot += y.size(0)
            acc = correct / tot

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | DCR_acc={acc*100:.2f}% | tau={tau:.3f}")

        # Early stopping
        if acc > best_acc:
            best_acc = acc
            bad_epochs = 0
            torch.save(model.state_dict(), "best_dcr_mnist_addition_from_bacon.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping (no improvement for {args.patience} epochs).")
                break

    print(f"\nBest DCR acc: {best_acc*100:.2f}% (checkpoint: best_dcr_mnist_addition_from_bacon.pt)")
    print_pair_rules_dcr(model, test_loader, tau_eval=args.tau_end, device=device)


if __name__ == "__main__":
    main()