#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os, sys, random, math
from typing import Iterator, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, IterableDataset, DataLoader
from torchvision import datasets, transforms

# Use local bacon package
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

from bacon.binaryTreeLogicNet import binaryTreeLogicNet
from bacon.aggregators.math.operator_set import OperatorSetAggregator


# ============================================================
# Datasets
# ============================================================

class MnistAdditionStream(IterableDataset):
    """
    Streaming dataset: each iteration yields a random pair of MNIST digits
    and the sum label in [0..18]. Used for training.
    """
    def __init__(self, root: str, train: bool, epoch_len: int, seed: int = 42):
        super().__init__()
        self.seed = seed
        self.epoch_len = epoch_len
        aug = [
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
        ] if train else []
        self.mnist = datasets.MNIST(
            root=root,
            train=train,
            download=True,
            transform=transforms.Compose(
                aug + [
                    transforms.ToTensor(),
                    transforms.Normalize((0.1307,), (0.3081,))
                ]
            ),
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


class MnistAdditionFixed(Dataset):
    """
    Fixed dataset: pre-sampled pairs for stable evaluation.
    """
    def __init__(self, root: str, train: bool, size_pairs: int, seed: int = 999):
        super().__init__()
        rng = random.Random(seed)
        self.mnist = datasets.MNIST(
            root=root,
            train=train,
            download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
            ]),
        )
        n = len(self.mnist)
        self.pairs: List = []
        for _ in range(size_pairs):
            i1 = rng.randrange(n)
            i2 = rng.randrange(n)
            _, d1 = self.mnist[i1]
            _, d2 = self.mnist[i2]
            self.pairs.append((i1, i2, int(d1 + d2)))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i1, i2, s = self.pairs[idx]
        x1, _ = self.mnist[i1]
        x2, _ = self.mnist[i2]
        return (x1, x2), s


# ============================================================
# CNN + Concept head + tiny BACON
# ============================================================

class SmallCnn(nn.Module):
    def __init__(self, out_features: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, out_features),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.conv(x))


class ImageConceptHead(nn.Module):
    """
    Maps CNN features to a 10-way (digit) concept distribution via Gumbel-softmax.
    """
    def __init__(self, feat_dim: int, n_concepts: int = 10):
        super().__init__()
        self.logit_head = nn.Linear(feat_dim, n_concepts)

    def forward(self, features: torch.Tensor, tau: float, hard: bool = False) -> torch.Tensor:
        logits = self.logit_head(features)  # [B,10]
        probs = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=1)
        return probs  # [B,10], sums to 1


class TinyBaconAdditionModel(nn.Module):
    """
    CNN -> per-image 10D concept probs -> expected digits
    -> normalized to [0,1] -> 2D input into a 2-input BACON with operator finder.

    BACON's output in [0,1] is interpreted as normalized sum (0..18)/18.
    """

    def __init__(self, freeze_cnn: bool = False, device=None):
        super().__init__()
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.cnn = SmallCnn(128)
        self.concept_head = ImageConceptHead(128, 10)

        if freeze_cnn:
            for p in self.cnn.parameters():
                p.requires_grad = False

        # Operator finder aggregator (arithmetic)
        op_agg = OperatorSetAggregator(
            kind="arith",
            op_names=["add", "sub", "mul", "div"],
            use_gumbel=True,
            tau=1.0,
        )

        # Tiny BACON with 2 inputs and a single internal node
        self.bacon = binaryTreeLogicNet(
            input_size=2,
            weight_mode="trainable",
            weight_normalization="minmax",
            aggregator=op_agg,
            normalize_andness=False,      # we don't really need AND-ness in pure arithmetic
            tree_layout="left",
            loss_amplifier=1.0,
            is_frozen=False,
        ).to(self.device)

        # We don't need Sinkhorn permutation for 2 inputs; but your existing
        # binaryTreeLogicNet will still create it. That's fine: it just learns
        # identity or swap.

        self.to(self.device)

    def _concept_digits(self, x: torch.Tensor, tau: float, hard: bool):
        """
        x: [B,1,28,28]
        Returns:
            digit_hat: [B], in [0,9], as expectation over digit probs.
            probs: [B,10]
        """
        z = self.cnn(x)                      # [B,128]
        p = self.concept_head(z, tau=tau, hard=hard)  # [B,10]
        digits = torch.arange(10, device=x.device).float()
        digit_hat = (p * digits.view(1, -1)).sum(dim=1)  # expectation
        return digit_hat, p

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, tau: float, hard: bool = False):
        """
        Returns:
            y_hat_norm: [B] in [0,1] (normalized sum)
            d1_hat, d2_hat: [B], per-image digit estimates (0..9)
        """
        x1 = x1.to(self.device)
        x2 = x2.to(self.device)

        d1_hat, p1 = self._concept_digits(x1, tau, hard=False)
        d2_hat, p2 = self._concept_digits(x2, tau, hard=False)

        # Normalize digits to [0,1] for the arithmetic aggregator
        d1_norm = d1_hat / 9.0
        d2_norm = d2_hat / 9.0

        pair = torch.stack([d1_norm, d2_norm], dim=1)  # [B,2]

        # BACON expects [B,input_size]
        y_bacon = self.bacon(pair).view(-1)  # [B]

        return y_bacon, d1_hat, d2_hat


# ============================================================
# Diagnostics
# ============================================================

def dump_operator_stats(model: TinyBaconAdditionModel):
    """
    Print operator selection weights (softmax over logits) per node.
    """
    agg = model.bacon.aggregator
    if not isinstance(agg, OperatorSetAggregator):
        return

    print("Operator weights:")
    if agg.op_logits_per_node is not None:
        for idx, logits in enumerate(agg.op_logits_per_node):
            w = F.softmax(logits.detach(), dim=0)
            weights_str = ", ".join(
                f"{name}={w[i]:.3f}" for i, name in enumerate(agg.op_names)
            )
            print(f"  node {idx}: {weights_str}")
    else:
        w = F.softmax(agg.op_logits_global.detach(), dim=0)
        weights_str = ", ".join(
            f"{name}={w[i]:.3f}" for i, name in enumerate(agg.op_names)
        )
        print(f"  global: {weights_str}")


# ============================================================
# Train / Eval
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="./data", type=str)
    ap.add_argument("--train-pairs", default=80000, type=int)
    ap.add_argument("--test-pairs", default=5000, type=int)
    ap.add_argument("--batch-size", default=128, type=int)
    ap.add_argument("--epochs", default=40, type=int)
    ap.add_argument("--lr", default=1e-3, type=float)
    ap.add_argument("--tau-start", default=3.0, type=float)
    ap.add_argument("--tau-end", default=0.6, type=float)
    ap.add_argument("--seed", default=0, type=int)
    ap.add_argument("--patience", default=10, type=int)
    ap.add_argument("--freeze-cnn", action="store_true")
    ap.add_argument("--save-path", default="tiny_bacon_add.pt", type=str)

    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Datasets
    train_ds = MnistAdditionStream(
        args.data, train=True, epoch_len=args.train_pairs, seed=args.seed
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=2)

    test_ds = MnistAdditionFixed(
        args.data, train=False, size_pairs=args.test_pairs, seed=999
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Model
    model = TinyBaconAdditionModel(
        freeze_cnn=args.freeze_cnn,
        device=device,
    )

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=1e-4
    )

    best_score = float("inf")   # we measure MAE/MSE, lower is better
    bad_epochs = 0

    for epoch in range(1, args.epochs + 1):
        t = (epoch - 1) / max(1, args.epochs - 1)
        tau = args.tau_start + (args.tau_end - args.tau_start) * t

        model.train()
        running_loss, seen = 0.0, 0

        for (x1, x2), y in train_loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device).float()

            # target normalized to [0,1]
            y_norm = y / 18.0

            y_hat_norm, d1_hat, d2_hat = model(x1, x2, tau=tau, hard=False)

            # regression loss on normalized sum
            loss_main = F.mse_loss(y_hat_norm, y_norm)

            opt.zero_grad(set_to_none=True)
            loss_main.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            bs = x1.size(0)
            running_loss += float(loss_main.item()) * bs
            seen += bs
            if seen >= args.train_pairs:
                break

        sched.step()
        train_loss = running_loss / max(1, seen)

        # small annealing for operator Gumbel temperature
        agg = model.bacon.aggregator
        if isinstance(agg, OperatorSetAggregator):
            agg.tau = max(0.1, agg.tau * 0.95)

        # ----- Eval -----
        model.eval()
        with torch.no_grad():
            se_sum, ae_sum, n_eval = 0.0, 0.0, 0
            for (x1, x2), y in test_loader:
                x1, x2, y = x1.to(device), x2.to(device), y.to(device).float()
                y_norm = y / 18.0
                y_hat_norm, _, _ = model(x1, x2, tau=args.tau_end, hard=True)
                se = (y_hat_norm - y_norm) ** 2
                ae = torch.abs(y_hat_norm - y_norm)
                se_sum += se.sum().item()
                ae_sum += ae.sum().item()
                n_eval += y.size(0)

            mse = se_sum / max(1, n_eval)
            mae = ae_sum / max(1, n_eval)

        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} "
            f"| MSE={mse:.4f} | MAE={mae:.4f} | tau={tau:.3f}"
        )
        dump_operator_stats(model)

        score = mse  # lower is better
        if score < best_score:
            best_score = score
            bad_epochs = 0
            torch.save(
                {"model_state": model.state_dict(), "args": vars(args)},
                args.save_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping (no improvement for {args.patience} epochs).")
                break

    print(f"\nBest MSE: {best_score:.4f} (saved to {args.save_path})")


if __name__ == "__main__":
    main()
