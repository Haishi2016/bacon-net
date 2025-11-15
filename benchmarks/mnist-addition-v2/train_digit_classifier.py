#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Train a shared DigitClassifier on MNIST digits.

This is the common backbone that all reasoning methods (BACON, DCR, LTN,
DeepProbLog, DeepStochLog, etc.) can reuse.

Usage (basic):

    python train_digit_classifier.py

Optional overrides:

    python train_digit_classifier.py --epochs 20 --lr 5e-4 --augment

Checkpoint:

    checkpoints/digit_classifier_best.pt
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from config import DATASET_CFG, TRAIN_CFG, BACKBONE_CFG
from backbone import DigitClassifier


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_mnist_loaders(
    data_root: str,
    batch_size: int,
    num_workers: int = 2,
    pin_memory: bool = True,
    augment: bool = False,
):
    """
    Return DataLoaders for standard MNIST digit classification.
    """

    base_transform = [
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ]

    if augment:
        aug = [
            transforms.RandomAffine(degrees=10, translate=(0.05, 0.05)),
        ]
        t_train = transforms.Compose(aug + base_transform)
    else:
        t_train = transforms.Compose(base_transform)

    t_test = transforms.Compose(base_transform)

    train_ds = datasets.MNIST(
        root=data_root,
        train=True,
        download=True,
        transform=t_train,
    )
    test_ds = datasets.MNIST(
        root=data_root,
        train=False,
        download=True,
        transform=t_test,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader


def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return correct / max(1, total)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", type=str, default=DATASET_CFG.data_root,
                    help="MNIST data root")
    ap.add_argument("--batch-size", type=int, default=TRAIN_CFG.batch_size)
    ap.add_argument("--epochs", type=int, default=TRAIN_CFG.epochs)
    ap.add_argument("--lr", type=float, default=TRAIN_CFG.lr)
    ap.add_argument("--weight-decay", type=float, default=TRAIN_CFG.weight_decay)
    ap.add_argument("--seed", type=int, default=TRAIN_CFG.seed)
    ap.add_argument("--num-workers", type=int, default=TRAIN_CFG.num_workers)
    ap.add_argument("--no-pin-memory", action="store_true",
                    help="Disable pin_memory in DataLoader")
    ap.add_argument("--augment", action="store_true",
                    help="Use simple affine augmentation for training")
    ap.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    ap.add_argument("--feat-dim", type=int, default=BACKBONE_CFG.feat_dim)
    args = ap.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    ckpt_path = os.path.join(args.checkpoint_dir, "digit_classifier_best.pt")

    train_loader, test_loader = get_mnist_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
        augment=args.augment,
    )

    model = DigitClassifier(feat_dim=args.feat_dim).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            logits, _ = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            bs = x.size(0)
            running_loss += float(loss.item()) * bs
            seen += bs

        train_loss = running_loss / max(1, seen)
        test_acc = evaluate(model, test_loader, device)

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
              f"test_acc={test_acc*100:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "feat_dim": args.feat_dim,
                    "config": {
                        "lr": args.lr,
                        "weight_decay": args.weight_decay,
                        "seed": args.seed,
                        "augment": args.augment,
                    },
                },
                ckpt_path,
            )
            print(f"  -> New best test_acc={best_acc*100:.2f}%, saved to {ckpt_path}")

    print(f"\nBest test_acc: {best_acc*100:.2f}% "
          f"(checkpoint: {ckpt_path})")


if __name__ == "__main__":
    main()
