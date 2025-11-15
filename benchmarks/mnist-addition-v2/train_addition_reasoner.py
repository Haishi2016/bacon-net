#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generic trainer for MNIST-addition reasoning models that sit on top of
a shared digit classifier backbone.

Example (LTN):

    python train_addition_reasoner.py --reasoner ltn

Later you can plug other models via --reasoner dcr, --reasoner bacon, etc.
"""

import argparse
import os
import random

import numpy as np
import torch
import torch.nn as nn

from config import DATASET_CFG, TRAIN_CFG
from data import make_loaders
from digit_utils import load_digit_classifier, pair_digit_probs
from reasoners import REASONER_REGISTRY


def set_seed(seed: int):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reasoner", type=str, default="ltn",
                    help="Reasoner key in REASONER_REGISTRY (e.g., ltn)")
    ap.add_argument("--digit-ckpt", type=str,
                    default="checkpoints/digit_classifier_best.pt",
                    help="Path to shared DigitClassifier checkpoint")
    ap.add_argument("--data-root", type=str, default=DATASET_CFG.data_root)
    ap.add_argument("--train-pairs", type=int, default=DATASET_CFG.train_pairs)
    ap.add_argument("--test-pairs", type=int, default=DATASET_CFG.test_pairs)
    ap.add_argument("--batch-size", type=int, default=TRAIN_CFG.batch_size)
    ap.add_argument("--epochs", type=int, default=TRAIN_CFG.epochs)
    ap.add_argument("--lr", type=float, default=TRAIN_CFG.lr)
    ap.add_argument("--weight-decay", type=float, default=TRAIN_CFG.weight_decay)
    ap.add_argument("--seed", type=int, default=TRAIN_CFG.seed)
    ap.add_argument("--num-workers", type=int, default=TRAIN_CFG.num_workers)
    ap.add_argument("--no-pin-memory", action="store_true")
    ap.add_argument("--augment-train", action="store_true",
                    help="Use augmentation in MnistAdditionPairs train set")
    ap.add_argument("--checkpoint-dir", type=str, default="checkpoints")
    ap.add_argument("--train-digit", action="store_true",
                    help="Also train the digit classifier (DeepProbLog-style)")
    args = ap.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Data loaders (addition pairs) ---
    # Overwrite DATASET_CFG-like parameters temporarily
    from data import MnistAdditionPairs
    from torch.utils.data import DataLoader

    train_ds = MnistAdditionPairs(
        root=args.data_root,
        train=True,
        n_pairs=args.train_pairs,
        seed=DATASET_CFG.train_seed,
        augment=args.augment_train,
    )
    test_ds = MnistAdditionPairs(
        root=args.data_root,
        train=False,
        n_pairs=args.test_pairs,
        seed=DATASET_CFG.test_seed,
        augment=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
    )

    # --- Shared digit classifier (frozen) ---
    digit_model = load_digit_classifier(
        checkpoint_path=args.digit_ckpt,
        device=device,
        eval_mode=False,                 # we'll control train/eval
        freeze=not args.train_digit,     # unfreeze if we want to train
    )
    print(f"Loaded digit backbone from {args.digit_ckpt} "
        f"({'trainable' if args.train_digit else 'frozen'})")

    # --- Reasoner selection ---
    if args.reasoner not in REASONER_REGISTRY:
        raise ValueError(f"Unknown reasoner '{args.reasoner}'. "
                         f"Available: {list(REASONER_REGISTRY.keys())}")

    ReasonerCls = REASONER_REGISTRY[args.reasoner]
    reasoner = ReasonerCls().to(device)
    print(f"Instantiated reasoner: {reasoner.__class__.__name__}")

    # Collect trainable parameters
    reasoner_params = [p for p in reasoner.parameters() if p.requires_grad]

    if args.train_digit:
        digit_params = [p for p in digit_model.parameters() if p.requires_grad]
    else:
        digit_params = []

    trainable_params = reasoner_params + digit_params

    if len(trainable_params) == 0:
        print("Reasoner has no trainable parameters; will run evaluation only.")
        optimizer = None
    else:
        # Train ALL params in reasoner, digit_model stays frozen
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

    ckpt_dir = args.checkpoint_dir
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f"{args.reasoner}_addition_best.pt")

    best_acc = 0.0

    if optimizer is None:
        # -------- Evaluation-only path (e.g., LTN with no params) --------
        reasoner.eval()
        digit_model.eval()

        correct, total = 0, 0
        with torch.no_grad():
            for (x1, x2), y in test_loader:
                x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                p1, feat1, p2, feat2 = pair_digit_probs(digit_model, x1, x2)
                y_hat = reasoner.predict(p1, p2)
                correct += (y_hat == y).sum().item()
                total += y.size(0)
        test_acc = correct / max(1, total)
        print(f"[Eval-only] {args.reasoner} test_acc: {test_acc*100:.2f}%")

        if args.reasoner == "ltn":
            print("\nRule view from LTN reasoner:")
            reasoner.print_pair_rules(test_loader, digit_model, device)

    else:
        # -------- Normal train + eval loop for trainable reasoners --------
        for epoch in range(1, args.epochs + 1):
            # --- Train ---
            reasoner.train()
            digit_model.eval()  # just in case

            running_loss = 0.0
            seen = 0

            for (x1, x2), y in train_loader:
                x1, x2, y = x1.to(device), x2.to(device), y.to(device)

                # Shared backbone → digit probabilities
                p1, feat1, p2, feat2 = pair_digit_probs(digit_model, x1, x2)

                # Reasoner loss (e.g., LTN, DCR, BACON, etc.)
                loss = reasoner.loss(p1, p2, y)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(reasoner.parameters(), 1.0)
                optimizer.step()

                bs = x1.size(0)
                running_loss += float(loss.item()) * bs
                seen += bs

            train_loss = running_loss / max(1, seen)

            # --- Eval ---
            reasoner.eval()
            digit_model.eval()

            correct, total = 0, 0
            with torch.no_grad():
                for (x1, x2), y in test_loader:
                    x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                    p1, feat1, p2, feat2 = pair_digit_probs(digit_model, x1, x2)
                    y_hat = reasoner.predict(p1, p2)
                    correct += (y_hat == y).sum().item()
                    total += y.size(0)
            test_acc = correct / max(1, total)

            print(f"Epoch {epoch:03d} | "
                f"train_loss={train_loss:.4f} | test_acc={test_acc*100:.2f}%")

            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(
                    {
                        "reasoner_state_dict": reasoner.state_dict(),
                        "reasoner": args.reasoner,
                        "config": {
                            "lr": args.lr,
                            "weight_decay": args.weight_decay,
                            "seed": args.seed,
                        },
                    },
                    ckpt_path,
                )
                print(f"  -> New best test_acc={best_acc*100:.2f}%, "
                    f"saved to {ckpt_path}")

        print(f"\nBest {args.reasoner} test_acc: {best_acc*100:.2f}% "
            f"(checkpoint: {ckpt_path})")

        if args.reasoner == "ltn":
            print("\nFinal rule view from LTN reasoner:")
            reasoner.print_pair_rules(test_loader, digit_model, device)



if __name__ == "__main__":
    main()
