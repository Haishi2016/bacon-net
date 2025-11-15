# mnist-addition-v3/train_addition.py

import argparse
import random
from typing import Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from backbone import DeepProbLogMNISTNet
from datasets import MNISTAdditionPairs
from reasoners import REASONER_REGISTRY


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="./data",
                    help="Root folder for MNIST data")
    ap.add_argument("--reasoner", type=str, default="deepproblog",
                    choices=REASONER_REGISTRY.keys())
    ap.add_argument("--train-pairs", type=int, default=30000,
                    help="Number of MNIST addition pairs for training")
    ap.add_argument("--test-pairs", type=int, default=5000,
                    help="Number of MNIST addition pairs for testing")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-cuda", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    use_cuda = (not args.no_cuda) and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    # ---------------- Data ----------------
    train_ds = MNISTAdditionPairs(
        root=args.data,
        train=True,
        n_pairs=args.train_pairs,
        seed=args.seed,
        download=True,
    )
    test_ds = MNISTAdditionPairs(
        root=args.data,
        train=False,
        n_pairs=args.test_pairs,
        seed=999,  # fixed different seed for test
        download=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=use_cuda,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=use_cuda,
    )

    # ---------------- Model ----------------
    # Digit CNN backbone (DeepProbLog paper architecture)
    digit_net = DeepProbLogMNISTNet(n_outputs=10).to(device)

    # Reasoner
    ReasonerCls = REASONER_REGISTRY[args.reasoner]
    reasoner = ReasonerCls().to(device)
    print(f"Instantiated reasoner: {reasoner.__class__.__name__}")

    # Joint optimizer: CNN + reasoner
    params = list(digit_net.parameters()) + list(reasoner.parameters())
    optimizer = torch.optim.Adam(params, lr=args.lr)

    # ---------------- Training loop ----------------
    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        digit_net.train()
        reasoner.train()
        running_loss = 0.0
        seen = 0

        for (x1, x2), y_sum in train_loader:
            x1 = x1.to(device)
            x2 = x2.to(device)
            y_sum = y_sum.to(device)  # [B]

            # CNN: digit logits → digit probabilities
            logits1 = digit_net(x1)              # [B,10]
            logits2 = digit_net(x2)              # [B,10]
            p1 = F.softmax(logits1, dim=1)       # [B,10]
            p2 = F.softmax(logits2, dim=1)       # [B,10]

            # DeepProbLog-style reasoning loss
            loss = reasoner.loss(p1, p2, y_sum)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            bs = x1.size(0)
            running_loss += float(loss.item()) * bs
            seen += bs

        train_loss = running_loss / max(1, seen)

        # ---------------- Eval ----------------
        digit_net.eval()
        reasoner.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for (x1, x2), y_sum in test_loader:
                x1 = x1.to(device)
                x2 = x2.to(device)
                y_sum = y_sum.to(device)

                logits1 = digit_net(x1)
                logits2 = digit_net(x2)
                p1 = F.softmax(logits1, dim=1)
                p2 = F.softmax(logits2, dim=1)

                y_pred = reasoner.predict(p1, p2)
                correct += (y_pred == y_sum).sum().item()
                total += y_sum.size(0)

        test_acc = correct / max(1, total)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} | "
            f"test_acc={test_acc*100:.2f}%"
        )

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(
                {
                    "digit_net": digit_net.state_dict(),
                    "reasoner": reasoner.state_dict(),
                    "args": vars(args),
                    "epoch": epoch,
                    "best_acc": best_acc,
                },
                "mnist_addition_v3_deepproblog_best.pt",
            )
            print(
                f"  -> New best test_acc={best_acc*100:.2f}%, "
                f"saved to mnist_addition_v3_deepproblog_best.pt"
            )

    print(f"\nBest test_acc: {best_acc*100:.2f}%\n")


if __name__ == "__main__":
    main()
