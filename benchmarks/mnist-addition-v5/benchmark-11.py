#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import os
import random
import sys
from typing import Iterator, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, IterableDataset, DataLoader
from torchvision import datasets, transforms

import matplotlib.pyplot as plt

# Ensure local bacon package is used instead of any installed site-package
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
    and (d1, d2, sum). Used for training.
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

    def __iter__(self) -> Iterator[Tuple[Tuple[torch.Tensor, torch.Tensor],
                                         Tuple[int, int, int]]]:
        worker_info = torch.utils.data.get_worker_info()
        base = self.seed + (0 if worker_info is None else worker_info.id)
        rng = random.Random(base)
        n = len(self.mnist)
        for _ in range(self.epoch_len):
            i1 = rng.randrange(n)
            i2 = rng.randrange(n)
            x1, d1 = self.mnist[i1]
            x2, d2 = self.mnist[i2]
            s = int(d1 + d2)
            yield (x1, x2), (int(d1), int(d2), s)


class MnistAdditionFixed(Dataset):
    """
    Fixed dataset: pre-sampled pairs for stable evaluation.
    Returns ((x1, x2), (d1, d2, sum)).
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
        self.pairs: List[Tuple[int, int, int, int, int]] = []
        for _ in range(size_pairs):
            i1 = rng.randrange(n)
            i2 = rng.randrange(n)
            _, d1 = self.mnist[i1]
            _, d2 = self.mnist[i2]
            s = int(d1 + d2)
            self.pairs.append((i1, i2, int(d1), int(d2), s))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i1, i2, d1, d2, s = self.pairs[idx]
        x1, _ = self.mnist[i1]
        x2, _ = self.mnist[i2]
        return (x1, x2), (d1, d2, s)


# ============================================================
# CNN + Concept Head
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


# ============================================================
# BACON-based model with operator finder
# ============================================================

class BaconAdditionWithPerception(nn.Module):
    """
    End-to-end model:

      (x1, x2) -> CNN -> digit distributions p1, p2 over digits 0..9
                  -> expected digits d1_hat, d2_hat
                  -> normalize -> [d1_norm, d2_norm]
                  -> BACON + OperatorSetAggregator (2 inputs, 1 internal node)
                  -> y_hat_norm in [0,1] approximating (d1+d2)/18
    """

    def __init__(self, freeze_cnn: bool = False):
        super().__init__()
        self.cnn = SmallCnn(128)
        self.concept_head = ImageConceptHead(128, 10)
        self.register_buffer("digit_values", torch.arange(10).float())

        if freeze_cnn:
            for p in self.cnn.parameters():
                p.requires_grad = False

        # Operator-set aggregator (add, sub, mul, div)
        agg = OperatorSetAggregator(
            kind="arith",
            op_names=["add", "sub", "mul", "div"],
            use_gumbel=True,
            auto_lock=True,
            lock_threshold=0.995,
            tau=1.0,
        )

        # Tiny BACON: 2 inputs (digit1_norm, digit2_norm)
        self.bacon = binaryTreeLogicNet(
            input_size=2,
            weight_mode="fixed",
            weight_normalization="minmax",
            aggregator=agg,
            normalize_andness=False,   # no graded AND/OR needed here
            tree_layout="left",
            loss_amplifier=1.0,
            is_frozen=False,
        )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, tau: float, hard: bool = False):
        """
        Returns:
            y_hat_norm: [B]  (predicted normalized sum in [0,1])
            p1: [B,10]       (digit distribution for x1)
            p2: [B,10]       (digit distribution for x2)
            d1_hat: [B]      (expected digit value from p1)
            d2_hat: [B]      (expected digit value from p2)
        """
        device = x1.device
        digit_vals = self.digit_values.to(device)  # [10]

        # CNN features
        z1 = self.cnn(x1)  # [B,128]
        z2 = self.cnn(x2)  # [B,128]

        # Digit distributions
        p1 = self.concept_head(z1, tau=tau, hard=hard)  # [B,10]
        p2 = self.concept_head(z2, tau=tau, hard=hard)  # [B,10]

        # Expected digits in [0,9]
        d1_hat = (p1 * digit_vals).sum(dim=1)  # [B]
        d2_hat = (p2 * digit_vals).sum(dim=1)  # [B]

        # Normalize to [0,1]
        d1_norm = (d1_hat / 9.0).unsqueeze(1)  # [B,1]
        d2_norm = (d2_hat / 9.0).unsqueeze(1)  # [B,1]

        bacon_in = torch.cat([d1_norm, d2_norm], dim=1)  # [B,2]

        y_hat = self.bacon(bacon_in).view(-1)            # [B]
        return y_hat, p1, p2, d1_hat, d2_hat


def dump_operator_stats(model: BaconAdditionWithPerception):
    """
    Print operator selection weights (softmax over logits) per node.
    """
    agg = model.bacon.aggregator
    if not isinstance(agg, OperatorSetAggregator):
        return

    print("Operator weights:")
    if getattr(agg, "op_logits_per_node", None) is not None:
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
# Loss helpers & diagnostics
# ============================================================

def entropy_loss_digit(p: torch.Tensor) -> torch.Tensor:
    """
    Entropy of digit distributions, normalized to [0,1] by log(10).
    """
    eps = 1e-8
    ent = -(p * (p + eps).log()).sum(dim=1)  # [B]
    return (ent / math.log(p.size(1))).mean()


def evaluate_and_print(
    model: BaconAdditionWithPerception,
    loader: DataLoader,
    tau_eval: float,
    device: torch.device,
    prefix: str = "Eval",
    print_ops: bool = True,
):
    """
    Evaluate:
      - MSE / MAE on sums (0..18)
      - Sum accuracy (rounded)
      - Digit1 and Digit2 accuracy (rounded)
      - Operator weights for each BACON node (if available).

    Returns:
      (mse, mae, acc_sum, acc_d1, acc_d2)
    """
    model.eval()
    sse = 0.0
    sae = 0.0
    tot = 0

    correct_sum = 0
    correct_d1 = 0
    correct_d2 = 0

    with torch.no_grad():
        for (x1, x2), (d1, d2, s) in loader:
            x1 = x1.to(device)
            x2 = x2.to(device)
            d1 = d1.to(device).long()
            d2 = d2.to(device).long()
            s = s.to(device).long()

            y_hat_norm, p1, p2, d1_hat, d2_hat = model(x1, x2, tau=tau_eval, hard=True)

            # Sum side
            s_hat = (y_hat_norm * 18.0).clamp(0.0, 18.0).round().long()
            diff = s_hat.float() - s.float()

            batch_size = s.size(0)
            sse += (diff ** 2).sum().item()
            sae += diff.abs().sum().item()
            tot += batch_size
            correct_sum += (s_hat == s).sum().item()

            # Digit side
            d1_pred = d1_hat.clamp(0.0, 9.0).round().long()
            d2_pred = d2_hat.clamp(0.0, 9.0).round().long()
            correct_d1 += (d1_pred == d1).sum().item()
            correct_d2 += (d2_pred == d2).sum().item()

    mse = sse / max(1, tot)
    mae = sae / max(1, tot)
    acc_sum = correct_sum / max(1, tot)
    acc_d1 = correct_d1 / max(1, tot)
    acc_d2 = correct_d2 / max(1, tot)

    print(
        f"{prefix} | MSE={mse:.4f} | MAE={mae:.4f} | "
        f"acc_sum={acc_sum*100:.2f}% | acc_d1={acc_d1*100:.2f}% | "
        f"acc_d2={acc_d2*100:.2f}%"
    )

    # (Optional) older-style operator print; you now also have dump_operator_stats
    if print_ops:
        agg = getattr(model.bacon, "aggregator", None)
        if agg is not None and hasattr(agg, "op_logits"):
            with torch.no_grad():
                op_logits = agg.op_logits  # [num_nodes, num_ops]
                op_probs = F.softmax(op_logits, dim=-1)
                if hasattr(agg, "op_names"):
                    op_names = agg.op_names
                else:
                    op_names = [f"op{i}" for i in range(op_probs.size(1))]

                print("Operator weights:")
                for i in range(op_probs.size(0)):
                    probs_i = op_probs[i]
                    parts = [f"{op_names[j]}={probs_i[j].item():.3f}" for j in range(len(op_names))]
                    print(f"  node {i}: " + ", ".join(parts))

    return mse, mae, acc_sum, acc_d1, acc_d2


# ============================================================
# Visualization helpers
# ============================================================

def _denorm_mnist(x: torch.Tensor) -> torch.Tensor:
    """
    Reverse the MNIST normalization: x_norm = (x - 0.1307) / 0.3081
    So x = x_norm * 0.3081 + 0.1307
    """
    mean = 0.1307
    std = 0.3081
    return torch.clamp(x * std + mean, 0.0, 1.0)


def save_sample_grid(
    model: BaconAdditionWithPerception,
    loader: DataLoader,
    device: torch.device,
    tau_eval: float,
    filename: str,
    max_samples: int = 32,
    only_errors: bool = False,
):
    """
    Save a grid of sample pairs.

    If only_errors=True, saves only mispredicted sum pairs (up to max_samples).
    Otherwise just takes the first max_samples from the loader.
    """
    model.eval()

    imgs = []
    titles = []

    with torch.no_grad():
        for (x1, x2), (d1, d2, s) in loader:
            x1 = x1.to(device)
            x2 = x2.to(device)
            d1 = d1.to(device).long()
            d2 = d2.to(device).long()
            s = s.to(device).long()

            y_hat_norm, _, _, d1_hat, d2_hat = model(x1, x2, tau=tau_eval, hard=True)
            s_hat = (y_hat_norm * 18.0).clamp(0.0, 18.0).round().long()
            d1_pred = d1_hat.clamp(0.0, 9.0).round().long()
            d2_pred = d2_hat.clamp(0.0, 9.0).round().long()

            # Move to CPU for plotting
            x1_cpu = _denorm_mnist(x1.detach().cpu())
            x2_cpu = _denorm_mnist(x2.detach().cpu())
            d1_cpu = d1.detach().cpu()
            d2_cpu = d2.detach().cpu()
            s_cpu = s.detach().cpu()
            s_hat_cpu = s_hat.detach().cpu()
            d1_pred_cpu = d1_pred.detach().cpu()
            d2_pred_cpu = d2_pred.detach().cpu()

            for i in range(x1_cpu.size(0)):
                correct = (s_hat_cpu[i] == s_cpu[i])
                if only_errors and correct:
                    continue
                # Concatenate the two digits horizontally: [1, 28, 56]
                pair_img = torch.cat([x1_cpu[i], x2_cpu[i]], dim=2)

                imgs.append(pair_img)
                titles.append(
                    f"{int(d1_cpu[i])}+{int(d2_cpu[i])}={int(s_cpu[i])} → "
                    f"{int(s_hat_cpu[i])}  "
                    f"({int(d1_pred_cpu[i])},{int(d2_pred_cpu[i])})"
                )

                if len(imgs) >= max_samples:
                    break
            if len(imgs) >= max_samples:
                break

    if len(imgs) == 0:
        if only_errors:
            print("No mispredicted samples found; skipping error grid.")
        else:
            print("No samples collected for visualization (unexpected).")
        return

    # Build grid
    n = len(imgs)
    cols = min(8, n)
    rows = math.ceil(n / cols)

    plt.figure(figsize=(cols * 2.0, rows * 2.0))
    for idx, img in enumerate(imgs):
        ax = plt.subplot(rows, cols, idx + 1)
        ax.imshow(img.squeeze(0), cmap="gray")
        ax.axis("off")

        # Color titles red if wrong, green if correct
        true_sum = titles[idx].split("→")[0].strip()
        pred_part = titles[idx].split("→")[1].strip()
        # quick correctness check from string is messy; recompute directly:
        # but we can piggyback on comparison we already had:
        # Instead, compute correctness again here:
        # Not strictly necessary for visualization, so just color all black or red when only_errors.
        color = "red" if only_errors else "black"

        ax.set_title(titles[idx], fontsize=8, color=color)

    plt.tight_layout()
    out_dir = os.path.dirname(filename)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"Saved grid to {filename}")


def visualize_examples(
    model: BaconAdditionWithPerception,
    loader: DataLoader,
    device: torch.device,
    tau_eval: float,
    out_prefix: str,
):
    """
    Convenience wrapper: save
      - a grid of random samples
      - a grid of mispredicted samples (if any)
    """
    samples_path = out_prefix + "_samples.png"
    errors_path = out_prefix + "_errors.png"

    print("Saving sample grid...")
    save_sample_grid(
        model,
        loader,
        device,
        tau_eval=tau_eval,
        filename=samples_path,
        max_samples=32,
        only_errors=False,
    )

    print("Saving error grid (if any mispredictions)...")
    save_sample_grid(
        model,
        loader,
        device,
        tau_eval=tau_eval,
        filename=errors_path,
        max_samples=32,
        only_errors=True,
    )


# ============================================================
# Train / Eval
# ============================================================

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

    ap.add_argument("--pretrained", default=None, type=str,
                    help="Path to a saved checkpoint to only evaluate.")
    ap.add_argument("--save-path", default=None, type=str,
                    help="Where to save the best checkpoint.")

    ap.add_argument("--freeze-cnn", action="store_true",
                    help="Freeze CNN feature extractor parameters.")
    ap.add_argument(
        "--lambda-consistency",
        default=0.1,
        type=float,
        help="Weight for consistency loss: y(x1,x2) ~ y(x2,x1).",
    )

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
    model = BaconAdditionWithPerception(
        freeze_cnn=args.freeze_cnn,
    ).to(device)

    # Pretrained evaluation only
    if args.pretrained is not None:
        if not os.path.isfile(args.pretrained):
            raise FileNotFoundError(f"Checkpoint not found: {args.pretrained}")
        ckpt = torch.load(args.pretrained, map_location=device)
        state_dict = ckpt["model_state"] if isinstance(ckpt, dict) and "model_state" in ckpt else ckpt
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print(f"Warning: missing keys: {sorted(missing)}")
        if unexpected:
            print(f"Warning: unexpected keys: {sorted(unexpected)}")

        evaluate_and_print(model, test_loader, tau_eval=args.tau_end, device=device,
                           prefix="Pretrained eval", print_ops=True)
        dump_operator_stats(model)

        # Also visualize examples for a pretrained run
        out_prefix = os.path.splitext(args.pretrained)[0]
        visualize_examples(model, test_loader, device, tau_eval=args.tau_end, out_prefix=out_prefix)
        return

    # ------- Training setup -------
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=1e-4
    )

    best_mse = float("inf")
    bad_epochs = 0

    save_path = args.save_path or "best_bacon_mnist_addition_operator.pt"

    # ------- Training loop -------
    for epoch in range(1, args.epochs + 1):
        t = (epoch - 1) / max(1, args.epochs - 1)
        tau = args.tau_start + (args.tau_end - args.tau_start) * t

        model.train()
        running_loss, seen = 0.0, 0

        for (x1, x2), (d1, d2, s) in train_loader:
            x1 = x1.to(device)
            x2 = x2.to(device)
            d1 = d1.to(device).long()
            d2 = d2.to(device).long()
            s = s.to(device).long()

            # Normalize target sum to [0,1]
            s_norm = s.float() / 18.0

            y_hat_norm, p1, p2, d1_hat, d2_hat = model(x1, x2, tau=tau, hard=False)

            # Sum regression loss
            loss_main = F.mse_loss(y_hat_norm, s_norm)

            # Entropy on digit distributions
            loss_ent = (entropy_loss_digit(p1) + entropy_loss_digit(p2)) * 0.5 * args.entropy

            # Consistency: y(x1,x2) ~ y(x2,x1)
            if args.lambda_consistency > 0.0:
                y_ab, _, _, _, _ = model(x1, x2, tau=tau, hard=False)
                y_ba, _, _, _, _ = model(x2, x1, tau=tau, hard=False)
                loss_consistency = F.mse_loss(y_ab, y_ba)
            else:
                loss_consistency = torch.zeros((), device=device)

            lambda_digits = 0.1

            logits1 = model.concept_head.logit_head(model.cnn(x1))
            logits2 = model.concept_head.logit_head(model.cnn(x2))
            loss_digits = (
                F.cross_entropy(logits1, d1) + F.cross_entropy(logits2, d2)
            ) * 0.5 * lambda_digits

            loss = loss_main + loss_ent + args.lambda_consistency * loss_consistency + loss_digits

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

        # ----- Eval (with digit accuracies) -----
        print(
            f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | tau={tau:.3f}"
        )
        mse, mae, acc_sum, acc_d1, acc_d2 = evaluate_and_print(
            model,
            test_loader,
            tau_eval=args.tau_end,
            device=device,
            prefix=f"Eval epoch {epoch:03d}",
            print_ops=True,
        )

        dump_operator_stats(model)

        # ----- Early stopping & checkpointing -----
        if mse < best_mse:
            best_mse = mse
            bad_epochs = 0
            torch.save(
                {"model_state": model.state_dict(), "args": vars(args)},
                save_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping (no improvement in MSE for {args.patience} epochs).")
                break

    print(f"\nBest MSE: {best_mse:.4f} (checkpoint saved to {save_path})")

    # Final detailed diagnostics on test set
    evaluate_and_print(
        model,
        test_loader,
        tau_eval=args.tau_end,
        device=device,
        prefix="Final eval",
        print_ops=True,
    )
    dump_operator_stats(model)

    # Visualizations: samples + mispredictions
    out_prefix = os.path.splitext(save_path)[0]
    visualize_examples(model, test_loader, device, tau_eval=args.tau_end, out_prefix=out_prefix)


if __name__ == "__main__":
    main()
