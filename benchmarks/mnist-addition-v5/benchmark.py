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
# Multiclass BACON head (19 sums, 0..18)
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

        # BACON permutations over digit concepts (10 per image)
        p1p = self.perm1.input_to_leaf(p1)  # [B,10]
        p2p = self.perm2.input_to_leaf(p2)  # [B,10]

        # Pairwise AND via min-like aggregator (vectorized)
        B = p1p.size(0)
        x = p1p.unsqueeze(2).expand(B, 10, 10)
        y = p2p.unsqueeze(1).expand(B, 10, 10)
        a_and = torch.tensor(1.0, device=x.device)
        s = self.agg.aggregate_tensor(x, y, a_and, w0=0.5, w1=0.5)  # [B,10,10]

        # Smooth OR per sum k: 1 - prod(1 - s_ij) for i+j=k
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
        c_prob = torch.cat([p1, p2], dim=1)  # [B,20]
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


def print_soft_rule_attribution(model, test_loader, tau_eval=0.8, device='cpu', top_k=5):
    """
    Soft rule attribution with leakage:

      - For each sum k, compute contribution matrix C_k(i,j) averaged over
        samples predicted as k.
      - For each k, define valid pairs V_k = {(i,j): i+j = k}.
      - leakage_k = total mass on invalid pairs / total mass.
      - Print top-k valid pairs by normalized contribution.
    """
    model.eval()
    pair_sum = torch.zeros(19, 10, 10, device=device)
    count_k = torch.zeros(19, device=device)

    with torch.no_grad():
        for (x1, x2), _ in test_loader:
            x1, x2 = x1.to(device), x2.to(device)
            z1 = model.tower1(x1)
            z2 = model.tower2(x2)
            p1 = model.head1(z1, tau=tau_eval, hard=True)  # [B,10]
            p2 = model.head2(z2, tau=tau_eval, hard=True)  # [B,10]

            y_prob, _ = model(x1, x2, tau=tau_eval, hard=True)  # [B,19]
            k = y_prob.argmax(1)  # [B]

            outer = p1.unsqueeze(2) * p2.unsqueeze(1)  # [B,10,10]
            for kk in range(19):
                mask = (k == kk).float().view(-1, 1, 1)
                pair_sum[kk] += (outer * mask).sum(dim=0)
                count_k[kk] += mask.sum()

    print("\n=== BACON soft-rule attribution vs ground truth manifold ===")
    for kk in range(19):
        if count_k[kk] == 0:
            print(f"y_{kk}: (no samples)")
            continue

        contrib = pair_sum[kk] / count_k[kk]  # [10,10]
        total_mass = contrib.sum().item() + 1e-12

        valid_mask = torch.zeros_like(contrib)
        for i in range(10):
            j = kk - i
            if 0 <= j <= 9:
                valid_mask[i, j] = 1.0

        valid_mass = (contrib * valid_mask).sum().item()
        leakage = max(0.0, (total_mass - valid_mass) / total_mass)

        # Collect valid pairs and normalize their contribution by valid_mass
        valid_pairs = []
        if valid_mass > 0:
            for i in range(10):
                for j in range(10):
                    if valid_mask[i, j] > 0:
                        score = contrib[i, j].item() / valid_mass
                        valid_pairs.append((i, j, score))

        valid_pairs.sort(key=lambda t: t[2], reverse=True)
        top_valid = valid_pairs[:top_k]

        print(f"\ny_{kk}: leakage={leakage:.3f}")
        if top_valid:
            pieces = [f"({i}∧{j}):{v:.2f}" for (i, j, v) in top_valid]
            print("  Top-{} valid pairs by normalized contribution: {}".format(
                len(top_valid), " ∨ ".join(pieces)))
        else:
            print("  (no valid mass captured)")

    print("============================================================\n")


def count_trainable_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_model_state(model: nn.Module, path: str, device: torch.device):
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location=device)
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        state_dict = ckpt["model_state"]
    elif isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
    elif hasattr(ckpt, "state_dict"):
        state_dict = ckpt.state_dict()
    else:
        state_dict = ckpt
    cleaned = {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"Warning: missing keys: {sorted(missing)}")
    if unexpected:
        print(f"Warning: unexpected keys: {sorted(unexpected)}")


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
    ap.add_argument(
        "--single-sum",
        type=int,
        default=None,
        help="If set (0–18), train a single binary head: 'sum == K' vs 'sum != K' "
             "by using only the K-th output of the existing 19-way head. "
             "Everything else (CNN, concept head, BACON perms) stays the same."
    )

    # partial / noisy supervision controls
    ap.add_argument(
        "--supervised-sums",
        type=str,
        default=None,
        help="Comma-separated list of sums (0-18) with clean labels. "
             "Others are treated according to --unsup-mode. "
             "If omitted, all sums are supervised as in the original v1."
    )
    ap.add_argument(
        "--unsup-mode",
        type=str,
        default="clean",
        choices=["clean", "ignore", "noise"],
        help="Behavior for sums not in --supervised-sums: "
             "'clean' = treat all labels as clean (original behavior); "
             "'ignore' = exclude them from CE loss; "
             "'noise' = replace their labels with random sums."
    )
    ap.add_argument("--lambda-consistency", default=0.1, type=float,
                    help="weight for consistency regularization on unsupervised sums")

    # optional freezing of CNN towers
    ap.add_argument(
        "--freeze-cnn",
        action="store_true",
        help="Freeze the CNN towers (tower1/tower2) to isolate reasoning layer."
    )

    # saving / loading
    ap.add_argument("--pretrained", default=None, type=str,
                    help="Path to a saved checkpoint to load before training or for eval.")
    ap.add_argument("--save-path", default="best_bacon_mnist_addition_multiclass.pt", type=str,
                    help="Where to save the best model checkpoint.")
    ap.add_argument("--eval-only", action="store_true",
                    help="If set, load --pretrained and run eval + rule attribution only (no training).")

    args = ap.parse_args()

    if args.single_sum is not None:
        if not (0 <= args.single_sum <= 18):
            raise ValueError(f"--single-sum must be in [0,18], got {args.single_sum}")
        print(f"*** SINGLE-SUM MODE: training binary classifier for sum == {args.single_sum} ***")


    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parse supervised sums, if provided
    if args.supervised_sums is not None:
        sup_list = [s for s in args.supervised_sums.split(",") if s.strip() != ""]
        supervised_sums = sorted(int(s) for s in sup_list)
        print(f"Using supervised SUM labels: {supervised_sums} | unsup_mode={args.unsup_mode}")
        supervised_sums_tensor = torch.tensor(supervised_sums, device=device)
    else:
        supervised_sums = None
        supervised_sums_tensor = None
        print("All sums supervised as clean (original BACON v1 behavior).")

    train_ds = MnistAdditionStream(args.data, train=True, epoch_len=args.train_pairs, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=2)

    test_ds = MnistAdditionFixed(args.data, train=False, size_pairs=args.test_pairs, seed=999)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = BaconMultiHead().to(device)

    # Load checkpoint if provided
    if args.pretrained is not None:
        print(f"Loading checkpoint from {args.pretrained}")
        load_model_state(model, args.pretrained, device)

    # Optionally freeze CNN towers
    if args.freeze_cnn:
        for p in model.tower1.parameters():
            p.requires_grad = False
        for p in model.tower2.parameters():
            p.requires_grad = False
        print("CNN towers are FROZEN (tower1/tower2).")

    n_trainable = count_trainable_params(model)
    print(f"Trainable parameters: {n_trainable}")

    # Eval-only mode: just evaluate and visualize rules
    if args.eval_only:
        print("\nEval-only mode.")
        model.eval()
        with torch.no_grad():
            tot_all, correct_all = 0, 0
            for (x1, x2), y in test_loader:
                x1, x2 = x1.to(device), x2.to(device)
                y = y.to(device)
                y_prob, _ = model(x1, x2, tau=args.tau_end, hard=True)
                pred = y_prob.argmax(1)
                correct_all += (pred == y).sum().item()
                tot_all += y.size(0)
            acc_all = correct_all / max(1, tot_all)
        print(f"Eval-only | acc_all={acc_all*100:.2f}%")
        print_soft_rule_attribution(model, test_loader, tau_eval=args.tau_end, device=device)
        return

    # Loss/optim
    ce = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=args.lr,
        weight_decay=5e-4,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-4)

    best_acc_all = 0.0
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

            y_prob, c_prob = model(x1, x2, tau=tau, hard=False)
            y_true = y.clone()

            # ---------- partial / noisy supervision handling ----------
            if args.single_sum is not None:
                # Binary labels: 1 if sum == K, else 0
                y_bin = (y_true == args.single_sum).long()  # [B]

                # Take the K-th sum probability and build a 2-class distribution
                p_k = y_prob[:, args.single_sum].clamp(1e-6, 1 - 1e-6)        # [B]
                p_not_k = (1.0 - p_k).clamp(1e-6, 1 - 1e-6)                   # [B]
                probs_2 = torch.stack([p_not_k, p_k], dim=1)                  # [B,2]
                logits_2 = torch.logit(probs_2)                               # [B,2]

                loss_main = ce(logits_2, y_bin)
            elif supervised_sums is not None and args.unsup_mode in ("ignore", "noise"):
                supervised_mask = torch.isin(y_true, supervised_sums_tensor)

                if args.unsup_mode == "ignore":
                    logits = torch.logit(y_prob.clamp(1e-6, 1 - 1e-6))
                    if supervised_mask.any():
                        loss_main = ce(logits[supervised_mask], y_true[supervised_mask])
                    else:
                        loss_main = None

                elif args.unsup_mode == "noise":
                    unsup_mask = ~supervised_mask
                    if unsup_mask.any():
                        rand_labels = torch.randint(
                            low=0, high=19,
                            size=(unsup_mask.sum(),),
                            device=y_true.device
                        )
                        y_true[unsup_mask] = rand_labels
                    logits = torch.logit(y_prob.clamp(1e-6, 1 - 1e-6))
                    loss_main = ce(logits, y_true)
            else:
                logits = torch.logit(y_prob.clamp(1e-6, 1 - 1e-6))
                loss_main = ce(logits, y_true)

            # entropy regularizer (always over full batch)
            loss_ent = group_entropy_loss(c_prob) * args.entropy

            # symmetry / consistency regularizer
            y_ab, _ = model(x1, x2, tau)
            y_ba, _ = model(x2, x1, tau)
            loss_consistency = F.mse_loss(y_ab, y_ba)

            if loss_main is None:
                loss = loss_ent + args.lambda_consistency * loss_consistency
            else:
                loss = loss_main + loss_ent + args.lambda_consistency * loss_consistency
            # -------------------------------------------------------

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
            if args.single_sum is None:
                tot_all, correct_all = 0, 0
                tot_sup, correct_sup = 0, 0
                tot_uns, correct_uns = 0, 0

                for (x1, x2), y in test_loader:
                    x1, x2 = x1.to(device), x2.to(device)
                    y = y.to(device)
                    y_prob, _ = model(x1, x2, tau=args.tau_end, hard=True)
                    pred = y_prob.argmax(1)

                    correct_all += (pred == y).sum().item()
                    tot_all += y.size(0)

                    if supervised_sums_tensor is not None:
                        mask_sup = torch.isin(y, supervised_sums_tensor)
                        mask_uns = ~mask_sup
                        if mask_sup.any():
                            correct_sup += (pred[mask_sup] == y[mask_sup]).sum().item()
                            tot_sup += mask_sup.sum().item()
                        if mask_uns.any():
                            correct_uns += (pred[mask_uns] == y[mask_uns]).sum().item()
                            tot_uns += mask_uns.sum().item()

                acc_all = correct_all / max(1, tot_all)
                acc_sup = correct_sup / max(1, tot_sup) if tot_sup > 0 else 0.0
                acc_uns = correct_uns / max(1, tot_uns) if tot_uns > 0 else 0.0
                print(
                    f"Epoch {epoch:03d} | "
                    f"train_loss={train_loss:.4f} | "
                    f"acc_all={acc_all*100:.2f}% | "
                    f"acc_sup={acc_sup*100:.2f}% | "
                    f"acc_uns={acc_uns*100:.2f}% | "
                    f"tau={tau:.3f}"
                )
                last_eval_acc = acc_all
            else: 
                k = args.single_sum
                tot, correct = 0, 0
                for (x1, x2), y in test_loader:
                    x1, x2 = x1.to(device), x2.to(device)
                    y = y.to(device)
                    y_prob, _ = model(x1, x2, tau=args.tau_end, hard=True)

                    p_k = y_prob[:, k]
                    y_bin = (y == k)          # bool
                    pred_bin = (p_k >= 0.5)   # bool

                    correct += (pred_bin == y_bin).sum().item()
                    tot += y.size(0)

                acc_bin = correct / max(1, tot)
                print(
                    f"Epoch {epoch:03d} | train_loss={train_loss:.4f} "
                    f"| acc_bin(sum=={k})={acc_bin*100:.2f}% | tau={tau:.3f}"
                )
                last_eval_acc = acc_bin
       

        if last_eval_acc > best_acc_all:
            best_acc_all = last_eval_acc
            bad_epochs = 0
            ckpt = {
                "model_state": model.state_dict(),
                "args": vars(args),
            }
            torch.save(ckpt, args.save_path)
            print(f"  ↑ Saved new best model to {args.save_path} (acc_all={best_acc_all*100:.2f}%)")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping (no improvement for {args.patience} epochs).")
                break

    print(f"\nBest acc_all: {best_acc_all*100:.2f}% (checkpoint: {args.save_path})")
    print_soft_rule_attribution(model, test_loader, tau_eval=args.tau_end, device=device)


if __name__ == "__main__":
    main()
