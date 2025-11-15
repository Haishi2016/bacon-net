#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, math, random, os, sys
from typing import Iterator, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, IterableDataset, DataLoader
from torchvision import datasets, transforms

# Ensure local bacon package is used instead of any installed site-package
sys.path.insert(
    0,
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)

from bacon.binaryTreeLogicNet import binaryTreeLogicNet
from bacon.aggregators.bool import MinMaxAggregator


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
# BACON-based Reasoning Model with K as input
# ============================================================

class BaconAdditionModel(nn.Module):
    """
    Shared CNN + concept head -> 20D concept vector (10 for each image)
    + 19D one-hot for target sum K (0..18) -> BACON reasoning head.

    Training:
      - If single_sum is not None: always train on that K (binary task "sum==K?").
      - If single_sum is None: active K is rotated across epochs.
    """

    def __init__(
        self,
        freeze_cnn: bool = False,
    ):
        super().__init__()

        self.cnn = SmallCnn(128)
        self.concept_head = ImageConceptHead(128, 10)
        if freeze_cnn:
            for p in self.cnn.parameters():
                p.requires_grad = False

        agg = MinMaxAggregator()
        # 20 concept dims + 19 one-hot for K
        self.num_sums = 19
        self.bacon = binaryTreeLogicNet(
            input_size=20 + self.num_sums,
            weight_mode="fixed",
            weight_normalization="minmax",
            aggregator=agg,
            normalize_andness=True,
            tree_layout="left",
            loss_amplifier=1.0,            
            is_frozen=False,
        )
        self.bacon.auto_refine = False

    # ---------- feature / concept helpers ----------

    def _concepts_from_images(self, x1: torch.Tensor, x2: torch.Tensor,
                              tau: float, hard: bool) -> torch.Tensor:
        """
        Shared CNN + concept head -> [B,20] concept vector.
        """
        z1 = self.cnn(x1)  # [B,128]
        z2 = self.cnn(x2)  # [B,128]

        p1 = self.concept_head(z1, tau=tau, hard=hard)  # [B,10]
        p2 = self.concept_head(z2, tau=tau, hard=hard)  # [B,10]
        concepts = torch.cat([p1, p2], dim=1)           # [B,20]
        return concepts

    def _bacon_input(self, concepts: torch.Tensor, K: torch.Tensor) -> torch.Tensor:
        """
        Concatenate concepts [B,20] with K one-hot [B,19] -> [B,39].
        K: LongTensor [B] with values in [0..18].
        """
        K_onehot = F.one_hot(K, num_classes=self.num_sums).float()  # [B,19]
        bacon_in = torch.cat([concepts, K_onehot], dim=1)           # [B,39]
        return bacon_in

    # ---------- core forward variants ----------

    def forward_for_sum(self, x1: torch.Tensor, x2: torch.Tensor,
                        tau: float, K: int, hard: bool = False):
        """
        Forward for a specific target sum K (scalar int 0..18).
        Returns:
          y_prob: [B] in (0,1)  (probability "sum == K?")
          c_prob: [B,20] concept probabilities
        """
        concepts = self._concepts_from_images(x1, x2, tau, hard)  # [B,20]
        B = concepts.size(0)
        K_batch = torch.full((B,), K, device=concepts.device, dtype=torch.long)
        bacon_in = self._bacon_input(concepts, K_batch)           # [B,39]
        y = self.bacon(bacon_in)                                  # [B] or [B,1]
        y_prob = y.view(-1)
        return y_prob, concepts

    def scores_all_sums(self, x1: torch.Tensor, x2: torch.Tensor,
                        tau: float, hard: bool = False):
        """
        Compute scores for all sums 0..18 in one go (reusing concepts).
        Returns:
          scores: [B,19] where scores[:,k] ~ P(sum==k?)
          c_prob: [B,20] concepts
        """
        concepts = self._concepts_from_images(x1, x2, tau, hard)  # [B,20]
        B = concepts.size(0)
        scores = []

        for K in range(self.num_sums):
            K_batch = torch.full((B,), K, device=concepts.device, dtype=torch.long)
            bacon_in = self._bacon_input(concepts, K_batch)       # [B,39]
            y = self.bacon(bacon_in).view(-1)                     # [B]
            scores.append(y)

        scores = torch.stack(scores, dim=1)  # [B,19]
        return scores, concepts


# ============================================================
# Loss helpers & rule visualization
# ============================================================

def group_entropy_loss(c_prob: torch.Tensor) -> torch.Tensor:
    """
    Encourages each image's digit distribution to be low entropy.
    c_prob is [B,20] = [p1(10), p2(10)].
    """
    eps = 1e-8
    g1 = c_prob[:, :10]
    g2 = c_prob[:, 10:]
    ent = -(g1 * (g1 + eps).log()).sum(dim=1) - (g2 * (g2 + eps).log()).sum(dim=1)
    return (ent / math.log(10)).mean()


def print_soft_rules_vs_gt(
    model: BaconAdditionModel,
    test_loader: DataLoader,
    tau_eval: float = 0.8,
    device: str = "cpu",
    top_k_pairs: int = 5,
    single_sum: Optional[int] = None,
):
    """
    Soft rule visualization vs ground truth.
    If single_sum is None: multi-sum mode (decompose y_k over all sums).
    If single_sum is K: single-sum binary mode.
    """
    model.eval()
    num_sums = model.bacon.original_input_size - 20  # should be 19

    if single_sum is None:
        # ---------- Multi-sum mode ----------
        pair_sum = torch.zeros(num_sums, 10, 10, device=device)
        count_k = torch.zeros(num_sums, device=device)

        with torch.no_grad():
            for (x1, x2), _ in test_loader:
                x1 = x1.to(device)
                x2 = x2.to(device)

                # hard concepts for viz
                z1 = model.cnn(x1)
                z2 = model.cnn(x2)
                p1 = model.concept_head(z1, tau=tau_eval, hard=True)  # [B,10]
                p2 = model.concept_head(z2, tau=tau_eval, hard=True)  # [B,10]

                # scores for all sums
                scores, _ = model.scores_all_sums(x1, x2, tau=tau_eval, hard=True)  # [B,19]
                k_pred = scores.argmax(dim=1)                                       # [B]

                outer = p1.unsqueeze(2) * p2.unsqueeze(1)  # [B,10,10]

                for kk in range(num_sums):
                    mask = (k_pred == kk).float().view(-1, 1, 1)  # [B,1,1]
                    pair_sum[kk] += (outer * mask).sum(dim=0)
                    count_k[kk]  += mask.sum()

        print("\n=== BACON soft-rule attribution vs ground truth manifold (multi-sum) ===")
        eps = 1e-12
        for kk in range(num_sums):
            if count_k[kk] < 1:
                print(f"y_{kk}: (no predicted samples)")
                continue

            contrib = pair_sum[kk] / (count_k[kk] + eps)  # [10,10]
            total_mass = contrib.sum().item() + eps

            # valid pairs (i+j=k)
            valid_pairs = [(i, j) for i in range(10) for j in range(10) if i + j == kk]
            valid_mass = sum(contrib[i, j].item() for (i, j) in valid_pairs) + eps
            leakage = max(total_mass - valid_mass, 0.0) / valid_mass

            # normalized contributions over valid pairs
            valid_contrib = [
                (i, j, contrib[i, j].item() / valid_mass)
                for (i, j) in valid_pairs
            ]
            valid_contrib.sort(key=lambda t: t[2], reverse=True)
            top_valid = valid_contrib[:top_k_pairs]

            print(f"\ny_{kk}: leakage={leakage:.3f}")
            if not top_valid:
                print("  (no valid pairs have non-zero contribution)")
            else:
                human = " ∨ ".join([f"({i}∧{j}):{v:.2f}" for (i, j, v) in top_valid])
                print(f"  Top-{len(top_valid)} valid pairs by normalized contribution: {human}")
        print("============================================================\n")

    else:
        # ---------- Single-sum mode ----------
        K = single_sum
        pair_sum = torch.zeros(10, 10, device=device)
        count_pos = torch.zeros(1, device=device)

        with torch.no_grad():
            for (x1, x2), y in test_loader:
                x1 = x1.to(device)
                x2 = x2.to(device)
                y = y.to(device)

                z1 = model.cnn(x1)
                z2 = model.cnn(x2)
                p1 = model.concept_head(z1, tau=tau_eval, hard=True)  # [B,10]
                p2 = model.concept_head(z2, tau=tau_eval, hard=True)  # [B,10]

                y_prob, _ = model.forward_for_sum(x1, x2, tau=tau_eval, K=K, hard=True)  # [B]
                pred_bin = (y_prob >= 0.5)                                              # [B]

                outer = p1.unsqueeze(2) * p2.unsqueeze(1)             # [B,10,10]
                mask = pred_bin.float().view(-1, 1, 1)                # [B,1,1]
                pair_sum += (outer * mask).sum(dim=0)
                count_pos += mask.sum()

        print(f"\n=== BACON soft-rule attribution vs ground truth (single-sum K={K}) ===")
        eps = 1e-12
        if count_pos.item() < 1:
            print("No positive predictions; cannot build rule heatmap.")
            print("============================================================\n")
            return

        contrib = pair_sum / (count_pos + eps)    # [10,10]
        total_mass = contrib.sum().item() + eps

        valid_pairs = [(i, j) for i in range(10) for j in range(10) if i + j == K]
        valid_mass = sum(contrib[i, j].item() for (i, j) in valid_pairs) + eps
        leakage = max(total_mass - valid_mass, 0.0) / valid_mass

        valid_contrib = [
            (i, j, contrib[i, j].item() / valid_mass)
            for (i, j) in valid_pairs
        ]
        valid_contrib.sort(key=lambda t: t[2], reverse=True)
        top_valid = valid_contrib[:top_k_pairs]

        print(f"sum=={K}: leakage={leakage:.3f}")
        if not top_valid:
            print("  (no valid digit pairs have non-zero contribution)")
        else:
            human = " ∨ ".join([f"({i}∧{j}):{v:.2f}" for (i, j, v) in top_valid])
            print(f"  Top-{len(top_valid)} valid pairs by normalized contribution: {human}")
        print("============================================================\n")


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

    # Fixed K vs rotating K
    ap.add_argument("--single-sum", type=int, default=None,
                    help="If set (0..18), train a single BACON head for 'sum==K?'. "
                         "If None, rotate K across epochs and decode 19-way sum via argmax_K.")

    ap.add_argument(
        "--epochs-per-sum",
        default=5,
        type=int,
        help="Number of consecutive epochs to train on the same active sum (multi-sum with rotation).",
    )

    ap.add_argument(
        "--lambda-consistency",
        default=0.1,
        type=float,
        help="Weight for consistency regularization loss (y(x1,x2) ~ y(x2,x1)) for the same K.",
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
    model = BaconAdditionModel(
        freeze_cnn=args.freeze_cnn,
    ).to(device)

    num_sums = model.bacon.original_input_size - 20  # should be 19

    # Pre-sample which K to train each block in rotating mode
    if args.single_sum is None:
        # Approximate frequency of each sum k under uniform digits
        pair_counts = np.array([min(k + 1, num_sums - k) for k in range(num_sums)],
                               dtype=np.float32)
        pair_probs = pair_counts / pair_counts.sum()
        epochs_per_sum = max(1, args.epochs_per_sum)
        num_choices = math.ceil(args.epochs / epochs_per_sum)
        epoch_sums = np.random.choice(
            np.arange(num_sums), size=num_choices, p=pair_probs
        )
    else:
        epoch_sums = None

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

        model.eval()
        with torch.no_grad():
            if args.single_sum is not None:
                # binary eval for fixed K
                K = args.single_sum
                tot, correct = 0, 0
                for (x1, x2), y in test_loader:
                    x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                    y_prob, _ = model.forward_for_sum(x1, x2, tau=args.tau_end, K=K, hard=True)
                    y_bin = (y == K)
                    pred_bin = (y_prob >= 0.5)
                    correct += (pred_bin == y_bin).sum().item()
                    tot += y.size(0)
                acc_bin = correct / max(1, tot)
                print(f"Pretrained eval | acc_bin(sum=={K})={acc_bin*100:.2f}%")
                print_soft_rules_vs_gt(model, test_loader, tau_eval=args.tau_end,
                                       device=device, single_sum=K)
            else:
                # multi-class eval via scores_all_sums
                tot, correct = 0, 0
                for (x1, x2), y in test_loader:
                    x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                    scores, _ = model.scores_all_sums(x1, x2, tau=args.tau_end, hard=True)
                    pred = scores.argmax(dim=1)
                    correct += (pred == y).sum().item()
                    tot += y.size(0)
                acc_all = correct / max(1, tot)
                print(f"Pretrained eval | acc_all={acc_all*100:.2f}%")
                print_soft_rules_vs_gt(model, test_loader, tau_eval=args.tau_end,
                                       device=device, single_sum=None)
        return

    # ------- Training setup -------
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=1e-4
    )

    best_score = 0.0
    bad_epochs = 0

    # Decide save name
    if args.save_path is not None:
        base_save_path = args.save_path
    else:
        if args.single_sum is None:
            base_save_path = "best_bacon_mnist_addition_rotatingK.pt"
        else:
            base_save_path = f"best_bacon_mnist_addition_singleK{args.single_sum}.pt"

    # ------- Training loop -------
    for epoch in range(1, args.epochs + 1):
        t = (epoch - 1) / max(1, args.epochs - 1)
        tau = args.tau_start + (args.tau_end - args.tau_start) * t

        model.train()
        running_loss, seen = 0.0, 0

        # Choose active K for this epoch
        if args.single_sum is not None:
            active_sum = int(args.single_sum)
        else:
            epochs_per_sum = max(1, args.epochs_per_sum)
            block_idx = (epoch - 1) // epochs_per_sum  # 0-based block
            active_sum = int(epoch_sums[block_idx])

        for (x1, x2), y in train_loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)

            # Forward for this K
            y_prob, c_prob = model.forward_for_sum(x1, x2, tau=tau,
                                                   K=active_sum, hard=False)  # [B], [B,20]

            # Binary target: is sum == active_sum?
            y_bin = (y == active_sum).float()     # [B]
            p = y_prob.clamp(1e-6, 1 - 1e-6)
            log_p = torch.log(p)
            log_1_p = torch.log(1.0 - p)
            loss_main = -(y_bin * log_p + (1.0 - y_bin) * log_1_p).mean()

            # Entropy regularizer on concepts
            loss_ent = group_entropy_loss(c_prob) * args.entropy

            # Symmetry / consistency term for same K
            if args.lambda_consistency > 0.0:
                y_ab, _ = model.forward_for_sum(x1, x2, tau, K=active_sum, hard=False)
                y_ba, _ = model.forward_for_sum(x2, x1, tau, K=active_sum, hard=False)
                loss_consistency = F.mse_loss(y_ab, y_ba)
            else:
                loss_consistency = torch.tensor(0.0, device=device)

            loss = loss_main + loss_ent + args.lambda_consistency * loss_consistency

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

        # ----- Eval -----
        model.eval()
        with torch.no_grad():
            if args.single_sum is not None:
                # Binary eval for active_sum = K
                K = active_sum
                tot_all, correct_all = 0, 0
                pos_scores, neg_scores = [], []

                for (x1, x2), y in test_loader:
                    x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                    y_prob_eval, _ = model.forward_for_sum(
                        x1, x2, tau=args.tau_end, K=K, hard=True
                    )  # [B]
                    y_bin = (y == K)
                    pred_bin = (y_prob_eval >= 0.5)

                    correct_all += (pred_bin == y_bin).sum().item()
                    tot_all += y.size(0)

                    # Diagnostic collects
                    pos_mask = y_bin
                    neg_mask = ~y_bin
                    if pos_mask.any():
                        pos_scores.append(y_prob_eval[pos_mask])
                    if neg_mask.any():
                        neg_scores.append(y_prob_eval[neg_mask])

                acc_all = correct_all / max(1, tot_all)
                score = acc_all

                # Diagnostic dump: how separated are scores?
                if pos_scores and neg_scores:
                    pos_scores_all = torch.cat(pos_scores)
                    neg_scores_all = torch.cat(neg_scores)
                    print(
                        f"  [diag K={K}] "
                        f"pos_mean={pos_scores_all.mean():.3f}, "
                        f"neg_mean={neg_scores_all.mean():.3f}, "
                        f"pos_min={pos_scores_all.min():.3f}, pos_max={pos_scores_all.max():.3f}, "
                        f"neg_min={neg_scores_all.min():.3f}, neg_max={neg_scores_all.max():.3f}"
                    )

                print(
                    f"Epoch {epoch:03d} | train_loss={train_loss:.4f} "
                    f"| acc_bin(sum=={K})={acc_all*100:.2f}% "
                    f"| tau={tau:.3f}"
                )

            else:
                # Multi-class eval: decode sum via argmax_K score
                tot_all, correct_all = 0, 0
                # diagnostic for current active_sum
                pos_scores, neg_scores = [], []

                for (x1, x2), y in test_loader:
                    x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                    scores, _ = model.scores_all_sums(
                        x1, x2, tau=args.tau_end, hard=True
                    )  # [B,19]
                    pred = scores.argmax(dim=1)
                    correct_all += (pred == y).sum().item()
                    tot_all += y.size(0)

                    # Diagnostic for active_sum: treat it as binary head
                    y_prob_active = scores[:, active_sum]           # [B]
                    y_bin = (y == active_sum)                       # [B]
                    pos_mask = y_bin
                    neg_mask = ~y_bin
                    if pos_mask.any():
                        pos_scores.append(y_prob_active[pos_mask])
                    if neg_mask.any():
                        neg_scores.append(y_prob_active[neg_mask])

                acc_all = correct_all / max(1, tot_all)
                score = acc_all

                # Diagnostic dump for active_sum in multi-class mode
                if pos_scores and neg_scores:
                    pos_scores_all = torch.cat(pos_scores)
                    neg_scores_all = torch.cat(neg_scores)
                    print(
                        f"  [diag active_sum={active_sum}] "
                        f"pos_mean={pos_scores_all.mean():.3f}, "
                        f"neg_mean={neg_scores_all.mean():.3f}, "
                        f"pos_min={pos_scores_all.min():.3f}, pos_max={pos_scores_all.max():.3f}, "
                        f"neg_min={neg_scores_all.min():.3f}, neg_max={neg_scores_all.max():.3f}"
                    )

                print(
                    f"Epoch {epoch:03d} | train_loss={train_loss:.4f} "
                    f"| acc_all={acc_all*100:.2f}% | tau={tau:.3f} "
                    f"| active_sum={active_sum}"
                )

        # ----- Early stopping & checkpointing -----
        if score > best_score:
            best_score = score
            bad_epochs = 0
            torch.save(
                {"model_state": model.state_dict(), "args": vars(args)},
                base_save_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping (no improvement for {args.patience} epochs).")
                break

    if args.single_sum is None:
        print(f"\nBest acc_all: {best_score*100:.2f}% (saved to {base_save_path})")
        print_soft_rules_vs_gt(model, test_loader, tau_eval=args.tau_end,
                               device=device, single_sum=None)
    else:
        print(f"\nBest acc_bin(sum=={args.single_sum}): {best_score*100:.2f}% "
              f"(saved to {base_save_path})")
        print_soft_rules_vs_gt(model, test_loader, tau_eval=args.tau_end,
                               device=device, single_sum=args.single_sum)


if __name__ == "__main__":
    main()
