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
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
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
# BACON-based Reasoning Model
# ============================================================

class BaconAdditionModel(nn.Module):
    """
    Shared CNN + concept head -> 20D concept vector (10 for each image)
    -> BACON reasoning head(s).

    Modes:
      - multi-sum (single_sum is None):
          one BACON tree per sum 0..18 => 19-way output.
      - single-sum (single_sum = K):
          one BACON tree specialized to "sum == K?" => binary output.
    """

    def __init__(
        self,
        single_sum: Optional[int] = None,
        freeze_cnn: bool = False,
    ):
        super().__init__()
        self.single_sum = single_sum

        self.cnn = SmallCnn(128)
        self.concept_head = ImageConceptHead(128, 10)
        if freeze_cnn:
            for p in self.cnn.parameters():
                p.requires_grad = False

        agg = MinMaxAggregator()
        if self.single_sum is None:
            # 19 BACON trees, one per sum
            self.bacons = nn.ModuleList([
                binaryTreeLogicNet(
                    input_size=20,
                    is_frozen=False,
                    weight_mode="fixed",
                    weight_normalization="minmax",
                    aggregator=agg,
                    normalize_andness=False,
                    tree_layout="balanced",
                    loss_amplifier=1.0,
                )
                for _ in range(19)
            ])
        else:
            # single BACON tree for sum == K
            self.bacon_single = binaryTreeLogicNet(
                input_size=20,
                is_frozen=False,
                weight_mode="fixed",
                weight_normalization="minmax",
                aggregator=agg,
                normalize_andness=False,
                tree_layout="balanced",
                loss_amplifier=1.0,
            )

    def forward(self, x1: torch.Tensor, x2: torch.Tensor, tau: float, hard: bool = False):
        # shared CNN
        z1 = self.cnn(x1)  # [B,128]
        z2 = self.cnn(x2)  # [B,128]

        # shared concept head
        p1 = self.concept_head(z1, tau=tau, hard=hard)  # [B,10]
        p2 = self.concept_head(z2, tau=tau, hard=hard)  # [B,10]
        concepts = torch.cat([p1, p2], dim=1)           # [B,20]

        if self.single_sum is None:
            # 19-way BACON outputs
            ys = []
            for k in range(19):
                yk = self.bacons[k](concepts)   # [B] or [B,1]
                ys.append(yk.view(-1))
            y_prob = torch.stack(ys, dim=1)     # [B,19]
        else:
            # Single-sum mode: one scalar per sample (prob sum==K)
            yk = self.bacon_single(concepts)    # [B] or [B,1]
            y_prob = yk.view(-1)                # [B]

        c_prob = concepts  # [B,20], concept distribution for regularization / viz
        return y_prob, c_prob


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
    top_k_pairs: int = 10,
):
    """
    Soft rule visualization vs ground truth.

    Multi-sum mode:
      For each sum k, we:
        - look at samples predicted as k (argmax over 19D)
        - accumulate co-occurrence heatmaps of hard digit concepts for (d1,d2)
        - compute normalized contribution of valid (i+j=k) pairs and leakage.

    Single-sum mode (sum==K):
      We:
        - look at samples predicted as "sum==K" (prob >= 0.5)
        - accumulate co-occurrence heatmap for that binary head
        - compare to ground-truth valid pairs for K.
    """
    model.eval()

    if model.single_sum is None:
        # ---------- Multi-sum mode ----------
        pair_sum = torch.zeros(19, 10, 10, device=device)
        count_k = torch.zeros(19, device=device)

        with torch.no_grad():
            for (x1, x2), _ in test_loader:
                x1 = x1.to(device)
                x2 = x2.to(device)

                # hard concepts for viz
                z1 = model.cnn(x1)
                z2 = model.cnn(x2)
                p1 = model.concept_head(z1, tau=tau_eval, hard=True)  # [B,10] one-hot-ish
                p2 = model.concept_head(z2, tau=tau_eval, hard=True)  # [B,10]
                y_prob, _ = model(x1, x2, tau=tau_eval, hard=True)    # [B,19]
                k_pred = y_prob.argmax(dim=1)                         # [B]

                outer = p1.unsqueeze(2) * p2.unsqueeze(1)  # [B,10,10]

                for kk in range(19):
                    mask = (k_pred == kk).float().view(-1, 1, 1)  # [B,1,1]
                    pair_sum[kk] += (outer * mask).sum(dim=0)
                    count_k[kk]  += mask.sum()

        print("\n=== BACON soft-rule attribution vs ground truth manifold (multi-sum) ===")
        eps = 1e-12
        for kk in range(19):
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
        K = model.single_sum
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
                y_prob, _ = model(x1, x2, tau=tau_eval, hard=True)    # [B]
                pred_bin = (y_prob >= 0.5)                            # [B] True/False

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
                    help="Where to save the best checkpoint. "
                         "If omitted, a default name is used depending on mode.")

    ap.add_argument("--freeze-cnn", action="store_true",
                    help="Freeze CNN feature extractor parameters.")

    # Single-sum vs multi-sum mode
    ap.add_argument("--single-sum", type=int, default=None,
                    help="If set (0..18), train a single BACON head for 'sum==K'. "
                         "If None, train 19 BACON heads for all sums.")

    # Partial/noisy supervision (multi-sum only)
    ap.add_argument(
        "--supervised-sums",
        type=str,
        default=None,
        help="Comma-separated list of sums (0-18) with clean labels. "
             "Others are treated according to --unsup-mode. "
             "Ignored when --single-sum is not None."
    )
    ap.add_argument(
        "--unsup-mode",
        type=str,
        default="clean",
        choices=["clean", "ignore", "noise"],
        help="Behavior for sums not in --supervised-sums in multi-sum mode: "
             "'clean' = treat all labels as clean (original behavior); "
             "'ignore' = exclude them from CE loss; "
             "'noise' = replace their labels with random sums."
    )
    ap.add_argument(
        "--lambda-consistency",
        default=0.1,
        type=float,
        help="Weight for consistency regularization loss (y(x1,x2) ~ y(x2,x1)).",
    )

    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Parse supervised sums (multi-sum only)
    if args.single_sum is not None and args.supervised_sums is not None:
        print("Warning: --supervised-sums is ignored in --single-sum mode.")
        supervised_sums = None
        supervised_sums_tensor = None
    else:
        if args.supervised_sums is not None:
            sup_list = [s for s in args.supervised_sums.split(",") if s.strip() != ""]
            supervised_sums = sorted(int(s) for s in sup_list)
            print(f"Using supervised SUM labels: {supervised_sums} "
                  f"| unsup_mode={args.unsup_mode}")
        else:
            supervised_sums = None
        supervised_sums_tensor = None

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
        single_sum=args.single_sum,
        freeze_cnn=args.freeze_cnn,
    ).to(device)

    # Prepare supervised_sums_tensor AFTER device is known
    if supervised_sums is not None:
        supervised_sums_tensor = torch.tensor(supervised_sums, device=device)

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
            tot, correct = 0, 0
            if args.single_sum is None:
                # multi-sum accuracy over all sums
                for (x1, x2), y in test_loader:
                    x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                    y_prob, _ = model(x1, x2, tau=args.tau_end, hard=True)  # [B,19]
                    pred = y_prob.argmax(dim=1)
                    correct += (pred == y).sum().item()
                    tot += y.size(0)
                acc_all = correct / max(1, tot)
                print(f"Pretrained eval | acc_all={acc_all*100:.2f}%")
            else:
                # single-sum binary accuracy
                K = args.single_sum
                for (x1, x2), y in test_loader:
                    x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                    y_prob, _ = model(x1, x2, tau=args.tau_end, hard=True)  # [B]
                    y_bin = (y == K)
                    pred_bin = (y_prob >= 0.5)
                    correct += (pred_bin == y_bin).sum().item()
                    tot += y.size(0)
                acc_bin = correct / max(1, tot)
                print(f"Pretrained eval | acc_bin(sum=={K})={acc_bin*100:.2f}%")

        print_soft_rules_vs_gt(model, test_loader, tau_eval=args.tau_end, device=device)
        return

    # ------- Training setup -------
    ce_multiclass = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=1e-4
    )

    best_score = 0.0
    bad_epochs = 0

    # Decide base save name (used if --save-path not set)
    if args.save_path is not None:
        base_save_path = args.save_path
    else:
        if args.single_sum is None:
            base_save_path = "best_bacon_mnist_addition_multiclass.pt"
        else:
            base_save_path = f"best_bacon_mnist_addition_single_sum{args.single_sum}.pt"

    # ------- Training loop -------
    for epoch in range(1, args.epochs + 1):
        t = (epoch - 1) / max(1, args.epochs - 1)
        tau = args.tau_start + (args.tau_end - args.tau_start) * t

        model.train()
        running_loss, seen = 0.0, 0

        for (x1, x2), y in train_loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)

            y_prob, c_prob = model(x1, x2, tau=tau, hard=False)
            y_true = y.clone()

            # ----- Main supervised loss -----
            if args.single_sum is not None:
                # Binary "sum == K?" supervision
                K = args.single_sum
                y_bin = (y_true == K).float()       # [B]
                p = y_prob.clamp(1e-6, 1 - 1e-6)    # [B]
                log_p = torch.log(p)
                log_1_p = torch.log(1.0 - p)
                loss_main = -(y_bin * log_p + (1.0 - y_bin) * log_1_p).mean()
            else:
                # Multi-sum mode with optional partial/noisy supervision
                logits = torch.logit(
                    y_prob.clamp(1e-6, 1 - 1e-6)
                )  # [B,19] pseudo-logits

                if supervised_sums is not None and args.unsup_mode in ("ignore", "noise"):
                    supervised_mask = torch.isin(y_true, supervised_sums_tensor)

                    if args.unsup_mode == "ignore":
                        if supervised_mask.any():
                            loss_main = ce_multiclass(
                                logits[supervised_mask], y_true[supervised_mask]
                            )
                        else:
                            loss_main = None  # no CE this batch

                    elif args.unsup_mode == "noise":
                        unsup_mask = ~supervised_mask
                        if unsup_mask.any():
                            rand_labels = torch.randint(
                                low=0, high=19,
                                size=(unsup_mask.sum().item(),),
                                device=y_true.device,
                            )
                            y_true[unsup_mask] = rand_labels
                        loss_main = ce_multiclass(logits, y_true)
                else:
                    # All labels clean (original behavior)
                    loss_main = ce_multiclass(logits, y_true)

            # ----- Regularizers -----
            loss_ent = group_entropy_loss(c_prob) * args.entropy

            # Consistency: y(x1,x2) ~ y(x2,x1)
            y_ab, _ = model(x1, x2, tau)
            y_ba, _ = model(x2, x1, tau)
            loss_consistency = F.mse_loss(y_ab, y_ba)

            if loss_main is None:
                loss = loss_ent + args.lambda_consistency * loss_consistency
            else:
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
            tot_all, correct_all = 0, 0
            tot_sup, correct_sup = 0, 0
            tot_uns, correct_uns = 0, 0

            for (x1, x2), y in test_loader:
                x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                y_prob_eval, _ = model(x1, x2, tau=args.tau_end, hard=True)

                if args.single_sum is not None:
                    K = args.single_sum
                    y_bin = (y == K)
                    pred_bin = (y_prob_eval >= 0.5)
                    correct_all += (pred_bin == y_bin).sum().item()
                    tot_all += y.size(0)
                else:
                    pred = y_prob_eval.argmax(dim=1)
                    correct_all += (pred == y).sum().item()
                    tot_all += y.size(0)

                    if supervised_sums is not None:
                        mask_sup = torch.isin(y, supervised_sums_tensor)
                        mask_uns = ~mask_sup
                        if mask_sup.any():
                            correct_sup += (pred[mask_sup] == y[mask_sup]).sum().item()
                            tot_sup += mask_sup.sum().item()
                        if mask_uns.any():
                            correct_uns += (pred[mask_uns] == y[mask_uns]).sum().item()
                            tot_uns += mask_uns.sum().item()

        if args.single_sum is not None:
            acc_all = correct_all / max(1, tot_all)
            print(
                f"Epoch {epoch:03d} | train_loss={train_loss:.4f} "
                f"| acc_bin(sum=={args.single_sum})={acc_all*100:.2f}% "
                f"| tau={tau:.3f}"
            )
            score = acc_all
        else:
            acc_all = correct_all / max(1, tot_all)
            if supervised_sums is not None and tot_sup > 0 and tot_uns > 0:
                acc_sup = correct_sup / max(1, tot_sup)
                acc_uns = correct_uns / max(1, tot_uns)
                print(
                    f"Epoch {epoch:03d} | train_loss={train_loss:.4f} "
                    f"| acc_all={acc_all*100:.2f}% "
                    f"| acc_sup={acc_sup*100:.2f}% "
                    f"| acc_uns={acc_uns*100:.2f}% "
                    f"| tau={tau:.3f}"
                )
            else:
                print(
                    f"Epoch {epoch:03d} | train_loss={train_loss:.4f} "
                    f"| acc_all={acc_all*100:.2f}% | tau={tau:.3f}"
                )
            score = acc_all

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
    else:
        print(f"\nBest acc_bin(sum=={args.single_sum}): {best_score*100:.2f}% "
              f"(saved to {base_save_path})")

    print_soft_rules_vs_gt(model, test_loader, tau_eval=args.tau_end, device=device)


if __name__ == "__main__":
    main()
