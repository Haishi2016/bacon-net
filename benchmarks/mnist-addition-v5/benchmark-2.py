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
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..')))
from bacon.binaryTreeLogicNet import binaryTreeLogicNet
from bacon.aggregators.bool import MinMaxAggregator


# ------------------------------------------------------------
# Streaming Dataset (for training)
# ------------------------------------------------------------
class MnistAdditionStream(IterableDataset):
    """
    On-the-fly sampled pairs (x1, x2) with label sum = d1 + d2.
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
            )
        )

    def __iter__(self) -> Iterator:
        worker_info = torch.utils.data.get_worker_info()
        base_seed = self.seed + (0 if worker_info is None else worker_info.id)
        rng = random.Random(base_seed)
        n = len(self.mnist)
        for _ in range(self.epoch_len):
            i1 = rng.randrange(n)
            i2 = rng.randrange(n)
            x1, d1 = self.mnist[i1]
            x2, d2 = self.mnist[i2]
            yield (x1, x2), int(d1 + d2)


# ------------------------------------------------------------
# Fixed Dataset (for evaluation)
# ------------------------------------------------------------
class MnistAdditionFixed(Dataset):
    """
    Fixed list of MNIST pairs for stable test evaluation.
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

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        i1, i2, s = self.pairs[idx]
        x1, _ = self.mnist[i1]
        x2, _ = self.mnist[i2]
        return (x1, x2), s


# ------------------------------------------------------------
# Shared small CNN tower
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
        """
        Returns a [B,10] distribution over digits using Gumbel-Softmax.
        """
        logits = self.logit_head(features)
        probs = F.gumbel_softmax(logits, tau=tau, hard=hard, dim=1)
        return probs


# ------------------------------------------------------------
# Reasoner Heads
# ------------------------------------------------------------
class BaconMultiSumHead(nn.Module):
    """
    19 separate BACON trees. Each sees the same 20D concept vector:
    [p1(0..9), p2(0..9)]. The k-th tree learns “sum == k” as a graded formula.
    """
    def __init__(self, input_size=20):
        super().__init__()
        agg = MinMaxAggregator()
        self.trees = nn.ModuleList([
            binaryTreeLogicNet(
                input_size=input_size,
                is_frozen=False,
                weight_mode="fixed",
                weight_normalization="minmax",
                aggregator=agg,
                normalize_andness=False,
                tree_layout="paired",
                loss_amplifier=1.0,
            )
            for _ in range(19)
        ])

    def forward(self, p1, p2, c20):
        # c20 is [B,20], but we recompute it here for clarity of interface.
        x = c20  # [B,20]
        ys = []
        for tree in self.trees:
            yk = tree(x)  # expect [B] or [B,1]
            if yk.dim() == 1:
                yk = yk.unsqueeze(1)
            ys.append(yk)
        y = torch.cat(ys, dim=1)  # [B,19]
        y = y.clamp(1e-6, 1 - 1e-6)
        return y


class OuterProductReasoner(nn.Module):
    """
    LTN / DeepProbLog / DCR-style differentiable arithmetic head:
    - Interpret p1, p2 as distributions over digits.
    - Build outer product p1[i] * p2[j].
    - For each sum k, aggregate valid pairs {i+j=k} via fuzzy OR:
        1 - ∏(1 - p1[i]*p2[j]).
    This is not a faithful reproduction of each framework's internals,
    but gives a common differentiable arithmetic head for comparison.
    """
    def __init__(self, mode: str = "ltn"):
        super().__init__()
        self.mode = mode  # just for labeling / logging

        # Precompute mask[k,i,j] = 1 if i+j=k, else 0
        mask = torch.zeros(19, 10, 10)
        for k in range(19):
            for i in range(10):
                j = k - i
                if 0 <= j < 10:
                    mask[k, i, j] = 1.0
        self.register_buffer("pair_mask", mask)  # [19,10,10]

    def forward(self, p1, p2, c20):
        B = p1.size(0)
        device = p1.device
        outer = p1.unsqueeze(2) * p2.unsqueeze(1)  # [B,10,10]

        mask = self.pair_mask.to(device).unsqueeze(0)  # [1,19,10,10]
        s = outer.unsqueeze(1) * mask                 # [B,19,10,10]

        # fuzzy OR: 1 - prod(1 - s_ij) over valid pairs
        one_minus = (1.0 - s.clamp(1e-6, 1.0 - 1e-6))
        log_one_minus = (one_minus + 1e-12).log()
        log_prod = log_one_minus.sum(dim=(2, 3))  # [B,19]
        y = 1.0 - torch.exp(log_prod)
        y = y.clamp(1e-6, 1 - 1e-6)
        return y


# ------------------------------------------------------------
# Full model: CNN + concepts + reasoner
# ------------------------------------------------------------
class AdditionModel(nn.Module):
    def __init__(self, reasoner: str = "bacon"):
        super().__init__()
        self.tower = SmallCnn(128)
        self.head = ImageConceptHead(128, 10)

        reasoner = reasoner.lower()
        if reasoner == "bacon":
            self.reasoner_name = "bacon"
            self.reasoner = BaconMultiSumHead(input_size=20)
        elif reasoner in ("ltn", "deepproblog", "dcr"):
            self.reasoner_name = reasoner
            self.reasoner = OuterProductReasoner(mode=reasoner)
        else:
            raise ValueError(f"Unknown reasoner: {reasoner}")

    def forward(self, x1, x2, tau: float, hard: bool = False):
        # Shared CNN
        z1 = self.tower(x1)
        z2 = self.tower(x2)

        # Concepts for each image
        p1 = self.head(z1, tau=tau, hard=hard)  # [B,10]
        p2 = self.head(z2, tau=tau, hard=hard)  # [B,10]

        # 20-D concept vector for regularization etc.
        c20 = torch.cat([p1, p2], dim=1)        # [B,20]

        # Reasoner head
        y_prob = self.reasoner(p1, p2, c20)     # [B,19] in (0,1)

        return y_prob, c20


# ------------------------------------------------------------
# Utils: entropy regularizer & rule visualization
# ------------------------------------------------------------
def group_entropy_loss(c_prob: torch.Tensor) -> torch.Tensor:
    """
    Entropy over p1 and p2 separately (each 10-D), normalized to [0,1] and averaged.
    """
    eps = 1e-8
    g1 = c_prob[:, :10]
    g2 = c_prob[:, 10:]
    ent1 = -(g1 * (g1 + eps).log()).sum(dim=1)
    ent2 = -(g2 * (g2 + eps).log()).sum(dim=1)
    ent = ent1 + ent2
    return (ent / math.log(10)).mean()


def print_soft_rule_attribution_with_gt(
    model: AdditionModel,
    test_loader,
    tau_eval: float = 0.8,
    device: str = "cpu",
    top_k: int = 5,
):
    """
    Soft rule attribution:
    - For each test batch, get hard concepts (argmax) for both digits.
    - Compute outer product p1[i]*p2[j] as an indication of pair assignment.
    - Group contributions conditioned on the predicted sum y_hat.
    - For each k, we:
        - normalize contributions on valid pairs (i+j=k)
        - measure leakage: mass on invalid pairs
        - print top-k valid pairs with normalized contribution.

    This is independent of the specific reasoner and just looks at the
    concept-level manifold.
    """
    model.eval()
    pair_sum = torch.zeros(19, 10, 10, device=device)
    count_k = torch.zeros(19, device=device)

    with torch.no_grad():
        for (x1, x2), _ in test_loader:
            x1, x2 = x1.to(device), x2.to(device)

            # hard concepts from CNN + head
            z1 = model.tower(x1)
            z2 = model.tower(x2)
            p1 = model.head(z1, tau=tau_eval, hard=True)  # one-hot-ish
            p2 = model.head(z2, tau=tau_eval, hard=True)

            # predicted sum using full model
            y_prob, _ = model(x1, x2, tau=tau_eval, hard=True)
            k = y_prob.argmax(1)  # [B]

            outer = p1.unsqueeze(2) * p2.unsqueeze(1)  # [B,10,10]

            for kk in range(19):
                mask_batch = (k == kk).float().view(-1, 1, 1)  # [B,1,1]
                pair_sum[kk] += (outer * mask_batch).sum(dim=0)
                count_k[kk]  += mask_batch.sum()

    print("\n=== BACON soft-rule attribution vs ground truth manifold ===")
    for kk in range(19):
        if count_k[kk] == 0:
            print(f"\ny_{kk}: (no samples)")
            continue

        contrib = pair_sum[kk] / count_k[kk]  # [10,10]

        valid_pairs = []
        invalid_sum = 0.0
        for i in range(10):
            for j in range(10):
                v = contrib[i, j].item()
                if i + j == kk:
                    valid_pairs.append(((i, j), v))
                else:
                    invalid_sum += v

        total_valid = sum(v for (_, v) in valid_pairs) + 1e-12
        leakage = invalid_sum / (invalid_sum + total_valid)

        valid_pairs.sort(key=lambda x: x[1], reverse=True)
        top_valid = valid_pairs[:top_k]

        print(f"\ny_{kk}: leakage={leakage:.3f}")
        if not top_valid:
            print("  (no valid pairs with non-negligible mass)")
            continue

        normed = [(i, j, v / total_valid) for ((i, j), v) in top_valid]
        parts = [f"({i}∧{j}):{w:.2f}" for (i, j, w) in normed]
        print("  Top-{0} valid pairs by normalized contribution: {1}".format(
            top_k, " ∨ ".join(parts)
        ))

    print("============================================================\n")


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

    ap.add_argument("--reasoner", default="bacon",
                    choices=["bacon", "ltn", "deepproblog", "dcr"],
                    help="Choice of reasoning head on top of the CNN+concepts.")
    ap.add_argument("--freeze-cnn", action="store_true",
                    help="If set, freeze the CNN tower parameters.")

    # Partial / noisy supervision
    ap.add_argument(
        "--supervised-sums",
        type=str,
        default=None,
        help="Comma-separated list of sums (0-18) supervised with clean labels. "
             "Others handled via --unsup-mode. If omitted, all sums are supervised."
    )
    ap.add_argument(
        "--unsup-mode",
        type=str,
        default="clean",
        choices=["clean", "ignore", "noise"],
        help="For sums NOT in --supervised-sums: "
             "'clean'  = treat all as clean (original behavior); "
             "'ignore' = exclude them from CE loss; "
             "'noise'  = randomize their labels."
    )
    ap.add_argument("--lambda-consistency", default=0.0, type=float,
                    help="Weight for symmetry-consistency regularization "
                         "between f(x1,x2) and f(x2,x1).")

    # Checkpointing
    ap.add_argument("--pretrained", default=None, type=str,
                    help="If set, load this checkpoint and just evaluate + visualize.")
    ap.add_argument("--save-path", default="bacon_mnist_addition_multisum.pt", type=str)

    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device} | reasoner={args.reasoner}")

    # Parse supervised sums
    if args.supervised_sums is not None:
        sup_list = [s for s in args.supervised_sums.split(",") if s.strip() != ""]
        supervised_sums: Optional[List[int]] = sorted(int(s) for s in sup_list)
        print(f"Using supervised SUM labels: {supervised_sums} | unsup_mode={args.unsup_mode}")
        supervised_sums_tensor = torch.tensor(supervised_sums, device=device)
    else:
        supervised_sums = None
        supervised_sums_tensor = None
        print("All sums supervised as clean (original behavior).")

    # Data
    train_ds = MnistAdditionStream(args.data, train=True,
                                   epoch_len=args.train_pairs,
                                   seed=args.seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        num_workers=2,
    )

    test_ds = MnistAdditionFixed(args.data, train=False,
                                 size_pairs=args.test_pairs,
                                 seed=999)
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )

    # Model
    model = AdditionModel(reasoner=args.reasoner).to(device)

    if args.freeze_cnn:
        for p in model.tower.parameters():
            p.requires_grad = False
        print("CNN tower is frozen (no gradient updates).")

    # If pretrained: load & eval only
    if args.pretrained is not None:
        if not os.path.isfile(args.pretrained):
            raise FileNotFoundError(f"Checkpoint not found: {args.pretrained}")
        ckpt = torch.load(args.pretrained, map_location=device)
        state_dict = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
        model.load_state_dict(state_dict, strict=False)
        print(f"Loaded pretrained model from {args.pretrained}")

        # Evaluate
        model.eval()
        with torch.no_grad():
            tot, correct = 0, 0
            for (x1, x2), y in test_loader:
                x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                y_prob, _ = model(x1, x2, tau=args.tau_end, hard=True)
                pred = y_prob.argmax(1)
                correct += (pred == y).sum().item()
                tot += y.size(0)
        acc = correct / max(1, tot)
        print(f"Pretrained eval | overall acc_all={acc * 100:.2f}% | tau={args.tau_end:.3f}")
        print_soft_rule_attribution_with_gt(model, test_loader,
                                            tau_eval=args.tau_end,
                                            device=device)
        return

    # Training setup
    ce = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-4)

    best_acc_all = 0.0
    bad_epochs = 0
    last_eval_acc = None

    for epoch in range(1, args.epochs + 1):
        # Linear tau schedule
        t = (epoch - 1) / max(1, args.epochs - 1)
        tau = args.tau_start + (args.tau_end - args.tau_start) * t

        model.train()
        running_loss, seen = 0.0, 0

        for (x1, x2), y in train_loader:
            x1, x2, y = x1.to(device), x2.to(device), y.to(device)

            # Forward
            y_prob, c_prob = model(x1, x2, tau=tau, hard=False)

            # Main label tensor we can mutate in unsup-mode='noise'
            y_true = y.clone()

            # ---------- partial / noisy supervision ----------
            loss_main = None

            if supervised_sums is not None and args.unsup_mode in ("ignore", "noise"):
                supervised_mask = torch.isin(y_true, supervised_sums_tensor)

                if args.unsup_mode == "ignore":
                    logits = torch.logit(y_prob.clamp(1e-6, 1 - 1e-6))
                    if supervised_mask.any():
                        loss_main = ce(logits[supervised_mask], y_true[supervised_mask])
                    else:
                        loss_main = None  # this batch contributes only via regularizers

                elif args.unsup_mode == "noise":
                    unsup_mask = ~supervised_mask
                    if unsup_mask.any():
                        rand_labels = torch.randint(
                            low=0, high=19,
                            size=(unsup_mask.sum(),),
                            device=y_true.device,
                        )
                        y_true[unsup_mask] = rand_labels
                    logits = torch.logit(y_prob.clamp(1e-6, 1 - 1e-6))
                    loss_main = ce(logits, y_true)

            else:
                # Original “all clean” behavior
                logits = torch.logit(y_prob.clamp(1e-6, 1 - 1e-6))
                loss_main = ce(logits, y_true)
            # ------------------------------------------------

            # Entropy regularizer (always over full batch)
            loss_ent = group_entropy_loss(c_prob) * args.entropy

            # Optional consistency regularizer: f(x1,x2) ≈ f(x2,x1)
            if args.lambda_consistency > 0.0:
                y_ab, _ = model(x1, x2, tau=tau, hard=False)
                y_ba, _ = model(x2, x1, tau=tau, hard=False)
                loss_cons = F.mse_loss(y_ab, y_ba)
            else:
                loss_cons = torch.tensor(0.0, device=device)

            if loss_main is None:
                loss = loss_ent + args.lambda_consistency * loss_cons
            else:
                loss = loss_main + loss_ent + args.lambda_consistency * loss_cons

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

        # Eval on full 0..18 sums
        model.eval()
        with torch.no_grad():
            tot_all, correct_all = 0, 0
            tot_sup, correct_sup = 0, 0
            tot_uns, correct_uns = 0, 0

            for (x1, x2), y in test_loader:
                x1, x2, y = x1.to(device), x2.to(device), y.to(device)
                y_prob, _ = model(x1, x2, tau=args.tau_end, hard=True)
                pred = y_prob.argmax(1)

                correct = (pred == y)
                bs = y.size(0)

                tot_all += bs
                correct_all += correct.sum().item()

                if supervised_sums is not None:
                    mask_sup = torch.isin(y, supervised_sums_tensor)
                    mask_uns = ~mask_sup

                    if mask_sup.any():
                        tot_sup += mask_sup.sum().item()
                        correct_sup += correct[mask_sup].sum().item()
                    if mask_uns.any():
                        tot_uns += mask_uns.sum().item()
                        correct_uns += correct[mask_uns].sum().item()

        acc_all = correct_all / max(1, tot_all)
        acc_sup = (correct_sup / max(1, tot_sup)) if tot_sup > 0 else 0.0
        acc_uns = (correct_uns / max(1, tot_uns)) if tot_uns > 0 else 0.0

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} "
              f"| acc_all={acc_all * 100:.2f}% "
              f"| acc_sup={acc_sup * 100:.2f}% "
              f"| acc_uns={acc_uns * 100:.2f}% "
              f"| tau={tau:.3f}")

        last_eval_acc = acc_all

        # Early stopping & checkpointing based on acc_all
        if acc_all > best_acc_all:
            best_acc_all = acc_all
            bad_epochs = 0
            torch.save(
                {"state_dict": model.state_dict(), "args": vars(args)},
                args.save_path,
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping (no improvement for {args.patience} epochs).")
                break

    print(f"\nBest acc_all: {best_acc_all * 100:.2f}% (checkpoint: {args.save_path})")
    print_soft_rule_attribution_with_gt(
        model, test_loader, tau_eval=args.tau_end, device=device
    )


if __name__ == "__main__":
    main()
