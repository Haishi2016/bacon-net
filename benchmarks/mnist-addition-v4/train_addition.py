# mnist-addition-v4/train_addition_v4.py

import argparse
import math
import os
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from datasets import MnistAdditionStream, MnistAdditionFixed
from backbone import AdditionConceptBackbone
from reasoners import REASONER_REGISTRY
from reasoners.bacon_reasoner import group_entropy_loss  # reuse util


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
    ap.add_argument("--pretrained", default=None, type=str)

    # new: reasoner choice (for now only "bacon")
    ap.add_argument("--reasoner", default="bacon", choices=REASONER_REGISTRY.keys())

    # BACON-specific options kept for future use (e.g. auto-refine, perm-search)
    ap.add_argument("--auto-refine", action="store_true",
                    help="enable gated hard concepts (optional, BACON only)")
    ap.add_argument("--refine-tau-gate", default=None, type=float,
                    help="enable hard concepts only when tau <= this value")
    ap.add_argument("--refine-acc-gate", default=None, type=float,
                    help="enable hard concepts only when eval acc >= this value (0..1)")
    ap.add_argument("--perm-search", action="store_true",
                    help="enable permutation search on perm1/perm2 (BACON only)")
    ap.add_argument("--perm-k", default=32, type=int,
                    help="num candidate perms to try per search")
    ap.add_argument("--perm-noise", default=0.1, type=float,
                    help="noise std added to logits before Hungarian")
    ap.add_argument("--perm-acc-gate", default=None, type=float,
                    help="run perm search only when acc >= this")
    ap.add_argument("--perm-tau-gate", default=None, type=float,
                    help="run perm search only when tau <= this")

    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = MnistAdditionStream(
        args.data, train=True, epoch_len=args.train_pairs, seed=args.seed
    )
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=2)

    test_ds = MnistAdditionFixed(
        args.data, train=False, size_pairs=args.test_pairs, seed=999
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True
    )

    # Shared backbone (SmallCnn + Gumbel concept heads)
    backbone = AdditionConceptBackbone(feat_dim=128, n_concepts=10).to(device)

    # Reasoner (for now: only BACON)
    ReasonerCls = REASONER_REGISTRY[args.reasoner]
    reasoner = ReasonerCls().to(device)

    print(f"Using device: {device}")
    print(f"Reasoner: {reasoner.__class__.__name__}")

    # If you want to support --pretrained later, you can extend here
    if args.pretrained is not None:
        if not os.path.isfile(args.pretrained):
            raise FileNotFoundError(f"Checkpoint not found: {args.pretrained}")
        ckpt = torch.load(args.pretrained, map_location=device)
        # Expecting state_dict with 'backbone' and 'reasoner'
        if "backbone" in ckpt and "reasoner" in ckpt:
            backbone.load_state_dict(ckpt["backbone"])
            reasoner.load_state_dict(ckpt["reasoner"])
        else:
            # Fallback: assume full state_dict for a combined model (if you later make one)
            backbone.load_state_dict(ckpt, strict=False)
            reasoner.load_state_dict(ckpt, strict=False)

        # Eval only
        backbone.eval()
        reasoner.eval()
        with torch.no_grad():
            tot, correct = 0, 0
            for (x1, x2), y in test_loader:
                x1, x2 = x1.to(device), x2.to(device)
                y = y.to(device)
                p1, p2 = backbone(x1, x2, tau=args.tau_end, hard=True)
                y_prob = reasoner(p1, p2)
                pred = y_prob.argmax(1)
                correct += (pred == y).sum().item()
                tot += y.size(0)
            acc = correct / max(1, tot)
        print(f"Pretrained eval | acc={acc*100:.2f}% | tau={args.tau_end:.3f}")
        return

    # Optim / sched (joint backbone + reasoner)
    opt = torch.optim.AdamW(
        list(backbone.parameters()) + list(reasoner.parameters()),
        lr=args.lr,
        weight_decay=5e-4,
    )
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.epochs, eta_min=1e-4
    )

    best_acc = 0.0
    bad_epochs = 0
    last_eval_acc = None

    for epoch in range(1, args.epochs + 1):
        # τ linear annealing (same as v1)
        t = (epoch - 1) / max(1, args.epochs - 1)
        tau = args.tau_start + (args.tau_end - args.tau_start) * t

        # ---------------- Train ----------------
        backbone.train()
        reasoner.train()
        running_loss, seen = 0.0, 0

        for (x1, x2), y in train_loader:
            x1, x2 = x1.to(device), x2.to(device)
            y = y.to(device)

            # Optional external gating: switch to hard concepts when gate conditions are met
            use_hard = False
            if args.auto_refine:
                cond = True
                if args.refine_tau_gate is not None:
                    cond = cond and (tau <= args.refine_tau_gate)
                if args.refine_acc_gate is not None and last_eval_acc is not None:
                    cond = cond and (last_eval_acc >= float(args.refine_acc_gate))
                use_hard = cond

            # Backbone: get concept probabilities
            p1, p2 = backbone(x1, x2, tau=tau, hard=use_hard)   # [B,10] each

            # BACON loss (for now only reasoner = BACON)
            loss = reasoner.loss(p1, p2, y, entropy_weight=args.entropy)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(list(backbone.parameters()) + list(reasoner.parameters()), 1.0)
            opt.step()

            bs = x1.size(0)
            running_loss += float(loss.item()) * bs
            seen += bs
            if seen >= args.train_pairs:
                break

        sched.step()
        train_loss = running_loss / max(1, seen)

        # ---------------- Eval ----------------
        backbone.eval()
        reasoner.eval()
        with torch.no_grad():
            tot, correct = 0, 0
            for (x1, x2), y in test_loader:
                x1, x2 = x1.to(device), x2.to(device)
                y = y.to(device)
                p1, p2 = backbone(x1, x2, tau=args.tau_end, hard=True)
                y_prob = reasoner(p1, p2)
                pred = y_prob.argmax(1)
                correct += (pred == y).sum().item()
                tot += y.size(0)
            acc = correct / max(1, tot)

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | acc={acc*100:.2f}% | tau={tau:.3f}")
        last_eval_acc = acc

        # (Optional) you can re-insert perm_search block here later if needed,
        # adapting it to use backbone + reasoner.perm1/perm2.

        # Early stopping
        if acc > best_acc:
            best_acc = acc
            bad_epochs = 0
            torch.save(
                {
                    "backbone": backbone.state_dict(),
                    "reasoner": reasoner.state_dict(),
                    "epoch": epoch,
                    "best_acc": best_acc,
                    "args": vars(args),
                },
                "best_bacon_mnist_addition_v4.pt",
            )
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping (no improvement for {args.patience} epochs).")
                break

    print(f"\nBest acc: {best_acc*100:.2f}% (checkpoint: best_bacon_mnist_addition_v4.pt)")


if __name__ == "__main__":
    main()
