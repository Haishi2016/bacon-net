# mnist-addition-v4/train_addition_v4.py

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, math, random, os, sys
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from scipy.optimize import linear_sum_assignment

from datasets import MnistAdditionStream, MnistAdditionFixed
from bacon_model_v4 import BaconModelV4, group_entropy_loss

# For perm-search
from bacon.frozonInputToLeaf import frozenInputToLeaf


def print_pair_rules_all(model, test_loader, tau_eval=0.8, device='cpu'):
    model.eval()
    pair_sum = torch.zeros(19, 10, 10, device=device)
    count_k = torch.zeros(19, device=device)
    with torch.no_grad():
        for (x1, x2), _ in test_loader:
            x1, x2 = x1.to(device), x2.to(device)
            z1 = model.tower1(x1); z2 = model.tower2(x2)
            p1 = model.head1(z1, tau=tau_eval, hard=True)
            p2 = model.head2(z2, tau=tau_eval, hard=True)
            y_prob, _ = model(x1, x2, tau=tau_eval, hard=True)
            k = y_prob.argmax(1)
            outer = p1.unsqueeze(2) * p2.unsqueeze(1)
            for kk in range(19):
                mask = (k == kk).float().view(-1, 1, 1)
                pair_sum[kk] += (outer * mask).sum(dim=0)
                count_k[kk] += mask.sum()

    print("\n=== Pairwise rules (valid pairs i∧j with i+j=k) ===")
    for kk in range(19):
        if count_k[kk] == 0:
            print(f"y_{kk}: (no samples)")
            continue
        contrib = pair_sum[kk] / count_k[kk]
        pairs = [(i, j, contrib[i, j].item()) for i in range(10) for j in range(10)]
        pairs.sort(key=lambda t: t[2], reverse=True)
        valid = [(i, j, v) for (i, j, v) in pairs if i + j == kk]
        human = " ∨ ".join([f"({i}∧{j})" for (i, j, _) in valid]) or "(none)"
        print(f"y_{kk}: {human}")
    print("===============================================\n")


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

    # BACON-specific gates
    ap.add_argument("--auto-refine", action="store_true",
                    help="enable gated hard concepts (optional)")
    ap.add_argument("--refine-tau-gate", default=None, type=float,
                    help="enable hard concepts only when tau <= this value")
    ap.add_argument("--refine-acc-gate", default=None, type=float,
                    help="enable hard concepts only when eval acc >= this value (0..1)")
    ap.add_argument("--perm-search", action="store_true",
                    help="enable permutation search on perm1/perm2 (optional)")
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

    train_ds = MnistAdditionStream(args.data, train=True,
                                   epoch_len=args.train_pairs, seed=args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, num_workers=2)

    test_ds = MnistAdditionFixed(args.data, train=False,
                                 size_pairs=args.test_pairs, seed=999)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,
                             shuffle=False, num_workers=2, pin_memory=True)

    model = BaconModelV4().to(device)

    # Pretrained (optional) -- same logic as v1
    if args.pretrained is not None:
        if not os.path.isfile(args.pretrained):
            raise FileNotFoundError(f"Checkpoint not found: {args.pretrained}")
        try:
            ckpt = torch.load(args.pretrained, map_location=device, weights_only=True)
        except TypeError:
            ckpt = torch.load(args.pretrained, map_location=device)
        except Exception as e:
            print(f"Safe load (weights_only=True) failed: {e}. Falling back to standard torch.load (unsafe).")
            ckpt = torch.load(args.pretrained, map_location=device)

        if isinstance(ckpt, dict) and "state_dict" in ckpt:
            state_dict = ckpt["state_dict"]
        elif hasattr(ckpt, "state_dict"):
            state_dict = ckpt.state_dict()
        else:
            state_dict = ckpt

        cleaned = { (k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items() }
        missing, unexpected = model.load_state_dict(cleaned, strict=False)
        if missing:
            print(f"Warning: missing keys: {sorted(missing)}")
        if unexpected:
            print(f"Warning: unexpected keys: {sorted(unexpected)}")

        model.eval()
        with torch.no_grad():
            tot, correct = 0, 0
            for (x1, x2), y in test_loader:
                x1, x2 = x1.to(device), x2.to(device)
                y = y.to(device)
                y_prob, _ = model(x1, x2, tau=args.tau_end, hard=True)
                pred = y_prob.argmax(1)
                correct += (pred == y).sum().item()
                tot += y.size(0)
            acc = correct / max(1, tot)
        print(f"Pretrained eval | multiclass acc={acc*100:.2f}% | tau={args.tau_end:.3f}")
        print_pair_rules_all(model, test_loader, tau_eval=args.tau_end, device=device)
        return

    # Loss/optim
    ce = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=1e-4)

    best_acc = 0.0
    bad_epochs = 0
    last_eval_acc = None

    for epoch in range(1, args.epochs + 1):
        t = (epoch - 1) / max(1, args.epochs - 1)
        tau = args.tau_start + (args.tau_end - args.tau_start) * t

        # ---------------- Train ----------------
        model.train()
        running_loss, seen = 0.0, 0

        for (x1, x2), y in train_loader:
            x1, x2 = x1.to(device), x2.to(device)
            y = y.to(device)

            # Optional external gating: switch to hard concepts when gate conditions are met
            use_hard = False

            y_prob, c_prob = model(x1, x2, tau=tau, hard=use_hard)

            logits = torch.logit(y_prob.clamp(1e-6, 1 - 1e-6))
            loss_main = ce(logits, y)
            loss_ent = group_entropy_loss(c_prob) * args.entropy
            loss = loss_main + loss_ent

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

        # ---------------- Eval ----------------
        model.eval()
        with torch.no_grad():
            tot, correct = 0, 0
            for (x1, x2), y in test_loader:
                x1, x2 = x1.to(device), x2.to(device)
                y = y.to(device)
                y_prob, _ = model(x1, x2, tau=args.tau_end, hard=True)
                pred = y_prob.argmax(1)
                correct += (pred == y).sum().item()
                tot += y.size(0)
            acc = correct / max(1, tot)

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.4f} | acc={acc*100:.2f}% | tau={tau:.3f}")
        last_eval_acc = acc

        # ---------------- Optional permutation search (same as v1) ----------------
        if args.perm_search:
            cond = True
            if args.perm_tau_gate is not None:
                cond = cond and (tau <= args.perm_tau_gate)
            if args.perm_acc_gate is not None:
                cond = cond and (acc >= float(args.perm_acc_gate))

            if cond:
                try:
                    (px1, px2), py = next(iter(test_loader))
                except StopIteration:
                    px1, px2, py = None, None, None

                if px1 is not None:
                    px1, px2, py = px1.to(device), px2.to(device), py.to(device)

                    # Cache current logits from perm1/perm2
                    def get_logits(module):
                        itl = getattr(module, "input_to_leaf", None)
                        return getattr(itl, "logits", None)

                    L1 = get_logits(model.perm1)
                    L2 = get_logits(model.perm2)

                    if L1 is not None and L2 is not None:
                        def sinkhorn(logits, n_iters=20, temperature=1.0):
                            P = (logits / temperature).softmax(dim=1)
                            for _ in range(n_iters):
                                P = P / (P.sum(dim=1, keepdim=True) + 1e-12)
                                P = P / (P.sum(dim=0, keepdim=True) + 1e-12)
                            return P

                        def sample_perm(logits, noise_std):
                            noisy = logits + torch.randn_like(logits) * noise_std
                            P = sinkhorn(noisy)
                            P_np = P.detach().cpu().numpy()
                            row_ind, col_ind = linear_sum_assignment(-P_np)
                            perm = col_ind[row_ind.argsort()]
                            return torch.tensor(perm, device=logits.device, dtype=torch.long)

                        # Baseline loss
                        y_prob_base, _ = model(px1, px2, tau=args.tau_end, hard=True)
                        logits_base = torch.logit(y_prob_base.clamp(1e-6, 1 - 1e-6))
                        ce_eval = nn.CrossEntropyLoss()
                        best_loss = ce_eval(logits_base, py).item()
                        best_p1 = None
                        best_p2 = None

                        # Search perm1 keeping perm2 fixed
                        for _ in range(int(args.perm_k)):
                            p1_perm = sample_perm(L1, args.perm_noise)
                            with torch.no_grad():
                                z1 = model.tower1(px1); z2 = model.tower2(px2)
                                p1 = model.head1(z1, tau=args.tau_end, hard=True)
                                p2 = model.head2(z2, tau=args.tau_end, hard=True)
                                p1p = p1.index_select(1, p1_perm)
                                p2p = model.perm2.input_to_leaf(p2)

                                B = p1p.size(0)
                                x = p1p.unsqueeze(2).expand(B, 10, 10)
                                yv = p2p.unsqueeze(1).expand(B, 10, 10)
                                a_and = torch.tensor(1.0, device=x.device)
                                s = model.agg.aggregate_tensor(x, yv, a_and, w0=0.5, w1=0.5)

                                mask = s.new_zeros(19, 10, 10)
                                for kk in range(19):
                                    for i in range(10):
                                        j = kk - i
                                        if 0 <= j <= 9:
                                            mask[kk, i, j] = 1.0

                                one_minus = (1.0 - s.clamp(1e-6, 1 - 1e-6))
                                log_one_minus = (one_minus + 1e-12).log()
                                y_list = []
                                for kk in range(19):
                                    mk = mask[kk].unsqueeze(0).expand(B, 10, 10)
                                    log_prod_k = (log_one_minus * mk).sum(dim=(1, 2))
                                    yk = 1.0 - torch.exp(log_prod_k)
                                    y_list.append(yk.unsqueeze(1))
                                y_prob_cand = torch.cat(y_list, dim=1)
                                logits_cand = torch.logit(y_prob_cand.clamp(1e-6, 1 - 1e-6))
                                loss_cand = ce_eval(logits_cand, py).item()

                                if loss_cand < best_loss:
                                    best_loss = loss_cand
                                    best_p1 = p1_perm

                        # Search perm2 keeping perm1 fixed
                        for _ in range(int(args.perm_k)):
                            p2_perm = sample_perm(L2, args.perm_noise)
                            with torch.no_grad():
                                z1 = model.tower1(px1); z2 = model.tower2(px2)
                                p1 = model.head1(z1, tau=args.tau_end, hard=True)
                                p2 = model.head2(z2, tau=args.tau_end, hard=True)
                                p1p = model.perm1.input_to_leaf(p1)
                                p2p = p2.index_select(1, p2_perm)

                                B = p1p.size(0)
                                x = p1p.unsqueeze(2).expand(B, 10, 10)
                                yv = p2p.unsqueeze(1).expand(B, 10, 10)
                                a_and = torch.tensor(1.0, device=x.device)
                                s = model.agg.aggregate_tensor(x, yv, a_and, w0=0.5, w1=0.5)

                                mask = s.new_zeros(19, 10, 10)
                                for kk in range(19):
                                    for i in range(10):
                                        j = kk - i
                                        if 0 <= j <= 9:
                                            mask[kk, i, j] = 1.0

                                one_minus = (1.0 - s.clamp(1e-6, 1 - 1e-6))
                                log_one_minus = (one_minus + 1e-12).log()
                                y_list = []
                                for kk in range(19):
                                    mk = mask[kk].unsqueeze(0).expand(B, 10, 10)
                                    log_prod_k = (log_one_minus * mk).sum(dim=(1, 2))
                                    yk = 1.0 - torch.exp(log_prod_k)
                                    y_list.append(yk.unsqueeze(1))
                                y_prob_cand = torch.cat(y_list, dim=1)
                                logits_cand = torch.logit(y_prob_cand.clamp(1e-6, 1 - 1e-6))
                                loss_cand = ce_eval(logits_cand, py).item()

                                if loss_cand < best_loss:
                                    best_loss = loss_cand
                                    best_p2 = p2_perm

                        # Apply best perms if any improvement
                        if best_p1 is not None:
                            model.perm1.input_to_leaf = frozenInputToLeaf(best_p1, 10)
                            model.perm1.is_frozen = True
                        if best_p2 is not None:
                            model.perm2.input_to_leaf = frozenInputToLeaf(best_p2, 10)
                            model.perm2.is_frozen = True

        # Early stopping
        if acc > best_acc:
            best_acc = acc
            bad_epochs = 0
            torch.save(model.state_dict(), "best_bacon_mnist_addition_multiclass_v4.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"Early stopping (no improvement for {args.patience} epochs).")
                break

    print(f"\nBest acc: {best_acc*100:.2f}% (checkpoint: best_bacon_mnist_addition_multiclass_v4.pt)")
    print_pair_rules_all(model, test_loader, tau_eval=args.tau_end, device=device)


if __name__ == "__main__":
    main()
