"""Faithful port of dsb-ifi/cibm ``src/train.py`` (core training recipe).

Reproduces ``configure_training`` + ``train_model`` for the three CUB settings:

* ``baseline`` -- deterministic ``BasicMLP``: loss = CE(y) + BCE(c).sum(1).mean().
* ``ibe``      -- stochastic IB-Expressive: loss = CE + BCE.sum(1).mean()
                  + beta*(MI_const - I(X;C)); ``beta += beta_lr*(MI_const-MI)``
                  (Lagrangian dual update each step).
* ``ibb``      -- stochastic IB-Bottleneck: loss = CE + (1+beta)*BCE.sum(1).mean();
                  (1-beta)*H_C minimised w.r.t. the encoder heads only; beta fixed.

Trains an MLP on the cached InceptionV3 embeddings (run
``precompute_embeddings.py`` first). Defaults match the official
``run_experiment``: Adam lr 1e-3 / wd 0, 20 epochs, batch 128, cosine schedule,
beta 0.5, beta_lr -1e-2, MI_const 1, samples_mi 200.

Example::

    py -3 lab/cbm/cibm_repro/train.py --cache runs/cibm_repro/cub_inception \
        --variant ibe
"""

from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score

from data import CUBEmbeddingsDataModule
from losses import est_HC, est_MI
from models import BasicMLP, StochasticMLP


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def configure_training(
    is_stochastic, arch, activation, optimizer, use_scheduler, lr, wd, device, epochs
):
    if is_stochastic:
        model = StochasticMLP(arch, activation, backbone=None).to(device)
    else:
        model = BasicMLP(arch, activation, backbone=None).to(device)
    if optimizer == "adam":
        optim = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    else:
        optim = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=wd, momentum=0.9)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
        if use_scheduler
        else None
    )
    return model, optim, scheduler


@torch.no_grad()
def run_test(model, loader, num_concepts, device, blackbox):
    preds, gt = [], []
    model.eval()
    preds_concepts = torch.zeros((num_concepts,))
    for batch in loader:
        feats = batch[0].to(device)
        gt_concepts = batch[1].to(device)
        targets = batch[2].reshape(-1).to(device)
        logits, _, concept_preds = model(feats)
        preds.append(logits.cpu().argmax(dim=-1).numpy())
        gt.append(targets.cpu().numpy())
        if not blackbox:
            preds_concepts += (
                (concept_preds > 0).int() == gt_concepts
            ).detach().cpu().sum(dim=0)
    test_acc = accuracy_score(np.concatenate(gt).reshape(-1), np.concatenate(preds).reshape(-1))
    return test_acc, preds_concepts / len(loader.dataset)


def train_model(
    dataloaders, model, optim, scheduler, *, MI_const, beta, beta_lr, epochs,
    sz, use_HC, blackbox, logdir, device,
):
    train_loader = dataloaders["train"]
    val_loader = dataloaders.get("val")
    best_label_acc = -1.0

    for epoch in range(epochs):
        model.train()
        preds, gt = [], []
        preds_concepts = torch.zeros((train_loader.dataset.num_concepts,))

        for batch in train_loader:
            feats = batch[0].to(device)
            gt_concepts = batch[1].to(device)
            targets = batch[2].reshape(-1).to(device)
            logits, max_prob, concept_preds = model(feats)

            # L = CE(y) + BCE(c) [+ beta*(MI_const - I(X;C))  or  (1+beta) scaling]
            loss = F.cross_entropy(logits, targets)
            preds.append(logits.detach().cpu().argmax(dim=-1).numpy())
            gt.append(targets.detach().cpu().numpy())

            if not blackbox:
                batch_concepts_loss = F.binary_cross_entropy_with_logits(
                    concept_preds.float(), gt_concepts.float(), reduction="none"
                ).sum(dim=1).mean()
                if use_HC:
                    loss += (1 + beta) * batch_concepts_loss
                else:
                    loss += batch_concepts_loss
                preds_concepts += (
                    (concept_preds > 0).int() == gt_concepts
                ).detach().cpu().sum(dim=0)

            if max_prob is not None:  # stochastic model
                if use_HC:
                    H_C = est_HC(model, train_loader.dataset, sz=min(sz, len(train_loader.dataset)))
                    mi_loss = (1 - beta) * H_C
                    constraint_item = 0.0
                else:
                    MI = est_MI(
                        model, train_loader.dataset,
                        sz=min(sz, len(train_loader.dataset)), jensen=False,
                    )
                    constraint = (MI_const - MI)
                    mi_loss = beta * constraint
                    constraint_item = constraint.item()
            else:
                mi_loss = None
                constraint_item = 0.0

            # Lagrangian dual update of beta (IBE only; IBB keeps beta fixed).
            if not use_HC:
                beta += beta_lr * constraint_item

            if use_HC and max_prob is not None:
                # H(C) gradient only wrt encoder heads, not the classifier.
                mi_loss.backward(
                    retain_graph=True,
                    inputs=list(model.pred_mu.parameters()) + list(model.pred_sigma.parameters()),
                )
            elif max_prob is not None:
                loss += mi_loss

            loss.backward()
            optim.step()
            optim.zero_grad()

        if scheduler is not None:
            scheduler.step()

        train_concept_acc = (preds_concepts / len(train_loader.dataset)).mean()
        train_label_acc = accuracy_score(
            np.concatenate(gt).reshape(-1), np.concatenate(preds).reshape(-1)
        )

        # Selection on val (or train if merged-away).
        sel_loader = val_loader if val_loader is not None else train_loader
        sel_acc, sel_concept = _eval_acc(model, sel_loader, device, blackbox)
        print(
            f"epoch {epoch:3d} | beta {beta:+.4f} | "
            f"train label {train_label_acc:.4f} concept {train_concept_acc:.4f} | "
            f"val label {sel_acc:.4f} concept {sel_concept:.4f}"
        )
        if sel_acc > best_label_acc:
            best_label_acc = sel_acc
            torch.save(model.state_dict(), os.path.join(logdir, "model.pth"))

    # Final test using the best-on-val checkpoint.
    model.load_state_dict(torch.load(os.path.join(logdir, "model.pth")))
    test_acc, test_concepts = run_test(
        model, dataloaders["test"], train_loader.dataset.num_concepts, device, blackbox
    )
    return test_acc, test_concepts.mean().item()


@torch.no_grad()
def _eval_acc(model, loader, device, blackbox):
    preds, gt = [], []
    model.eval()
    preds_concepts = torch.zeros((loader.dataset.num_concepts,))
    for batch in loader:
        feats = batch[0].to(device)
        gt_concepts = batch[1].to(device)
        targets = batch[2].reshape(-1).to(device)
        logits, _, concept_preds = model(feats)
        preds.append(logits.cpu().argmax(dim=-1).numpy())
        gt.append(targets.cpu().numpy())
        if not blackbox:
            preds_concepts += (
                (concept_preds > 0).int() == gt_concepts
            ).detach().cpu().sum(dim=0)
    acc = accuracy_score(np.concatenate(gt).reshape(-1), np.concatenate(preds).reshape(-1))
    return acc, (preds_concepts / len(loader.dataset)).mean().item()


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache", required=True, help="embedding cache dir")
    p.add_argument("--variant", choices=["baseline", "ibe", "ibb"], default="ibe")
    p.add_argument("--hidden", type=int, nargs="*", default=[],
                   help="hidden layer widths between embed_dim and num_concepts")
    p.add_argument("--activation", choices=["relu", "tanh"], default="relu")
    p.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--wd", type=float, default=0.0)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--beta-lr", type=float, default=-1e-2)
    p.add_argument("--mi-const", type=float, default=1.0)
    p.add_argument("--samples-mi", type=int, default=200)
    p.add_argument("--no-scheduler", action="store_true")
    p.add_argument("--merge-train-val", action="store_true")
    p.add_argument("--num-runs", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="runs/cibm_repro")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device} | variant: {args.variant}")

    is_stochastic = args.variant in ("ibe", "ibb")
    use_HC = args.variant == "ibb"

    test_accs, concept_accs = [], []
    for run in range(args.num_runs):
        set_seed(args.seed + run)
        dm = CUBEmbeddingsDataModule(args.cache, merge_train_val=args.merge_train_val)
        dataloaders = {
            "train": dm.train_dataloader(args.batch_size),
            "test": dm.test_dataloader(args.batch_size),
        }
        val = dm.val_dataloader(args.batch_size)
        if val is not None:
            dataloaders["val"] = val

        n_concepts = dm.train_dataset.num_concepts
        n_classes = dm.train_dataset.num_classes
        arch = [dm.embed_dim, *args.hidden, n_concepts, n_classes]
        print(f"run {run}: arch={arch}  ({n_concepts} concepts, {n_classes} classes)")

        model, optim, scheduler = configure_training(
            is_stochastic, arch, args.activation, args.optimizer,
            not args.no_scheduler, args.lr, args.wd, device, args.epochs,
        )
        logdir = os.path.join(args.out, f"{args.variant}_run{run}")
        os.makedirs(logdir, exist_ok=True)

        test_acc, concept_acc = train_model(
            dataloaders, model, optim, scheduler,
            MI_const=args.mi_const, beta=args.beta, beta_lr=args.beta_lr,
            epochs=args.epochs, sz=args.samples_mi, use_HC=use_HC,
            blackbox=False, logdir=logdir, device=device,
        )
        print(f"run {run}: TEST label {test_acc:.4f} | concept {concept_acc:.4f}")
        test_accs.append(test_acc)
        concept_accs.append(concept_acc)

    print(
        f"\n=== {args.variant} over {args.num_runs} run(s) ===\n"
        f"label acc:   {np.mean(test_accs):.4f} +/- {np.std(test_accs):.4f}\n"
        f"concept acc: {np.mean(concept_accs):.4f} +/- {np.std(concept_accs):.4f}"
    )


if __name__ == "__main__":
    main()
