"""
Train the BACON-concept-layer CBM on MNIST.

    python train.py --epochs 5

The 10 BACON trees are frozen human structure; only the CNN is trained, and
only on digit labels (no concept supervision).  The point of the experiment is
that the fixed symbolic logic provides *structural reinforcement* that pushes
the concept layer toward the human-defined, human-aligned concepts.

Optional --rules <file.json> overrides the default concept set / formulas, e.g.
    {"concepts": ["a", "b"], "rules": {"0": "a AND NOT b", ...}}
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Make the local modules and the bacon package importable regardless of cwd.
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, _REPO_ROOT)

import config as default_config          # noqa: E402
from model import ConceptBaconCBM, binarization_penalty  # noqa: E402


def load_rules(path: str | None):
    if path is None:
        return default_config.CONCEPTS, {
            int(k): v for k, v in default_config.DIGIT_RULES.items()
        }
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    concepts = data["concepts"]
    rules = {int(k): v for k, v in data["rules"].items()}
    return concepts, rules


def check_separability(model) -> None:
    """Warn if the human rules produce identical idealized concept targets."""
    M = model.logic.ideal_concept_matrix()
    seen = {}
    collisions = []
    for k in range(M.shape[0]):
        key = tuple(M[k].tolist())
        if key in seen:
            collisions.append((seen[key], k))
        else:
            seen[key] = k
    if collisions:
        print(f"[WARN] rule collisions (identical idealized targets): {collisions}")
    else:
        print("[OK] all 10 digit rules have distinct idealized concept targets")


def make_loaders(data_root: str, batch_size: int):
    tf_train = transforms.Compose([
        transforms.RandomAffine(degrees=8, translate=(0.06, 0.06)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    tf_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST(data_root, train=True, download=True, transform=tf_train)
    test_ds = datasets.MNIST(data_root, train=False, download=True, transform=tf_test)
    train_ld = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                          num_workers=2, pin_memory=True)
    test_ld = DataLoader(test_ds, batch_size=512, shuffle=False,
                         num_workers=2, pin_memory=True)
    return train_ld, test_ld


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    concept_sum = torch.zeros(len(model.concept_names), device=device)
    # per-class mean concept activations for interpretability
    per_class = torch.zeros(10, len(model.concept_names), device=device)
    per_class_n = torch.zeros(10, device=device)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, probs, _ = model(x)
        pred = logits.argmax(1)
        correct += (pred == y).sum().item()
        total += y.numel()
        concept_sum += probs.sum(0)
        for c in range(10):
            m = y == c
            if m.any():
                per_class[c] += probs[m].sum(0)
                per_class_n[c] += m.sum()
    acc = correct / total
    concept_mean = (concept_sum / total).cpu()
    per_class_mean = (per_class / per_class_n.clamp(min=1).unsqueeze(1)).cpu()
    return acc, concept_mean, per_class_mean


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--data", type=str, default=os.path.join(_REPO_ROOT, "benchmarks", "mnist-addition", "data"))
    ap.add_argument("--rules", type=str, default=None, help="optional JSON rule set")
    ap.add_argument("--bin-weight", type=float, default=0.05,
                    help="weight of the concept binarization penalty")
    ap.add_argument("--and-andness", type=float, default=default_config.AND_ANDNESS)
    ap.add_argument("--or-andness", type=float, default=default_config.OR_ANDNESS)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default=os.path.join(_HERE, "checkpoint.pt"),
                    help="where to save the trained model + rules")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    concepts, rules = load_rules(args.rules)
    model = ConceptBaconCBM(
        concepts, rules,
        and_andness=args.and_andness, or_andness=args.or_andness,
    ).to(device)

    print(f"device={device}  concepts={len(concepts)}  classes={len(rules)}")
    check_separability(model)

    train_ld, test_ld = make_loaders(args.data, args.batch_size)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = correct = total = 0
        for x, y in train_ld:
            x, y = x.to(device), y.to(device)
            logits, probs, _ = model(x)
            task = F.cross_entropy(logits, y)
            binp = binarization_penalty(probs)
            loss = task + args.bin_weight * binp
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item() * y.numel()
            correct += (logits.argmax(1) == y).sum().item()
            total += y.numel()
        sched.step()
        tr_acc = correct / total
        te_acc, concept_mean, per_class_mean = evaluate(model, test_ld, device)
        print(f"epoch {epoch:2d} | loss {running/total:.4f} | "
              f"train acc {tr_acc:.4f} | test acc {te_acc:.4f} | "
              f"temp {model.log_temp.exp().item():.2f}")

    # ---- interpretability report ---------------------------------------- #
    print("\nMean concept activation over test set:")
    for name, v in zip(model.concept_names, concept_mean.tolist()):
        print(f"  {name:18s} {v:.3f}")

    print("\nPer-digit mean concept activation (rows=digit, cols=concept):")
    header = "digit " + " ".join(f"{n[:6]:>6s}" for n in model.concept_names)
    print(header)
    for d in range(10):
        row = " ".join(f"{v:6.2f}" for v in per_class_mean[d].tolist())
        print(f"  {d}   {row}")

    # Alignment score: how well the learned per-digit concepts match the
    # idealized human targets (ignoring don't-cares).
    M = model.logic.ideal_concept_matrix()
    care = M != 0.5
    agree = ((per_class_mean > 0.5).float() == M).float()
    align = (agree[care].mean().item()) if care.any() else float("nan")
    print(f"\nHuman-alignment of learned concepts (on cared literals): {align:.3f}")

    # ---- save checkpoint (concepts + rules travel with the weights) ----- #
    torch.save({
        "state_dict": model.state_dict(),
        "concepts": concepts,
        "rules": rules,
        "and_andness": args.and_andness,
        "or_andness": args.or_andness,
    }, args.out)
    print(f"\nsaved checkpoint -> {args.out}")


if __name__ == "__main__":
    main()
