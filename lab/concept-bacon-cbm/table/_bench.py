"""
Shared harness for the paper-table benchmark scripts.

Each script trains ONE model on ONE dataset for ``--iters`` random seeds and
reports the two table quantities as mean +/- std plus a LaTeX-ready cell:

  * ACC_Y : task (label) accuracy.
  * ACC_C : concept accuracy -- only defined for concept-based models on
            datasets that carry ground-truth concept labels.  Left blank ("--")
            otherwise (e.g. Blackbox, or any model on plain MNIST which has no
            concept annotations).

Naming convention for the scripts:  ``<model>-<dataset>-accuracy.py``
(e.g. ``blackbox-mnist-accuracy.py``), one script per table cell/row-segment.
"""

from __future__ import annotations

import os
import random
import statistics
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_CBM = os.path.abspath(os.path.join(_HERE, ".."))            # concept-bacon-cbm/
_CREAM = os.path.join(_CBM, "cream")                         # cream reproduction
_REPO = os.path.abspath(os.path.join(_CBM, "..", ".."))      # repo root (bacon pkg)
for _p in (_CBM, _CREAM, _REPO):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# --------------------------------------------------------------------------- #
# reproducibility / device
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------- #
# data
# --------------------------------------------------------------------------- #
def mnist_loaders(batch_size: int = 256):
    """MNIST train/test loaders (reuses the project's augmented pipeline)."""
    from train import make_loaders
    data = os.path.join(_REPO, "benchmarks", "mnist-addition", "data")
    return make_loaders(data, batch_size)


def mnist_labels():
    """Return (y_train, y_test) integer label tensors for MNIST (labels only)."""
    from torchvision import datasets
    root = os.path.join(_REPO, "benchmarks", "mnist-addition", "data")
    tr = datasets.MNIST(root, train=True, download=True)
    te = datasets.MNIST(root, train=False, download=True)
    return tr.targets.long(), te.targets.long()


def mnist_concept_spec(device: str = "cpu"):
    """Single source of truth for the MNIST concept set, shared by ALL rows.

    Every concept-based MNIST script must build its concepts / targets / reasoning
    graph from this spec, so no two rows can diverge on the concept definition
    (the fairness guarantee).  Fields:

      concept_names : list[str]        the 9 stroke concepts (config.CONCEPTS)
      formulas      : dict[int,str]    per-digit BACON rules (config.DIGIT_RULES)
      K             : int              number of concepts
      M             : (10,K)           idealized signatures in {0, 0.5, 1}
      ctgt          : (10,K)           0/1 concept supervision targets
      cmask         : (10,K)           1 where the rule constrains the concept
      A_Y           : (10,K)           reasoning graph (concepts each digit uses)
      target_fn     : y -> (ctgt[y], cmask[y])   for train_with_concepts()
    """
    import types

    import config as cfg
    from bacon_logic import BaconLogicBank, collect_concepts, parse_formula

    rules = {int(k): v for k, v in cfg.DIGIT_RULES.items()}
    bank = BaconLogicBank(cfg.CONCEPTS, rules)
    M = bank.ideal_concept_matrix().to(device)      # (10, K) in {0, 0.5, 1}
    ctgt = M.clamp(0.0, 1.0)                         # 0/1 on constrained entries
    cmask = (M != 0.5).float()                       # supervise only rule literals
    K = M.shape[1]
    idx = {n: i for i, n in enumerate(cfg.CONCEPTS)}
    A_Y = torch.zeros(len(rules), K, device=device)
    for d, f in rules.items():
        for name in collect_concepts(parse_formula(f), set()):
            A_Y[d, idx[name]] = 1.0

    def target_fn(y):
        return ctgt[y], cmask[y]

    return types.SimpleNamespace(
        concept_names=list(cfg.CONCEPTS), formulas=rules, K=K,
        M=M, ctgt=ctgt, cmask=cmask, A_Y=A_Y, target_fn=target_fn,
        # BACON/OCBM compatibility: MNIST stroke concepts are independent binary
        index=idx, mutex_groups=[], binary_concepts=list(range(K)))


# --------------------------------------------------------------------------- #
# generic task-only training / evaluation
# --------------------------------------------------------------------------- #
def train_task_only(model, loader, device, epochs: int = 6, lr: float = 1e-3):
    """Supervised training on labels only. Model.forward -> (logits, *rest)."""
    import torch.nn.functional as F
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)[0]
            loss = F.cross_entropy(logits, y)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
    return model


@torch.no_grad()
def task_accuracy(model, loader, device) -> float:
    """Top-1 label accuracy. Model.forward -> (logits, *rest)."""
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)[0]
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total


def train_with_concepts(model, loader, target_fn, device,
                        epochs: int = 6, lr: float = 1e-3, lam: float = 1.0):
    """Joint CBM training: CE(task) + lam * masked BCE(concepts).

    ``target_fn(y) -> (ctrue, cmask)`` supplies per-batch ground-truth concept
    targets and a 0/1 mask (mask=0 entries are ignored, e.g. don't-care literals
    in a rule signature).  Model.forward -> (logits, cprobs) with cprobs in (0,1).
    """
    import torch.nn.functional as F
    model.to(device).train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    for _ in range(epochs):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, cprobs = model(x)
            loss = F.cross_entropy(logits, y)
            if cprobs is not None:
                ctrue, cmask = target_fn(y)
                bce = F.binary_cross_entropy(cprobs, ctrue, reduction="none")
                loss = loss + lam * (bce * cmask).sum() / cmask.sum().clamp(min=1)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
    return model


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def _stats(vals):
    if not vals:
        return None
    mean = statistics.mean(vals)
    std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
    return mean, std


def _cell(stat):
    if stat is None:
        return "--"
    mean, std = stat
    return f"${mean * 100:.2f}_{{\\pm{std * 100:.2f}}}$"


def report(model_name: str, dataset: str, acc_y, acc_c=None):
    """Print a human-readable summary + a LaTeX-ready cell for one table entry."""
    n = len(acc_y)
    sy = _stats(acc_y)
    sc = _stats(acc_c) if acc_c else None
    print(f"\n=== {model_name}  /  {dataset}   (n={n}) ===")
    print(f"  ACC_Y = {sy[0] * 100:6.2f} +/- {sy[1] * 100:.2f}   "
          f"runs = {[round(v * 100, 2) for v in acc_y]}")
    if sc is not None:
        print(f"  ACC_C = {sc[0] * 100:6.2f} +/- {sc[1] * 100:.2f}   "
              f"runs = {[round(v * 100, 2) for v in acc_c]}")
    else:
        print("  ACC_C =    --   (no concept labels for this model/dataset)")
    print(f"  LaTeX cell ->  ACC_Y {_cell(sy)}   ACC_C {_cell(sc)}")
    return {"acc_y": sy, "acc_c": sc}
