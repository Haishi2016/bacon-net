"""
Emergent-concept MNIST training: a shared CNN concept encoder + 10 SEPARATE
trainable BACON logic trees (one per digit), jointly trained on digit labels
ONLY.  The concepts are UNNAMED / unsupervised -- they must *emerge* so that ten
independent graded-logic trees can separate the digits.

Each digit tree is a genuine, independent ``bacon.binaryTreeLogicNet`` (from the
bacon-net framework):
  * its own ``FullWeightAggregator`` (per-node andness = conjunction<->disjunction),
  * its own per-node weights,
  * its own soft input permutation (Sinkhorn) -- routes the K concepts its way,
  * its own identity/negation transformation per concept (so a tree can require
    a concept to be ABSENT).
So the ten trees are fully independent structures, not one shared head.

    # single run with K concepts
    python train_emergent_concepts.py --concepts 7 --epochs 12

    # scan K = 2..10 to find the fewest concepts that still classify well
    python train_emergent_concepts.py --scan --scan-min 2 --scan-max 10 --epochs 10

The concept encoder + all ten trees train together with one optimizer; the soft
input permutations are annealed from broad routing to near-discrete over the run.
A small binarization penalty pushes concepts toward crisp 0/1 truth degrees.
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, _REPO_ROOT)

from bacon.binaryTreeLogicNet import binaryTreeLogicNet          # noqa: E402
from bacon.aggregators.lsp import FullWeightAggregator           # noqa: E402
from bacon.transformationLayer import (IdentityTransformation,   # noqa: E402
                                       NegationTransformation)
from model import ConceptCNN, binarization_penalty               # noqa: E402
from train import make_loaders                                   # noqa: E402


class MultiTreeBaconCBM(nn.Module):
    """CNN concept encoder -> K sigmoid concepts -> 10 independent BACON trees."""

    def __init__(self, n_concepts: int, n_classes: int = 10,
                 tree_layout: str = "left", logit_temperature: float = 4.0,
                 sinkhorn_iters: int = 20, weight_mode: str = "trainable",
                 no_negation: bool = False, device=None):
        super().__init__()
        self.n_concepts = n_concepts
        self.n_classes = n_classes
        self.weight_mode = weight_mode
        self.no_negation = no_negation
        self.tree_layout = tree_layout
        self.device = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = ConceptCNN(n_concepts)
        # one genuinely independent tree per class (own aggregator instance)
        self.trees = nn.ModuleList([
            binaryTreeLogicNet(
                input_size=n_concepts,
                aggregator=FullWeightAggregator(),      # fresh per tree
                tree_layout=tree_layout,
                normalize_andness=True,
                weight_mode=weight_mode,                # 'trainable' or 'fixed'
                                                        # (fixed -> node weights
                                                        # frozen at 0.5 = equal;
                                                        # only andness/routing/
                                                        # transforms learn)
                weight_value=0.5,
                weight_normalization="softmax",         # soft weights -> genuine
                                                        # 2-input graded aggregation
                                                        # (minmax collapses to hard
                                                        # single-child selection)
                use_permutation_layer=(tree_layout not in ("full",)),
                use_transformation_layer=True,          # identity / negation only
                transformations=([IdentityTransformation(n_concepts)]
                                 if no_negation else
                                 [IdentityTransformation(n_concepts),
                                  NegationTransformation(n_concepts)]),
                sinkhorn_iters=sinkhorn_iters,
                device=self.device,
            )
            for _ in range(n_classes)
        ])
        for t in self.trees:
            t.cache_layer_outputs = False               # speed: skip viz cache
        self.log_temp = nn.Parameter(
            torch.tensor(float(torch.log(torch.tensor(logit_temperature)))))
        self._frozen = False

    def concept_probs(self, x):
        return torch.sigmoid(self.encoder(x))

    def forward(self, x):
        probs = self.concept_probs(x)                            # (B, K)
        truths = torch.cat([t(probs) for t in self.trees], dim=1)  # (B, 10)
        truths = truths.clamp(1e-6, 1.0 - 1e-6)
        logits = self.log_temp.exp() * (torch.log(truths) - torch.log1p(-truths))
        return logits, probs, truths

    def anneal(self, progress: float):
        for t in self.trees:
            t.anneal_routing(progress)

    # -- hard-permutation freezing (matches baconNet locked_perm) -----------
    @torch.no_grad()
    def permutation_confidence(self) -> float:
        """Mean peak routing weight per leaf, averaged over the 10 trees."""
        if getattr(self, "_frozen", False):
            return 1.0
        vals = []
        for t in self.trees:
            itl = t.input_to_leaf
            if not hasattr(itl, "sinkhorn"):
                continue
            P = itl.sinkhorn(itl.logits, n_iters=itl.sinkhorn_iters,
                             temperature=itl.temperature)
            vals.append(P.max(dim=1).values.mean())
        return torch.stack(vals).mean().item() if vals else 1.0

    def permutation_sparsity_loss(self) -> torch.Tensor:
        """Mean routing row-entropy over the 10 trees; minimizing it peaks the
        Sinkhorn permutations toward near-hard (bijective) assignments."""
        if getattr(self, "_frozen", False):
            return self.log_temp.new_zeros(())
        losses = []
        for t in self.trees:
            itl = t.input_to_leaf
            if not hasattr(itl, "sinkhorn"):
                continue
            P = itl.sinkhorn(itl.logits, n_iters=itl.sinkhorn_iters,
                             temperature=itl.temperature)
            losses.append(-(P.clamp_min(1e-9) * P.clamp_min(1e-9).log()).sum(1).mean())
        return torch.stack(losses).mean() if losses else self.log_temp.new_zeros(())

    @torch.no_grad()
    def freeze(self) -> None:
        """Hungarian-harden every tree's soft permutation into a locked hard
        permutation (frozenInputToLeaf) and stop training the routing logits."""
        if getattr(self, "_frozen", False):
            return
        from scipy.optimize import linear_sum_assignment
        from bacon.frozonInputToLeaf import frozenInputToLeaf
        for t in self.trees:
            itl = t.input_to_leaf
            if not hasattr(itl, "sinkhorn"):
                continue
            P = itl.sinkhorn(itl.logits, n_iters=itl.sinkhorn_iters,
                             temperature=itl.temperature)
            rows, cols = linear_sum_assignment(-P.detach().cpu().numpy())
            assign = [0] * len(rows)
            for i in range(len(rows)):
                assign[int(rows[i])] = int(cols[i])
            t.input_to_leaf = frozenInputToLeaf(assign, itl.logits.shape[1]).to(self.device)
            t.is_frozen = True
            t.locked_perm = torch.tensor(assign)
        self._frozen = True

    def prepare_frozen_structure(self) -> None:
        """Swap in frozenInputToLeaf placeholders so a frozen checkpoint (with
        P_hard buffers) can be loaded via load_state_dict."""
        from bacon.frozonInputToLeaf import frozenInputToLeaf
        for t in self.trees:
            itl = t.input_to_leaf
            if not hasattr(itl, "logits"):
                continue
            num_leaves, num_inputs = itl.logits.shape
            t.input_to_leaf = frozenInputToLeaf(list(range(num_leaves)),
                                                num_inputs).to(self.device)
            t.is_frozen = True
        self._frozen = True


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    per_class = torch.zeros(10, model.n_concepts, device=device)
    per_class_n = torch.zeros(10, device=device)
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, probs, _ = model(x)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
        for c in range(10):
            m = y == c
            if m.any():
                per_class[c] += probs[m].sum(0)
                per_class_n[c] += m.sum()
    acc = correct / total
    per_class_mean = (per_class / per_class_n.clamp(min=1).unsqueeze(1)).cpu()
    return acc, per_class_mean


def train_one(n_concepts, train_ld, test_ld, device, epochs=12, lr=1e-3,
              bin_weight=0.0, tree_layout="left", seed=0, weight_mode="trainable",
              verbose=True, harden=False, perm_sparsity=5.0, freeze_conf=0.90,
              freeze_frac=0.85, no_negation=False):
    import copy
    torch.manual_seed(seed)
    model = MultiTreeBaconCBM(n_concepts, tree_layout=tree_layout,
                              weight_mode=weight_mode, no_negation=no_negation,
                              device=device).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    best_acc = 0.0
    best_pc = None
    best_state = None
    frozen = False
    can_freeze = harden and tree_layout == "left"
    force_at = int(freeze_frac * epochs)
    for epoch in range(epochs):
        model.train()
        # anneal routing broad -> discrete over the first ~70% of training, then
        # hold at the sharpened routing so the last epochs fine-tune stably
        # (annealing to hard routing at the very end snaps the permutation).
        prog = min(1.0, epoch / max(1.0, 0.7 * epochs - 1))
        if not frozen:
            model.anneal(prog)
        run = correct = total = 0
        for x, y in train_ld:
            x, y = x.to(device), y.to(device)
            logits, probs, _ = model(x)
            loss = F.cross_entropy(logits, y) + bin_weight * binarization_penalty(probs)
            if can_freeze and not frozen:
                # entropy penalty (ramped with anneal so concepts form first)
                # peaks the routing so it can be hard-frozen without collapse
                loss = loss + perm_sparsity * prog * model.permutation_sparsity_loss()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run += loss.item() * y.numel()
            correct += (logits.argmax(1) == y).sum().item()
            total += y.numel()
        sched.step()
        # confidence-triggered hard freeze, then frozen finetuning
        if can_freeze and not frozen:
            conf = model.permutation_confidence()
            if conf >= freeze_conf or epoch >= force_at:
                model.freeze()
                frozen = True
                if verbose:
                    print(f"  [freeze] ep {epoch + 1} conf {conf:.3f} -> trees hardened")
        te_acc, pc = evaluate(model, test_ld, device)
        # only keep hard checkpoints when hardening (soft ones are not faithful)
        if (not can_freeze or frozen) and te_acc >= best_acc:
            best_acc, best_pc = te_acc, pc
            best_state = copy.deepcopy(model.state_dict())
        if verbose:
            cs = (f" | conf {model.permutation_confidence():.3f}"
                  if can_freeze and not frozen else (" | FROZEN" if frozen else ""))
            print(f"  epoch {epoch + 1:2d}/{epochs} | loss {run / total:.4f} | "
                  f"train {correct / total:.4f} | test {te_acc:.4f} | "
                  f"temp {model.log_temp.exp().item():.2f}{cs}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return best_acc, model, best_pc


def _print_concept_table(per_class_mean):
    K = per_class_mean.shape[1]
    print("\nPer-digit mean concept activation (rows=digit, cols=concept c0..):")
    print("      " + " ".join(f"c{j:<4d}" for j in range(K)))
    for d in range(10):
        print(f"  {d}:  " + " ".join(f"{v:5.2f}" for v in per_class_mean[d].tolist()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concepts", type=int, default=7,
                    help="number of emergent concepts K (single-run mode)")
    ap.add_argument("--scan", action="store_true",
                    help="scan K from --scan-min to --scan-max instead")
    ap.add_argument("--scan-min", type=int, default=2)
    ap.add_argument("--scan-max", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=1,
                    help="number of seeds per K in scan mode (reports mean+/-std)")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bin-weight", type=float, default=0.0,
                    help="concept binarization penalty; >0 can collapse the "
                         "cold-start (all concepts -> 1), keep 0 unless warmed up")
    ap.add_argument("--layout", type=str, default="left",
                    choices=["left", "balanced", "paired", "full", "alternating"])
    ap.add_argument("--weight-mode", type=str, default="trainable",
                    choices=["trainable", "fixed"],
                    help="'fixed' freezes node weights at 0.5 (equal) so the "
                         "tree can't reweight concepts -- forces purer concepts")
    ap.add_argument("--save", type=str, default=None,
                    help="path to save the trained model (single-run mode)")
    ap.add_argument("--no-negation", action="store_true",
                    help="drop the Negation transform: trees may use POSITIVE "
                         "concepts only (no elimination-by-absence rules)")
    ap.add_argument("--harden", action="store_true",
                    help="drive routing hard (sparsity penalty) then Hungarian-freeze "
                         "each tree's permutation + frozen finetune -> faithful hard "
                         "per-concept logic trees (prune-analyzable)")
    ap.add_argument("--perm-sparsity", type=float, default=5.0,
                    help="weight of the routing row-entropy penalty (--harden)")
    ap.add_argument("--freeze-conf", type=float, default=0.90,
                    help="mean peak-routing confidence that triggers the hard freeze")
    ap.add_argument("--freeze-frac", type=float, default=0.85,
                    help="force the freeze by this fraction of epochs if conf never reached")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", type=str,
                    default=os.path.join(_REPO_ROOT, "benchmarks", "mnist-addition", "data"))
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ld, test_ld = make_loaders(args.data, args.batch_size)
    print(f"device={device}  layout={args.layout}  epochs={args.epochs}")

    if not args.scan:
        print(f"\n=== Training with K={args.concepts} emergent concepts "
              f"(weights: {args.weight_mode}) ===")
        acc, model, pc = train_one(args.concepts, train_ld, test_ld, device,
                                   epochs=args.epochs, lr=args.lr,
                                   bin_weight=args.bin_weight, tree_layout=args.layout,
                                   seed=args.seed, weight_mode=args.weight_mode,
                                   harden=args.harden, perm_sparsity=args.perm_sparsity,
                                   freeze_conf=args.freeze_conf, freeze_frac=args.freeze_frac,
                                   no_negation=args.no_negation)
        print(f"\nBest test accuracy (K={args.concepts}): {acc * 100:.2f}%")
        _print_concept_table(pc)
        if args.save:
            torch.save({"state_dict": model.state_dict(), "K": args.concepts,
                        "layout": args.layout, "weight_mode": args.weight_mode,
                        "seed": args.seed, "acc": acc,
                        "no_negation": bool(args.no_negation),
                        "frozen": bool(getattr(model, "_frozen", False))}, args.save)
            print(f"saved model -> {args.save}")
        return

    print(f"\n=== Scanning K = {args.scan_min}..{args.scan_max}  "
          f"({args.seeds} seed{'s' if args.seeds > 1 else ''} each) ===")
    import statistics
    results = {}
    for K in range(args.scan_min, args.scan_max + 1):
        accs = []
        for s in range(args.seeds):
            acc, _, _ = train_one(K, train_ld, test_ld, device, epochs=args.epochs,
                                  lr=args.lr, bin_weight=args.bin_weight,
                                  tree_layout=args.layout, seed=args.seed + s,
                                  weight_mode=args.weight_mode, verbose=False)
            accs.append(acc)
            print(f"  K={K} seed {args.seed + s}: {acc * 100:.2f}%")
        results[K] = accs
        m = statistics.mean(accs)
        sd = statistics.pstdev(accs) if len(accs) > 1 else 0.0
        print(f"  => K={K}: {m * 100:.2f} +/- {sd * 100:.2f}%  "
              f"(best {max(accs) * 100:.2f})")

    print("\n" + "=" * 46)
    print(f"{'K (concepts)':>12s} {'mean':>8s} {'std':>7s} {'best':>7s}")
    means = {}
    for K, accs in results.items():
        m = statistics.mean(accs)
        sd = statistics.pstdev(accs) if len(accs) > 1 else 0.0
        means[K] = m
        print(f"{K:>12d} {m * 100:>7.2f} {sd * 100:>6.2f} {max(accs) * 100:>6.2f}")
    best_k = max(means, key=means.get)
    thresh = means[best_k] - 0.005
    sweet = min(k for k, m in means.items() if m >= thresh)
    print(f"\nbest: K={best_k} ({means[best_k] * 100:.2f}%);  "
          f"min-concept sweet spot (within 0.5% mean): K={sweet} "
          f"({means[sweet] * 100:.2f}%)")


if __name__ == "__main__":
    main()
