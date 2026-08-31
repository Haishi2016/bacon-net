"""
Emergent concepts on CUB-200 (the pivotal scale test).

ResNet-18 -> K UNNAMED sigmoid concepts -> 200 vectorized graded-logic BACON
trees (one per species, VectorTreeLogicHead) -> species logits.  Trained on
SPECIES LABELS ONLY (no attribute supervision).  Then AUTO-NAME each emerged
concept against the 112 human CUB attributes via per-attribute ROC-AUC -- fully
automatic, no hand-designed probes (the generalizable regime).

Promising if: (a) few concepts still classify species reasonably, and (b) the
emerged concepts align with human bird attributes (some attribute AUC >> 0.5).

    python cub_emergent.py --concepts 32 --epochs 20
"""

from __future__ import annotations

import argparse
import copy
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "table"))
sys.path.insert(0, _REPO_ROOT)

import _cub                                                     # noqa: E402
from bacon.vectorizedLogicHead import VectorTreeLogicHead, VectorLogicHead  # noqa: E402
from bacon.vectorizedFullTree import VectorFullTreeHead  # noqa: E402
from bacon.vectorizedHybridTree import VectorHybridTreeHead  # noqa: E402
from bacon.vectorizedRectTree import VectorRectTreeHead  # noqa: E402
from bacon.aggregators.lsp import FullWeightAggregator          # noqa: E402
from eval_shapes import roc_auc                                 # noqa: E402


class CUBEmergent(nn.Module):
    def __init__(self, K, n_species=200, temp=6.0, head="tree", sinkhorn_iters=20,
                 branching=4, fulltree_final_temp=0.05, fulltree_straight_through=False,
                 fulltree_coefficients=False, fulltree_max_egress=1,
                 fulltree_bin_frac=0.25, fulltree_negation=False, concept_scale=1.0,
                 fulltree_aggregator="full_weight", fulltree_experts=5,
                 hybrid_permutation=True,
                 rect_width=None, rect_depth=4, rect_max_parents=1,
                 rect_root_lam=1.0, rect_parent_lam=0.1, rect_compact_lam=0.05,
                 rect_binarize_lam=0.0, rect_straight_through=False,
                 rect_leaf_shortcut=False):
        super().__init__()
        self.K = K
        self.head_type = head
        self.head_n_species = n_species
        self.branching = branching
        self.concept_scale = float(concept_scale)
        self.backbone = _cub._make_resnet()
        self.concept = nn.Linear(512, K)
        if head == "alt":
            # alternating coeff/aggregation head: learns per-head coefficients
            # (weights) AND anchor-operator mixtures, alternately (static anchors).
            self.head = VectorLogicHead(K, n_species, layout="alternating")
        elif head == "fulltree":
            # permutation-free, egress-hardened full tree (branching funnel);
            # N-ary graded-logic nodes with continuous andness.
            self.head = VectorFullTreeHead(K, n_species, branching=branching,
                                           final_temperature=fulltree_final_temp,
                                           straight_through=fulltree_straight_through,
                                           use_coefficients=fulltree_coefficients,
                                           max_egress=fulltree_max_egress,
                                           use_negation=fulltree_negation,
                                           aggregator=fulltree_aggregator,
                                           num_experts=fulltree_experts)
        elif head == "hybrid":
            # left-associative binary spine over the important features + a
            # shallow full sub-tree pooling the rest (fed at the deepest node).
            # A per-head Sinkhorn permutation learns which concepts land on the
            # spine; an identity/negation gate lets any leaf read NOT c.
            self.head = VectorHybridTreeHead(K, n_species, bin_frac=fulltree_bin_frac,
                                             branching=branching,
                                             max_egress=fulltree_max_egress,
                                             use_coefficients=fulltree_coefficients,
                                             use_negation=fulltree_negation,
                                             use_permutation_layer=hybrid_permutation,
                                             sinkhorn_iters=sinkhorn_iters,
                                             final_temperature=fulltree_final_temp)
        elif head == "recttree":
            # rectangular learned-convergence DAG: constant-width layers, tree
            # shape EMERGES from three penalties (single-root pointer, <=N-parent
            # fan-out, non-compactness) rather than a fixed funnel. Penalties are
            # added to the loss in train_one via head.regularization().
            self.head = VectorRectTreeHead(K, n_species, width=rect_width,
                                           depth=rect_depth,
                                           max_parents=rect_max_parents,
                                           root_lam=rect_root_lam,
                                           parent_lam=rect_parent_lam,
                                           compact_lam=rect_compact_lam,
                                           binarize_lam=rect_binarize_lam,
                                           straight_through=rect_straight_through,
                                           leaf_shortcut=rect_leaf_shortcut,
                                           use_negation=fulltree_negation,
                                           use_coefficients=fulltree_coefficients,
                                           final_temperature=fulltree_final_temp)
        else:
            self.head = VectorTreeLogicHead(
                K, n_species, FullWeightAggregator(),
                use_permutation_layer=True, use_transformation_layer=True,
                sinkhorn_iters=sinkhorn_iters)
        self.log_temp = nn.Parameter(torch.tensor(float(math.log(temp))))

    def concept_probs(self, x):
        # `concept_scale` > 1 steepens the sigmoid: it gives the uncertain
        # mid-range more output resolution (crisper concept truths for the tree).
        return torch.sigmoid(self.concept_scale * self.concept(self.backbone(x)))

    def forward(self, x):
        c = self.concept_probs(x)
        t = self.head(c).clamp(1e-6, 1.0 - 1e-6)
        logits = self.log_temp.exp() * (torch.log(t) - torch.log1p(-t))
        return logits, c, t

    def anneal(self, p):
        if hasattr(self.head, "anneal"):
            self.head.anneal(p)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for img, c, y in loader:
        img, y = img.to(device), y.to(device)
        correct += (model(img)[0].argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total


@torch.no_grad()
def collect(model, loader, device):
    model.eval()
    cs, attrs = [], []
    for img, c, y in loader:
        cs.append(model.concept_probs(img.to(device)).cpu())
        attrs.append(c)
    return torch.cat(cs), torch.cat(attrs)              # (N,K), (N,112)


def mean_abs_corr(C):
    Cc = C - C.mean(0)
    Z = Cc / (Cc.std(0) + 1e-8)
    corr = (Z.T @ Z) / C.shape[0]
    K = C.shape[1]
    return corr[~torch.eye(K, dtype=torch.bool)].abs().mean().item()


def top_attr_per_concept(C, A):
    tops = []
    for i in range(C.shape[1]):
        best = (0.5, -1)
        for a in range(A.shape[1]):
            col = A[:, a]
            if 0 < col.sum() < len(col):
                au = roc_auc(C[:, i], col.long())
                if au > best[0]:
                    best = (au, a)
        tops.append(best)                               # (auc, attr_idx)
    return tops


def balanced_ovr_loss(truths, y, n_classes):
    """Balanced one-vs-rest loss so every per-species tree gets a real training
    signal each batch, not just the ~1/200 positives softmax-CE gives it.

    Each tree is treated as an independent detector "is this species c?":
      - its OWN class image is a positive (full weight 1),
      - every OTHER class image is a negative, cross-fed but DOWN-WEIGHTED to
        1/(C-1) each so the ~199 negatives sum to weight ~1 and can't drown the
        rare positive (prevents the trivial "always say negative" cheat).
    truths: (B, C) graded-logic tree outputs in (0,1); y: (B,) class indices.
    """
    onehot = F.one_hot(y, n_classes).float()
    bce = F.binary_cross_entropy(truths, onehot, reduction="none")   # (B, C)
    pos = (bce * onehot).sum(1)                                      # weight 1
    neg = (bce * (1.0 - onehot)).sum(1) / (n_classes - 1)            # mean negative
    return (pos + neg).mean()


def train_one(K, tl, vl, device, epochs, seed, head="tree",
              anneal_frac=0.7, anneal_cap=0.75, loss_mode="ce", ovr_weight=0.3,
              harden=False, sinkhorn_iters=20, perm_sparsity=5.0,
              freeze_conf=0.90, freeze_frac=0.85,
              concept_lam=0.0, concept_targets=None,
              decorr_lam=0.0, branching=4, fulltree_final_temp=0.05,
              fulltree_straight_through=False, fulltree_coefficients=False,
              fulltree_scan=0, fulltree_max_egress=1, fulltree_bin_frac=0.25,
              fulltree_negation=False, concept_scale=1.0,
              fulltree_aggregator="full_weight", fulltree_experts=5,
              hybrid_permutation=True,
              rect_width=None, rect_depth=4, rect_max_parents=1,
              rect_root_lam=1.0, rect_parent_lam=0.1, rect_compact_lam=0.05,
              rect_binarize_lam=0.0, rect_straight_through=False,
              rect_leaf_shortcut=False, early_stop_patience=0, min_freeze_frac=0.3,
              ckpt_path=None, ckpt_every=0):
    torch.manual_seed(seed)
    model = CUBEmergent(K, head=head, sinkhorn_iters=sinkhorn_iters,
                        branching=branching,
                        fulltree_final_temp=fulltree_final_temp,
                        fulltree_straight_through=fulltree_straight_through,
                        fulltree_coefficients=fulltree_coefficients,
                        fulltree_max_egress=fulltree_max_egress,
                        fulltree_bin_frac=fulltree_bin_frac,
                        fulltree_negation=fulltree_negation,
                        concept_scale=concept_scale,
                        fulltree_aggregator=fulltree_aggregator,
                        fulltree_experts=fulltree_experts,
                        hybrid_permutation=hybrid_permutation,
                        rect_width=rect_width, rect_depth=rect_depth,
                        rect_max_parents=rect_max_parents, rect_root_lam=rect_root_lam,
                        rect_parent_lam=rect_parent_lam, rect_compact_lam=rect_compact_lam,
                        rect_binarize_lam=rect_binarize_lam,
                        rect_straight_through=rect_straight_through,
                        rect_leaf_shortcut=rect_leaf_shortcut).to(device)
    if concept_targets is not None:
        concept_targets = torch.as_tensor(concept_targets, device=device)
    bb_ids = {id(p) for p in model.backbone.parameters()}
    bb = [p for p in model.parameters() if id(p) in bb_ids]
    heads = [p for p in model.parameters() if id(p) not in bb_ids]
    opt = torch.optim.Adam([{"params": bb, "lr": 1e-4}, {"params": heads, "lr": 1e-3}])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(epochs, 1))
    best = 0.0
    best_state = None
    best_soft = 0.0            # peak SOFT test acc (pre-freeze) + its snapshot
    best_soft_state = None
    no_improve = 0            # epochs since the soft peak (for early stopping)
    frozen = False
    is_ft = head in ("fulltree", "hybrid")
    is_rect = head == "recttree"
    can_freeze = harden and head in ("tree", "fulltree", "hybrid", "recttree")
    # egress hardening collapses if frozen while routing is still soft -> require
    # near-one-hot routing (conf>=0.999) for the full tree (MNIST lesson).
    eff_freeze_conf = 0.999 if is_ft else freeze_conf

    def _sparsity_loss():
        if is_rect:
            return torch.zeros((), device=device)   # structure penalty added separately
        return (model.head.egress_sparsity_loss() if is_ft
                else model.head.permutation_sparsity_loss())

    def _confidence():
        if is_rect:
            return 0.0                              # no confidence metric -> freeze at force_at
        return float(model.head.egress_confidence() if is_ft
                     else model.head.permutation_confidence())

    def _freeze():
        if is_rect:
            model.head.harden()
            return
        if is_ft and fulltree_scan > 0:
            # candidate-scan freeze: treat the soft egress as a distribution over
            # discrete trees; sample `fulltree_scan` hard routings and keep, per
            # head, the one whose hard output best reproduces the soft output.
            model.eval()
            xs, n = [], 0
            with torch.no_grad():
                for img, _c, _y in vl:
                    xs.append(model.concept_probs(img.to(device))); n += img.size(0)
                    if n >= 2048:
                        break
            calib = torch.cat(xs)[:2048]
            mse = model.head.freeze_egress_scan(calib, num_candidates=fulltree_scan)
            print(f"    [scan-freeze] {fulltree_scan} candidates -> "
                  f"faithfulness MSE {float(mse):.4e}", flush=True)
        elif is_ft:
            model.head.freeze_egress()
        else:
            model.head.freeze_permutation()

    force_at = int(freeze_frac * epochs)
    start_ep = 0
    # ---- resume from a full checkpoint (power-outage safety) --------------
    # ckpt_path holds the CURRENT model+optimizer+scheduler+epoch (not just the
    # best snapshot), so an interrupted run continues where it left off.
    if ckpt_path and os.path.exists(ckpt_path):
        try:
            ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        except Exception as e:                                    # corrupt/partial
            ck = None
            print(f"    [resume] could not read {ckpt_path} ({e}); starting fresh",
                  flush=True)
        if ck is not None and "opt" in ck:                       # resumable (full) ckpt
            model.load_state_dict(ck["model"])
            opt.load_state_dict(ck["opt"])
            sched.load_state_dict(ck["sched"])
            start_ep = int(ck.get("epoch", 0))
            best = float(ck.get("best", 0.0))
            best_state = ck.get("best_state", None)
            frozen = bool(ck.get("frozen", False))
            print(f"    [resume] {ckpt_path} -> continue at ep {start_ep + 1}/{epochs} "
                  f"(best {best * 100:.2f}%, frozen={frozen})", flush=True)
    for ep in range(start_ep, epochs):
        model.train()
        # Ramp routing sharpness over the first `anneal_frac` of the run to a
        # CAP (< 1.0), then HOLD.  Pushing routing fully hard at the very end
        # collapsed CUB training (loss explodes ~ep105/120 at cap 0.85); the
        # accuracy peak lives around anneal ~0.65-0.75, so we hold there and
        # give the model the tail epochs to settle instead of snapping harder.
        raw = min(1.0, ep / max(1.0, anneal_frac * (epochs - 1)))
        model.anneal(anneal_cap * raw)
        run = total = 0
        for img, c, y in tl:
            img, y = img.to(device), y.to(device)
            logits, cpred, truths = model(img)
            if loss_mode == "ovr":
                loss = balanced_ovr_loss(truths, y, model.head_n_species)
            elif loss_mode == "hybrid":
                # softmax CE for cross-tree calibration (accuracy) +
                # down-weighted OvR for per-tree disentanglement.
                loss = (F.cross_entropy(logits, y)
                        + ovr_weight * balanced_ovr_loss(truths, y, model.head_n_species))
            else:
                loss = F.cross_entropy(logits, y)
            if concept_lam > 0.0 and concept_targets is not None:
                # SUPERVISE the bottleneck against the selected human attributes
                # (turns the emergent OCBM into a supervised-concept OCBM).
                loss = loss + concept_lam * F.binary_cross_entropy(
                    cpred, c.to(device)[:, concept_targets])
            if decorr_lam > 0.0 and cpred.shape[0] > 1:
                # DECORRELATE concept activations: penalise the squared
                # off-diagonal correlation of the batch's concept probs so the
                # emergent bottleneck cannot spend many concepts on the same
                # attribute (the yellow-back-x5 redundancy collapse). Label-free.
                zc = cpred - cpred.mean(0, keepdim=True)
                std = cpred.std(0, keepdim=True).clamp_min(1e-2)
                z = zc / std
                corr = (z.t() @ z) / z.shape[0]                # K x K
                off = corr - torch.diag(torch.diagonal(corr))
                loss = loss + decorr_lam * (off ** 2).sum() / (K * (K - 1))
            if can_freeze and not frozen:
                # entropy penalty drives routing peaked so it can be hard-frozen;
                # RAMP it with the anneal (gentle early so concepts form first,
                # strong late so the permutation sharpens before the freeze).
                loss = loss + perm_sparsity * raw * _sparsity_loss()
            if is_rect:
                # rectangular head's structural penalties (single-root pointer,
                # <=N-parent fan-out, non-compactness) from the last forward.
                loss = loss + model.head.regularization()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run += loss.item() * y.numel(); total += y.numel()
        sched.step()
        acc = evaluate(model, vl, device)                 # SOFT acc while not frozen
        # confidence-triggered hard freeze, then frozen finetuning
        if can_freeze and not frozen:
            # track the SOFT peak so we can HARDEN FROM THE BEST checkpoint rather
            # than a later, more-overfit epoch. Faithful for the straight-through
            # rect head (soft ~ hard); for the other heads it is only used when
            # explicitly early-stopping (they freeze at confidence/force anyway).
            if acc > best_soft:
                best_soft = acc
                best_soft_state = copy.deepcopy(model.state_dict())
                no_improve = 0
            else:
                no_improve += 1
            conf = _confidence()
            patience_hit = (early_stop_patience > 0 and no_improve >= early_stop_patience
                            and ep >= int(min_freeze_frac * epochs))
            if conf >= eff_freeze_conf or ep >= force_at or patience_hit:
                if best_soft_state is not None:
                    # restore the PEAK model, then commit the discrete tree from it.
                    model.load_state_dict(best_soft_state)
                    print(f"    [early-stop] restore soft-peak {best_soft * 100:.2f}% "
                          f"(ep-gap {no_improve}) before harden", flush=True)
                _freeze()
                frozen = True
                acc = evaluate(model, vl, device)         # report the FROZEN acc
                print(f"    [freeze] ep {ep + 1} conf {conf:.3f} -> "
                      f"{'rect-harden' if is_rect else ('egress' if is_ft else 'permutation')} hardened")
        # only keep hard checkpoints when hardening (soft ones are not faithful)
        if (not can_freeze or frozen) and acc > best:
            best = acc
            best_state = copy.deepcopy(model.state_dict())  # keep the PEAK model
        conf_str = (f" | conf {_confidence():.3f}"
                    if can_freeze and not frozen else (" | FROZEN" if frozen else ""))
        print(f"    epoch {ep + 1:2d}/{epochs} | loss {run / total:.3f} | "
              f"test {acc * 100:.2f}%{conf_str}")
        # periodic full-state checkpoint (crash/outage safety + resume). Saves
        # the CURRENT model/optimizer/scheduler/epoch AND the best snapshot, so
        # the run can continue after an interruption. Written to a temp file then
        # atomically renamed, so a crash mid-write cannot corrupt the checkpoint.
        if ckpt_path and ckpt_every and (ep + 1) % ckpt_every == 0:
            ck = {"model": model.state_dict(), "opt": opt.state_dict(),
                  "sched": sched.state_dict(), "epoch": ep + 1,
                  "best": best, "best_state": best_state, "frozen": frozen,
                  "K": K, "acc": best}
            tmp = ckpt_path + ".tmp"
            torch.save(ck, tmp)
            os.replace(tmp, ckpt_path)                            # atomic swap
            print(f"    [ckpt] ep {ep + 1} -> {ckpt_path} (best {best * 100:.2f}%)",
                  flush=True)
    if best_state is not None:
        model.load_state_dict(best_state)  # restore peak for --save / concept decode
    return best, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concepts", type=int, default=32)
    ap.add_argument("--scan-ks", type=str, default=None,
                    help="comma list of K to scan, e.g. 8,16,32,64")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--head", type=str, default="tree", choices=["tree", "alt", "fulltree"],
                    help="'tree'=VectorTreeLogicHead (left, real andness); "
                         "'alt'=VectorLogicHead alternating (coeff+anchor ops); "
                         "'fulltree'=VectorFullTreeHead (egress-hardened funnel, no permutation)")
    ap.add_argument("--branching", type=int, default=4,
                    help="funnel branching factor for --head fulltree")
    ap.add_argument("--loss", type=str, default="ce", choices=["ce", "ovr", "hybrid"],
                    help="'ce'=softmax cross-entropy (trees compete); "
                         "'ovr'=balanced one-vs-rest BCE (each tree an independent "
                         "detector, cross-fed down-weighted negatives); "
                         "'hybrid'=ce + ovr_weight*ovr (calibration + disentanglement)")
    ap.add_argument("--ovr-weight", type=float, default=0.3,
                    help="lambda for the OvR term in --loss hybrid")
    ap.add_argument("--harden", action="store_true",
                    help="drive routing hard (sparsity penalty) then Hungarian-freeze "
                         "the permutation + frozen finetune, so the tree is a faithful "
                         "per-concept logic tree (prune-analyzable)")
    ap.add_argument("--sinkhorn-iters", type=int, default=20,
                    help="Sinkhorn iterations (higher = sharper routing; medical uses 200)")
    ap.add_argument("--perm-sparsity", type=float, default=5.0,
                    help="weight of the permutation row-entropy penalty (--harden)")
    ap.add_argument("--freeze-conf", type=float, default=0.90,
                    help="mean peak-routing confidence that triggers the hard freeze")
    ap.add_argument("--freeze-frac", type=float, default=0.85,
                    help="force the freeze by this fraction of epochs if conf never reached")
    ap.add_argument("--anneal-cap", type=float, default=0.75,
                    help="max annealing progress (routing sharpness cap); use ~1.0 with --harden")
    ap.add_argument("--decorr-lam", type=float, default=0.0,
                    help="weight of the concept-activation decorrelation penalty "
                         "(pushes emergent concepts onto distinct attributes)")
    ap.add_argument("--save", type=str, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tl = DataLoader(_cub._CUBImages("train", True), batch_size=args.batch_size,
                    shuffle=True, num_workers=args.workers, pin_memory=True)
    vl = DataLoader(_cub._CUBImages("test", False), batch_size=128,
                    shuffle=False, num_workers=args.workers, pin_memory=True)
    names = _cub._load_attr_groups()[0]

    ks = ([int(k) for k in args.scan_ks.split(",")] if args.scan_ks
          else [args.concepts])
    print(f"CUB emergent  Ks={ks}  epochs={args.epochs}  head={args.head}  "
          f"loss={args.loss}  harden={args.harden}  device={device}")

    summary = {}
    for K in ks:
        print(f"\n=== K = {K} ===")
        best, model = train_one(K, tl, vl, device, args.epochs, args.seed,
                                head=args.head, loss_mode=args.loss,
                                ovr_weight=args.ovr_weight, harden=args.harden,
                                sinkhorn_iters=args.sinkhorn_iters,
                                perm_sparsity=args.perm_sparsity,
                                freeze_conf=args.freeze_conf,
                                freeze_frac=args.freeze_frac,
                                anneal_cap=args.anneal_cap,
                                decorr_lam=args.decorr_lam,
                                branching=args.branching)
        C, A = collect(model, vl, device)
        corr = mean_abs_corr(C)
        tops = top_attr_per_concept(C, A)
        distinct = len({a for _, a in tops})
        pos_strong = sum(1 for au, _ in tops if au >= 0.70)
        summary[K] = (best, corr, distinct, pos_strong)
        print(f"  => K={K}: acc {best * 100:.2f}%  concept-corr {corr:.3f}  "
              f"distinct-top-attrs {distinct}/{K}  pos-match(>=0.70) {pos_strong}/{K}")
        if len(ks) == 1:
            print("\n  per-concept top attribute:")
            for i, (au, a) in enumerate(tops):
                print(f"    c{i:2d}: {names[a][:40] if a >= 0 else '--'} ({au:.2f})")
            if args.save:
                torch.save({"state_dict": model.state_dict(), "K": K, "acc": best}, args.save)
                print(f"  saved -> {args.save}")

    if len(ks) > 1:
        print("\n" + "=" * 66)
        print(f"{'K':>4s} {'acc%':>8s} {'corr':>8s} {'distinct/K':>12s} {'pos>=.70/K':>12s}")
        for K in ks:
            b, cr, d, ps = summary[K]
            print(f"{K:>4d} {b * 100:>7.2f} {cr:>8.3f} {d:>8d}/{K:<3d} {ps:>8d}/{K:<3d}")
        print("\n  distinct-top-attrs & low corr = diverse concepts (no collapse);")
        print("  distinct<<K or high corr = concept collapse/redundancy.")


if __name__ == "__main__":
    main()

