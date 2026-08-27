"""Validate VectorFullTreeHead end-to-end on MNIST (does the full tree LEARN?).

Reuses the existing ConceptCNN encoder + MNIST loaders, but replaces the 10
per-class binaryTreeLogicNet trees with ONE vectorized full tree (branching
funnel, egress-hardened, NO permutation). Confirms the head trains to good
accuracy and hard-freezes without collapse.

    py -3 -u validate_fulltree_mnist.py --concepts 5 --epochs 30 --branching 4
"""

from __future__ import annotations

import argparse
import copy
import os
import sys

import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, _REPO_ROOT)

from bacon.vectorizedFullTree import VectorFullTreeHead          # noqa: E402
from bacon.vectorizedHybridTree import VectorHybridTreeHead      # noqa: E402
from model import ConceptCNN                                     # noqa: E402
from train import make_loaders                                   # noqa: E402


class FullTreeCBM(torch.nn.Module):
    def __init__(self, K, n_classes=10, branching=4, max_egress=1,
                 weight_mode="static", straight_through=False, use_coefficients=False,
                 logit_temperature=4.0, bin_frac=0.0, use_negation=False):
        super().__init__()
        self.encoder = ConceptCNN(K)
        if bin_frac and bin_frac > 0:
            # hybrid: binary spine over the important features + full sub-tree
            self.head = VectorHybridTreeHead(K, n_classes, bin_frac=bin_frac,
                                             branching=branching, max_egress=max_egress,
                                             use_coefficients=use_coefficients)
        else:
            self.head = VectorFullTreeHead(K, n_classes, branching=branching,
                                           max_egress=max_egress, weight_mode=weight_mode,
                                           straight_through=straight_through,
                                           use_coefficients=use_coefficients,
                                           use_negation=use_negation)
        self.log_temp = torch.nn.Parameter(
            torch.tensor(float(torch.log(torch.tensor(logit_temperature)))))

    def concept_probs(self, x):
        return torch.sigmoid(self.encoder(x))

    def forward(self, x):
        probs = self.concept_probs(x)
        truths = self.head(probs).clamp(1e-6, 1 - 1e-6)          # (B, C)
        logits = self.log_temp.exp() * (torch.log(truths) - torch.log1p(-truths))
        return logits, probs, truths


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct = total = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits, _, _ = model(x)
        correct += (logits.argmax(1) == y).sum().item()
        total += y.numel()
    return correct / total


@torch.no_grad()
def _collect_concepts(model, loader, device, cap=2048):
    """Gather up to ``cap`` head-input concept activations for scan calibration."""
    model.eval()
    xs, n = [], 0
    for x, _ in loader:
        c = model.concept_probs(x.to(device))
        xs.append(c); n += c.size(0)
        if n >= cap:
            break
    return torch.cat(xs)[:cap]


def _andness_label(a: float) -> str:
    if a >= 1.5:
        return "AND!"          # hard conjunction
    if a >= 0.7:
        return "AND"
    if a > 0.55:
        return "wAND"
    if a >= 0.45:
        return "MEAN"
    if a > 0.3:
        return "wOR"
    if a > -0.5:
        return "OR"
    return "OR!"                # hard disjunction


@torch.no_grad()
def _dest_to_src(head, h):
    """Per layer, map each destination node -> list of source nodes routed to it
    (uses the hardened one-hot egress routing; only the '1' edges survive)."""
    maps = []
    for l in range(head.depth):
        R = getattr(head, f"frozen_route_{l}")[h]        # [w_in, w_out], one-hot rows
        dest = R.argmax(dim=1)
        d2s = {}
        for i in range(R.size(0)):
            d2s.setdefault(int(dest[i]), []).append(i)
        maps.append(d2s)
    return maps


@torch.no_grad()
def extract_tree(head, h, concept_names=None):
    """Build the cleaned graded-logic tree for head h. Cleanup: drop unrouted
    (zero) edges, prune dead (childless) nodes, collapse single-child pass-throughs."""
    d2s = _dest_to_src(head, h)
    andness = [torch.sigmoid(head.andness_bias[l][h]) * 3 - 1 for l in range(head.depth)]
    names = concept_names or [f"c{i}" for i in range(head.input_size)]

    def build(l, j):
        if l == 0:
            return {"leaf": names[j]}
        kids_idx = d2s[l - 1].get(j, [])
        kids = [build(l - 1, i) for i in kids_idx]
        kids = [k for k in kids if k is not None]        # prune dead subtrees
        if not kids:
            return None                                   # dead node
        if len(kids) == 1:
            return kids[0]                                # collapse pass-through
        return {"a": float(andness[l - 1][j]), "kids": kids}

    return build(head.depth, 0)                           # root = last layer, node 0


def _fmt_tree(node):
    if node is None:
        return "(dead)"
    if "leaf" in node:
        return node["leaf"]
    lbl = _andness_label(node["a"])
    return f"{lbl}[{node['a']:+.2f}]({', '.join(_fmt_tree(k) for k in node['kids'])})"


def _collect_stats(node, andvals, leaves):
    if node is None or "leaf" in node:
        if node is not None:
            leaves.add(node["leaf"])
        return
    andvals.append(node["a"])
    for k in node["kids"]:
        _collect_stats(k, andvals, leaves)


def print_trees(model, class_names=None):
    head = model.head
    if not bool(head.egress_frozen):
        print("  (routing not frozen -- freeze first to read a hard tree)")
        return
    print(f"\n=== learned full trees (widths {head.widths}, egress-hardened, "
          f"static andness) ===")
    all_and, sizes = [], []
    for h in range(head.num_heads):
        tree = extract_tree(head, h)
        avals, leaves = [], set()
        _collect_stats(tree, avals, leaves)
        all_and += avals
        sizes.append(len(leaves))
        name = class_names[h] if class_names else f"class {h}"
        print(f"\n  {name}: uses {len(leaves)}/{head.input_size} concepts, "
              f"{len(avals)} agg nodes")
        print(f"    {_fmt_tree(tree)}")
    import numpy as np
    a = np.array(all_and) if all_and else np.array([0.0])
    n_and = int((a >= 0.55).sum()); n_or = int((a <= 0.45).sum())
    n_mid = len(a) - n_and - n_or
    print(f"\n  andness over {len(a)} nodes: mean {a.mean():+.2f}  "
          f"AND(>0.55) {n_and}  MEAN {n_mid}  OR(<0.45) {n_or}")
    print(f"  concepts used per class: mean {np.mean(sizes):.1f} "
          f"(min {min(sizes)}, max {max(sizes)})")
    print("  NOTE: static continuous andness only -- partial absorption / value "
          "& conditional gates are NOT enabled in this head yet.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--concepts", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--branching", type=int, default=4)
    ap.add_argument("--weight-mode", type=str, default="static",
                    choices=["static", "value", "full"],
                    help="static | value (dynamic andness) | full (+partial absorption)")
    ap.add_argument("--straight-through", action="store_true",
                    help="hard-argmax egress in forward (trains AS the discrete tree)")
    ap.add_argument("--coefficients", action="store_true",
                    help="add smooth per-source coefficient (relevance) layers")
    ap.add_argument("--scan", type=int, default=0,
                    help="candidate-scan freeze: sample N hard trees from the soft "
                         "routing and keep the per-head most faithful (0 = greedy argmax)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--perm-sparsity", type=float, default=5.0)
    ap.add_argument("--freeze-conf", type=float, default=0.90)
    ap.add_argument("--freeze-frac", type=float, default=0.85)
    ap.add_argument("--inspect", action="store_true", help="print the learned trees")
    ap.add_argument("--bin-frac", type=float, default=0.0,
                    help="hybrid head: fraction of inputs on the binary spine (0 = pure full tree)")
    ap.add_argument("--max-egress", type=int, default=1,
                    help="parents each node may feed after hardening (1=tree; >1=DAG)")
    ap.add_argument("--negation", action="store_true",
                    help="per-concept identity/negation gate (lets leaves read NOT c)")
    ap.add_argument("--save", type=str, default=None)
    ap.add_argument("--data", default=os.path.join(_REPO_ROOT, "benchmarks",
                                                   "mnist-addition", "data"))
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ld, test_ld = make_loaders(args.data, 256)
    model = FullTreeCBM(args.concepts, branching=args.branching,
                        weight_mode=args.weight_mode,
                        straight_through=args.straight_through,
                        use_coefficients=args.coefficients,
                        max_egress=args.max_egress,
                        bin_frac=args.bin_frac,
                        use_negation=args.negation).to(device)
    _widths = getattr(model.head, "widths", None) or model.head.full.widths
    _kind = f"hybrid(bin={model.head.n_bin},full={model.head.n_full})" if args.bin_frac > 0 else "fulltree"
    print(f"{_kind} widths={_widths}  K={args.concepts}  "
          f"mode={args.weight_mode}  egress={args.max_egress}  device={device}", flush=True)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(args.epochs, 1))
    best_acc, best_state = 0.0, None
    frozen = False
    force_at = int(args.freeze_frac * args.epochs)

    for ep in range(args.epochs):
        model.train()
        prog = min(1.0, ep / max(1.0, 0.7 * args.epochs - 1))
        if not frozen:
            model.head.anneal(prog)
        run = tot = 0
        for x, y in train_ld:
            x, y = x.to(device), y.to(device)
            logits, _, _ = model(x)
            loss = F.cross_entropy(logits, y)
            loss = loss + model.head.transform_regularization()
            if not frozen:
                loss = loss + args.perm_sparsity * prog * model.head.egress_sparsity_loss()
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            run += loss.item() * y.numel(); tot += y.numel()
        sched.step()

        if not frozen:
            conf = float(model.head.egress_confidence())
            if conf >= args.freeze_conf or ep >= force_at:
                if args.scan > 0:
                    calib = _collect_concepts(model, test_ld, device, cap=2048)
                    mse = model.head.freeze_egress_scan(
                        calib, num_candidates=args.scan)
                    print(f"    [scan-freeze] ep {ep+1} conf {conf:.3f} | "
                          f"{args.scan} candidates -> faithfulness MSE {float(mse):.4e}",
                          flush=True)
                else:
                    model.head.freeze_egress()
                    print(f"    [freeze] ep {ep+1} egress-conf {conf:.3f} -> hardened",
                          flush=True)
                frozen = True

        acc = evaluate(model, test_ld, device)
        if frozen and acc > best_acc:
            best_acc = acc
            best_state = copy.deepcopy(model.state_dict())
        tag = "FROZEN" if frozen else f"conf {float(model.head.egress_confidence()):.3f}"
        print(f"epoch {ep+1:2d}/{args.epochs} | loss {run/tot:.3f} | "
              f"test {acc*100:.2f}% | {tag}", flush=True)

    print(f"\nBEST (frozen/faithful) MNIST acc: {best_acc*100:.2f}%", flush=True)

    if best_state is not None:
        model.load_state_dict(best_state)
    if args.save:
        torch.save({"state_dict": model.state_dict(), "K": args.concepts,
                    "branching": args.branching, "acc": best_acc}, args.save)
        print(f"saved -> {args.save}", flush=True)
    if args.inspect:
        if not bool(model.head.egress_frozen):
            model.head.freeze_egress()
        print_trees(model)


if __name__ == "__main__":
    main()
