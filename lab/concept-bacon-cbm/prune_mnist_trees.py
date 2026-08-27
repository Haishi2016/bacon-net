"""Structural prune analysis of the hardened MNIST digit trees.

Each digit tree is a scalar binaryTreeLogicNet (hard-frozen). We apply BACON's
STRUCTURAL prune_features (bypass a leaf; pruned aggregators use raw [1,0]/[0,1]
weights directly, no re-normalization) cumulatively in leaf order, keeping a
prune whenever it does not drop overall test accuracy below baseline - tol.
No greedy search. We then print the SIMPLIFIED graded-logic expression (pruned
leaves dropped, their nodes collapsed to pass-through).

    python prune_mnist_trees.py --load saved/k5_harden.pt --tol 0.001
"""

from __future__ import annotations

import argparse
import copy
import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _HERE)
sys.path.insert(0, _REPO_ROOT)

from train_emergent_concepts import MultiTreeBaconCBM, make_loaders, evaluate  # noqa: E402


def load_model(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    m = MultiTreeBaconCBM(ckpt["K"], weight_mode=ckpt.get("weight_mode", "trainable"),
                          no_negation=ckpt.get("no_negation", False),
                          device=device).to(device)
    if ckpt.get("frozen"):
        m.prepare_frozen_structure(); m.load_state_dict(ckpt["state_dict"])
    else:
        m.load_state_dict(ckpt["state_dict"]); m.anneal(1.0)
    m.eval()
    return m, ckpt


def _andness(a):
    return "AND" if a > 0.9 else ("OR" if a < 0.1 else "MEAN")


def tree_static(tree, K):
    """Capture leaf concepts, negation, per-node andness + softmax weights."""
    lc = tree.input_to_leaf.P_hard.argmax(1).tolist()
    tl = tree.transformation_layer
    tnames = [type(t).__name__.replace("Transformation", "") for t in tl.transformations]
    neg = [tnames[i] == "Negation" for i in tl.logits.argmax(1).tolist()]
    andn = [(torch.sigmoid(tree.biases[i]) * 3 - 1).item() for i in range(tree.num_layers)]
    sw = [torch.softmax(tree.weights[i].detach(), dim=0).tolist() for i in range(tree.num_layers)]
    return lc, neg, andn, sw


def leaf_str(lc, neg, j):
    return ("\\bar c_{%d}" % lc[j]) if neg[j] else ("c_{%d}" % lc[j])


def simplified_expr(lc, neg, andn, sw, K, pruned):
    """Render the left-fold with pruned leaves dropped / their nodes collapsed.
    Returns (latex_expr, surviving_leaf_indices)."""
    L = lambda j: leaf_str(lc, neg, j)
    surv = []
    # seed = result after node 0 (combines leaf 0 and leaf 1); prune_features on
    # leaf0/leaf1 both bypass agg0 -> exactly one of them survives.
    if 0 in pruned and 1 not in pruned:
        acc = L(1); surv = [1]
    elif 1 in pruned and 0 not in pruned:
        acc = L(0); surv = [0]
    elif 0 in pruned and 1 in pruned:
        acc = L(0); surv = [0]
    else:
        acc = (f"\\mathcal{{A}}_{{{andn[0]:.2f}}}[{L(0)}{{:}}{sw[0][0]:.2f},\\ "
               f"{L(1)}{{:}}{sw[0][1]:.2f}]")
        surv = [0, 1]
    for node in range(1, K - 1):
        leaf = node + 1
        if leaf in pruned:
            continue
        surv.append(leaf)
        acc = (f"\\mathcal{{A}}_{{{andn[node]:.2f}}}[{acc}{{:}}{sw[node][0]:.2f},\\ "
               f"{L(leaf)}{{:}}{sw[node][1]:.2f}]")
    return acc, surv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", type=str, default=os.path.join(_HERE, "saved", "k5_harden.pt"))
    ap.add_argument("--tol", type=float, default=0.001,
                    help="max allowed drop in overall test accuracy while pruning")
    ap.add_argument("--data", type=str,
                    default=os.path.join(_REPO_ROOT, "benchmarks", "mnist-addition", "data"))
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_model(args.load, device)
    K = ckpt["K"]
    _, test_ld = make_loaders(args.data, 256)
    base_acc, _ = evaluate(model, test_ld, device)
    print(f"\nSTRUCTURAL PRUNE  {os.path.basename(args.load)}  K={K}  "
          f"baseline acc {base_acc*100:.2f}%  tol {args.tol}")

    static = {d: tree_static(model.trees[d], K) for d in range(10)}
    orig_w = {d: [w.data.clone() for w in model.trees[d].weights] for d in range(10)}
    pruned = {d: set() for d in range(10)}

    def reapply(d, pset):
        t = model.trees[d]
        for i, w in enumerate(t.weights):
            w.data.copy_(orig_w[d][i])
        t.pruned_aggregators.clear()
        for leaf in sorted(pset):
            t.prune_features(leaf)

    for d in range(10):
        for leaf in range(K):
            reapply(d, pruned[d] | {leaf})
            acc, _ = evaluate(model, test_ld, device)
            if acc >= base_acc - args.tol:
                pruned[d].add(leaf)
            else:
                reapply(d, pruned[d])          # revert this leaf
        reapply(d, pruned[d])                   # leave committed prunes applied

    final_acc, _ = evaluate(model, test_ld, device)
    total_leaves = 10 * K
    total_pruned = sum(len(p) for p in pruned.values())
    print(f"final acc {final_acc*100:.2f}%   pruned {total_pruned}/{total_leaves} leaves "
          f"({total_pruned/total_leaves*100:.0f}%)\n")

    for d in range(10):
        lc, neg, andn, sw = static[d]
        expr, surv = simplified_expr(lc, neg, andn, sw, K, pruned[d])
        concepts = sorted({lc[j] for j in surv})
        print(f"digit {d}:  {len(surv)}/{K} leaves survive  "
              f"(concepts {['c%d' % c for c in concepts]})")
        print(f"   $ {expr} $")


if __name__ == "__main__":
    main()
