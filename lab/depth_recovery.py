r"""Synthetic depth-recovery experiment (#2): can the recttree head LEARN deep
graded-logic structure when the task PROVABLY requires it?

We build a ground-truth full binary AND/OR formula of depth D over K = 2**D
distinct binary variables (random operator per node, random leaf negations),
sample binary inputs, and label them by evaluating the formula. Then we train a
VectorRectTreeHead whose funnel widths match the target tree
([2**(D-1), ..., 1]) and ask whether it (a) FITS the function and (b) RECOVERS
crisp per-layer operators (andness -> 1 for AND, 0 for OR) rather than collapsing
the deep layers to neutral (a~0.5).

Three arms isolate optimization vs structure:
  soft            : plain soft training (baseline)
  straight_through: forward uses the hard top-N structure (trains the discrete
                    tree directly -> undiluted gradients)
  grad_amp        : soft training + per-layer backward gradient amplification
                    gamma**(depth-1-l) on the LOWER (leaf-adjacent) layers, to
                    counter the geometric gradient decay through conjunctions
  hard_anneal     : soft training + a RAMPED binarize (gate-entropy) penalty that
                    polarizes the edge gates toward {0,1} before harden, directly
                    targeting the soft->hard discretization gap

This SEPARATES "structurally can't optimize depth" from "CUB just doesn't need
depth": if soft fails to recover a depth-4 formula but straight_through/grad_amp
succeed, the vanishing-gradient limit is real and fixable; if all recover it,
the head can learn depth and CUB's shallowness is dataset nature.

    py -3 lab/depth_recovery.py
"""
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from bacon.vectorizedRectTree import VectorRectTreeHead   # noqa: E402


# ------------------------------------------------------------- target formula
def make_formula(depth, rng):
    """Full binary AND/OR tree over K=2**depth vars; returns (eval_fn, layer_ops).

    layer_ops[L] is the list of {'AND','OR'} at internal layer L (L=1 just above
    the leaves ... L=depth is the root), for scoring recovered andness by layer.
    """
    K = 2 ** depth
    negate = [bool(rng.integers(0, 2)) for _ in range(K)]      # per-leaf NOT

    # build a nested structure of indices; internal nodes carry an op.
    def build(lo, hi, level):
        if hi - lo == 1:
            return {"leaf": lo}
        mid = (lo + hi) // 2
        op = "AND" if rng.integers(0, 2) == 0 else "OR"
        return {"op": op, "level": level,
                "l": build(lo, mid, level - 1), "r": build(mid, hi, level - 1)}

    root = build(0, K, depth)
    layer_ops = {L: [] for L in range(1, depth + 1)}

    def collect(node):
        if "leaf" in node:
            return
        layer_ops[node["level"]].append(node["op"])
        collect(node["l"]); collect(node["r"])
    collect(root)

    def ev(x):                                                 # x: (B,K) in {0,1}
        def rec(node):
            if "leaf" in node:
                v = x[:, node["leaf"]]
                return (1 - v) if negate[node["leaf"]] else v
            a, b = rec(node["l"]), rec(node["r"])
            return torch.minimum(a, b) if node["op"] == "AND" else torch.maximum(a, b)
        return rec(root)
    return ev, layer_ops, K


def gen_data(ev, K, n, rng):
    X = torch.tensor(rng.integers(0, 2, size=(n, K)), dtype=torch.float32)
    y = ev(X)
    return X, y


# ------------------------------------------------------------------ head build
def build_head(K, depth, straight_through=False):
    widths = [2 ** (depth - 1 - i) for i in range(depth)]      # [2^(D-1),...,1]
    return VectorRectTreeHead(
        K, num_heads=1, layer_widths=widths, max_parents=1,
        root_lam=0.0, parent_lam=0.0, compact_lam=0.0,          # pure fit/recovery
        straight_through=straight_through, use_negation=True,
        normalize_andness=True, temperature=2.0, final_temperature=0.05)


def layer_andness(head, l):
    a = torch.sigmoid(head.andness_bias[l][0]) * 3.0 - 1.0      # (w_out,)
    return a.detach()


# ----------------------------------------------------------------------- train
def run_arm(depth, arm, seed, epochs=500):
    import numpy as np
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    ev, layer_ops, K = make_formula(depth, rng)
    Xtr, ytr = gen_data(ev, K, 6000, rng)
    Xte, yte = gen_data(ev, K, 4000, rng)

    head = build_head(K, depth, straight_through=(arm == "straight_through"))
    if arm == "grad_amp":
        gamma = 3.0
        for l in range(depth):
            s = gamma ** (depth - 1 - l)                        # amplify LOWER layers
            head.edge_logits[l].register_hook(lambda g, s=s: g * s)
            head.andness_bias[l].register_hook(lambda g, s=s: g * s)
    opt = torch.optim.Adam(head.parameters(), lr=0.05)
    for ep in range(epochs):
        head.train()
        head.anneal(ep / epochs)
        if arm == "hard_anneal":
            # ramp the gate-entropy penalty in over the 2nd half of training so
            # edges polarize to {0,1} before harden (closes the discretization gap).
            head.binarize_lam = 1.0 * max(0.0, (ep / epochs - 0.3) / 0.7)
        opt.zero_grad()
        out = head(Xtr).squeeze(1).clamp(1e-6, 1 - 1e-6)
        loss = F.binary_cross_entropy(out, ytr) + head.regularization()
        loss.backward()
        opt.step()

    head.eval()
    with torch.no_grad():
        soft = ((head(Xte).squeeze(1) > 0.5).float() == yte).float().mean().item()
    # per-layer decisiveness of learned operators (|a-0.5| > 0.3 => committed)
    decis = []
    for L in range(1, depth + 1):
        a = layer_andness(head, L - 1)
        decis.append(float((a.sub(0.5).abs() > 0.3).float().mean()))
    head.harden()
    with torch.no_grad():
        hard = ((head(Xte).squeeze(1) > 0.5).float() == yte).float().mean().item()
    return soft, hard, decis


def main():
    depths = [2, 3, 4]
    arms = ["soft", "straight_through", "grad_amp", "hard_anneal"]
    seeds = [0, 1, 2]
    print(f"{'depth':<6}{'arm':<18}{'soft_fit':>9}{'hard_fit':>9}   decisive-by-layer (leaf->root)")
    for D in depths:
        for arm in arms:
            s_acc, h_acc, dec = [], [], []
            for sd in seeds:
                s, h, d = run_arm(D, arm, sd)
                s_acc.append(s); h_acc.append(h); dec.append(d)
            s_m = sum(s_acc) / len(s_acc)
            h_m = sum(h_acc) / len(h_acc)
            dec_m = [sum(x[i] for x in dec) / len(dec) for i in range(D)]
            dtxt = " ".join(f"{v:.2f}" for v in dec_m)
            print(f"{D:<6}{arm:<18}{s_m:>9.3f}{h_m:>9.3f}   [{dtxt}]")


if __name__ == "__main__":
    main()
