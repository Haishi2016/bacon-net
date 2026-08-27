"""Unit tests for VectorFullTreeHead (vectorized full tree, egress-hardened).

Verifies:
  * output shape (batch, heads) and range (0,1);
  * single-node K=2 tree matches lsp_power_mean of the two inputs;
  * andness biases control conjunction vs disjunction;
  * egress hardening: each source feeds <= max_egress parents (rows one/ k-hot);
  * ingress is free: a destination may receive multiple sources (N-ary);
  * routing is NOT permutation-invariant (real routing, no permutation needed);
  * egress sparsity loss + confidence behave; gradients flow.

Run directly:
    py -3 tests/unit/test_vector_full_tree.py
"""

from __future__ import annotations

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _ROOT)

from bacon.vectorizedFullTree import VectorFullTreeHead              # noqa: E402
from bacon.aggregators.lsp.full_weight import lsp_power_mean         # noqa: E402

_results = []


def check(name, cond, detail=""):
    ok = bool(cond)
    _results.append((name, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    return ok


def test_shape_and_range():
    print("\n=== output shape / range / funnel widths ===")
    torch.manual_seed(0)
    K, H, B = 24, 4, 8
    head = VectorFullTreeHead(K, H, branching=4)
    head.eval()
    x = torch.rand(B, K)
    y = head(x)
    check("output shape (batch, heads)", tuple(y.shape) == (B, H), f"{tuple(y.shape)}")
    check("outputs in (0,1)", bool((y > 0).all() and (y < 1).all()))
    # funnel: starts at K, ends at 1, strictly decreasing, depth ~ log_b(K)
    check("funnel widths start K end 1", head.widths[0] == K and head.widths[-1] == 1,
          f"{head.widths}")
    check("funnel widths strictly decreasing",
          all(head.widths[i] > head.widths[i + 1] for i in range(head.depth)),
          f"{head.widths}")
    check("b=4 funnel is shallow (depth << K)", head.depth < K // 2,
          f"depth={head.depth} widths={head.widths}")
    # branching=1 recovers the reduce-by-1 triangle
    tri = VectorFullTreeHead(6, H, branching=1)
    check("branching=1 == reduce-by-1 triangle",
          tri.widths == list(range(6, 0, -1)), f"{tri.widths}")


def test_single_node_matches_power_mean():
    print("\n=== K=2 single node == lsp_power_mean ===")
    torch.manual_seed(1)
    H, B = 3, 16
    head = VectorFullTreeHead(2, H, use_gumbel=False)
    head.eval()
    x = torch.rand(B, 2)
    with torch.no_grad():
        y = head(x)
        # one layer 2->1: both sources route to the single dest, weights ~0.5/0.5
        E = head._egress(0)                       # [H,2,1]
        w = head._child_weights(E)                # [H,2,1] sums to 1 over sources
        a = torch.sigmoid(head.andness_bias[0]) * 3 - 1   # [H,1]
        X = x.permute(1, 0).unsqueeze(-1).unsqueeze(-1).expand(2, B, H, 1)
        Wn = w.permute(1, 0, 2).unsqueeze(1)
        ref = lsp_power_mean(X, a.unsqueeze(0), Wn).squeeze(-1)
    check("K=2 head == manual lsp_power_mean", torch.allclose(y, ref, atol=1e-6),
          f"max|d|={float((y-ref).abs().max()):.2e}")


def test_andness_controls_logic():
    print("\n=== andness controls conjunction/disjunction ===")
    H, B = 1, 32
    x = torch.rand(B, 3)
    xmin, xmax = x.min(1).values, x.max(1).values
    hi = VectorFullTreeHead(3, H, use_gumbel=False)
    lo = VectorFullTreeHead(3, H, use_gumbel=False)
    with torch.no_grad():
        for p in hi.andness_bias:
            p.fill_(6.0)     # sigmoid(6)*3-1 ~ 1.99 -> hard-AND
        for p in lo.andness_bias:
            p.fill_(-6.0)    # ~ -1 -> hard-OR
        y_and = hi(x).squeeze(1)
        y_or = lo(x).squeeze(1)
    check("high andness => conjunctive (<= min + tol)", bool((y_and <= xmin + 1e-2).all()),
          f"mean(y_and-min)={float((y_and-xmin).mean()):.3f}")
    check("low andness => disjunctive (>= max - tol)", bool((y_or >= xmax - 1e-2).all()),
          f"mean(y_or-max)={float((y_or-xmax).mean()):.3f}")


def test_egress_hardening():
    print("\n=== egress hardening (each source <= max_egress parents) ===")
    torch.manual_seed(2)
    K, H = 6, 4
    for me in (1, 2):
        head = VectorFullTreeHead(K, H, max_egress=me)
        head.freeze_egress()
        ok_bin = True
        ok_egress = True
        ingress_multi = False
        for l in range(head.depth):
            R = getattr(head, f"frozen_route_{l}")            # [H, w_in, w_out]
            ok_bin = ok_bin and bool(((R == 0) | (R == 1)).all())
            # egress = per-source outgoing count <= max_egress
            ok_egress = ok_egress and bool((R.sum(dim=2) <= me + 1e-6).all())
            # ingress: some destination receives >1 source (N-ary), when possible
            if R.size(1) > R.size(2):
                ingress_multi = ingress_multi or bool((R.sum(dim=1) > 1).any())
        check(f"max_egress={me}: routes binary after freeze", ok_bin)
        check(f"max_egress={me}: each source feeds <= {me} parent(s)", ok_egress)
        if me == 1:
            check("ingress is FREE: some node aggregates multiple children (N-ary)",
                  ingress_multi)


def test_not_permutation_invariant():
    print("\n=== routing is real (NOT permutation-invariant) ===")
    torch.manual_seed(3)
    K, H, B = 5, 3, 8
    head = VectorFullTreeHead(K, H, use_gumbel=False)
    head.eval()
    x = torch.rand(B, K)
    perm = torch.randperm(K)
    with torch.no_grad():
        y1 = head(x)
        y2 = head(x[:, perm])
    check("permuting inputs changes output (no permutation layer needed)",
          not torch.allclose(y1, y2, atol=1e-4), f"max|d|={float((y1-y2).abs().max()):.2e}")


def test_egress_loss_and_grads():
    print("\n=== egress loss, confidence, gradients ===")
    torch.manual_seed(4)
    K, H, B = 6, 4, 8
    head = VectorFullTreeHead(K, H)
    x = torch.rand(B, K)
    loss = head(x).sum() + head.egress_sparsity_loss()
    loss.backward()
    g_route = all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.route_logits)
    g_and = all(p.grad is not None and torch.isfinite(p.grad).all() for p in head.andness_bias)
    check("gradients flow to routing logits", g_route)
    check("gradients flow to andness biases", g_and)
    conf = float(head.egress_confidence())
    check("egress confidence in [0,1]", 0.0 <= conf <= 1.0, f"conf={conf:.3f}")
    # annealing sharpens temperature toward final
    head.anneal(1.0)
    check("anneal drives temperature to final", abs(float(head.temperature) - head._temp_final) < 1e-5,
          f"temp={float(head.temperature):.3f}")


def test_negation():
    print("\n=== identity/negation gate ===")
    torch.manual_seed(0)
    head = VectorFullTreeHead(8, 4, branching=4, max_egress=2,
                              use_coefficients=True, use_negation=True)
    x = torch.rand(16, 8)
    out = head(x)
    check("negation soft output shape", tuple(out.shape) == (16, 4))
    check("negation output bounded", bool((out > 0).all() and (out < 1).all()))
    # a leaf forced to negate returns 1 - c
    head.transform_logits.data[:, 0, 0] = -10.0
    head.transform_logits.data[:, 0, 1] = 10.0
    v = (torch.zeros(2, 8) + 0.9).unsqueeze(1).expand(2, 4, 8)
    neg = head._apply_negation(v)
    check("forced-negate leaf gives 1-c", abs(float(neg[0, 0, 0]) - 0.1) < 1e-4,
          f"got {float(neg[0, 0, 0]):.4f}")
    # freeze commits the gate to a binary polarity (stays readable)
    head.anneal(1.0); head.freeze_egress()
    check("transform frozen after freeze", bool(head.transform_frozen))
    check("frozen_transform is binary",
          set(head.frozen_transform.unique().tolist()) <= {0.0, 1.0})
    check("frozen forward shape", tuple(head(x).shape) == (16, 4))
    # gradients reach the gate
    h2 = VectorFullTreeHead(8, 4, branching=4, use_negation=True)
    h2(torch.rand(4, 8)).sum().backward()
    check("gradients flow to transform_logits",
          h2.transform_logits.grad is not None and torch.isfinite(h2.transform_logits.grad).all())
    # off by default: no transform params, negation is a no-op
    h3 = VectorFullTreeHead(8, 4, branching=4)
    check("no negation params when off", not hasattr(h3, "transform_logits"))


def main():
    test_shape_and_range()
    test_negation()
    test_single_node_matches_power_mean()
    test_andness_controls_logic()
    test_egress_hardening()
    test_not_permutation_invariant()
    test_egress_loss_and_grads()
    n_pass = sum(1 for _, ok in _results if ok)
    n_tot = len(_results)
    print(f"\n{'='*60}\nSUMMARY: {n_pass}/{n_tot} checks passed")
    return 0 if n_pass == n_tot else 1


if __name__ == "__main__":
    raise SystemExit(main())
