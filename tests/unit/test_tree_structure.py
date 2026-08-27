"""Structural unit tests for the BACON tree modules.

Verifies the *intended* structure of the two "lift the binary-tree limit" trees
and clarifies whether the alternating design is needed:

  * FullyConnectedTree  (bacon/fullyConnectedTree.py)
  * AlternatingTree      (bacon/alternatingTree.py)
  * VectorLogicHead      (bacon/vectorizedLogicHead.py) -- the vectorized head
    the CUB experiment actually uses; included to show the GL routing-collapse.

Run directly (no pytest needed):
    py -3 tests/unit/test_tree_structure.py
"""

from __future__ import annotations

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _ROOT)

from bacon.fullyConnectedTree import FullyConnectedTree      # noqa: E402
from bacon.alternatingTree import AlternatingTree            # noqa: E402
from bacon.vectorizedLogicHead import VectorLogicHead        # noqa: E402


class MockAggregator:
    """Minimal aggregator: records andness seen, returns a weighted mean.

    Lets us test *tree structure/routing/andness* independent of any specific
    LSP/GL aggregator math. Matches the interface the trees call:
    ``aggregate(values_list, andness, weights_list)`` + optional start_forward.
    """

    uses_edge_scales = True

    def __init__(self):
        self.andness_seen = []

    def start_forward(self):
        self.andness_seen = []

    def aggregate(self, values, andness, weights):
        self.andness_seen.append(
            float(andness) if torch.is_tensor(andness) else float(andness)
        )
        num = sum(w * v for w, v in zip(weights, values))
        den = sum(weights) + 1e-8
        return num / den


_results = []


def check(name, cond, detail=""):
    ok = bool(cond)
    _results.append((name, ok, detail))
    mark = "PASS" if ok else "FAIL"
    print(f"  [{mark}] {name}" + (f"  -- {detail}" if detail else ""))
    return ok


# --------------------------------------------------------------------------- #
def test_fully_connected_tree():
    print("\n=== FullyConnectedTree ===")
    n = 6
    dev = torch.device("cpu")
    t = FullyConnectedTree(num_inputs=n, shape="triangle", max_egress=1,
                           device=dev)

    # (1) Triangle structure: widths n, n-1, ..., 1
    check("triangle layer widths",
          t.layer_widths == list(range(n, 0, -1)),
          f"widths={t.layer_widths}")
    check("depth = n-1", t.depth == n - 1, f"depth={t.depth}")

    # (2) Forward shape
    agg = MockAggregator()
    x = torch.rand(4, n)
    y = t(x, aggregator=agg)
    check("forward output shape (batch,1)", tuple(y.shape) == (4, 1),
          f"got {tuple(y.shape)}")

    # (3) Andness is LEARNABLE and varies per node (biases -> sigmoid*3-1)
    with torch.no_grad():
        for b in t.biases:
            b.copy_(torch.randn_like(b) * 2.0)
    agg = MockAggregator()
    _ = t(x, aggregator=agg)
    andness = torch.tensor(agg.andness_seen)
    check("andness in (-1,2)", bool((andness >= -1.001).all() and (andness <= 2.001).all()),
          f"range=[{andness.min():.2f},{andness.max():.2f}]")
    check("andness VARIES across nodes (learnable graded andness)",
          float(andness.std()) > 1e-3, f"std={float(andness.std()):.3f}")

    # (4) Soft routing with max_egress is ROW-normalized (each source's
    #     outgoing weights sum to 1 -> egress budget of 1 per source)
    w0 = t._compute_edge_weights(0)  # (in, out)
    row_sums = w0.sum(dim=1)
    check("soft egress: rows sum to 1 (row-softmax)",
          torch.allclose(row_sums, torch.ones_like(row_sums), atol=1e-4),
          f"row_sums[:3]={row_sums[:3].tolist()}")

    # (5) Hardening to egress=1 (each source -> exactly ONE parent)
    t.harden(mode="argmax_row")
    edges = t.get_edge_weights()
    egress_ok = all(
        torch.all(e.sum(dim=1) <= 1.0 + 1e-6) and
        torch.all((e == 0) | (e == 1))
        for e in edges
    )
    per_source_one = all(torch.all(e.sum(dim=1) == 1.0) for e in edges)
    check("hardened edges are binary (0/1)",
          all(torch.all((e == 0) | (e == 1)) for e in edges))
    check("EGRESS<=1: each source contributes to at most one parent",
          egress_ok and per_source_one)

    # (6) Contrast: default 'argmax' hardening is INGRESS (col=1), not egress.
    t.unharden()
    t.harden(mode="argmax")
    edges_c = t.get_edge_weights()
    col_one = all(torch.all(e.sum(dim=0) == 1.0) for e in edges_c)
    some_row_multi = any(torch.any(e.sum(dim=1) > 1.0) for e in edges_c)
    check("NOTE: mode='argmax' gives ingress=1 (col sums=1), can break egress",
          col_one, f"col=1:{col_one}  some_source>1_parent:{some_row_multi}")


# --------------------------------------------------------------------------- #
def test_alternating_tree():
    print("\n=== AlternatingTree ===")
    n = 6
    dev = torch.device("cpu")
    t = AlternatingTree(num_inputs=n, max_egress=1, use_straight_through=True,
                        device=dev)

    # (1) Structure: agg widths n-1..1, coeff layers n..2, triangle convergence
    agg_widths = [l.out_width for l in t.agg_layers]
    check("agg layer out-widths = [n-1..1]",
          agg_widths == list(range(n - 1, 0, -1)), f"{agg_widths}")
    check("num_agg_nodes = n(n-1)/2",
          t.num_agg_nodes == n * (n - 1) // 2, f"{t.num_agg_nodes}")
    check("coeff layers separate from agg layers",
          len(t.coeff_layers) == n - 1, f"coeff_layers={len(t.coeff_layers)}")

    # (2) Forward shape
    agg = MockAggregator()
    x = torch.rand(4, n)
    y = t(x, aggregator=agg)
    check("forward output shape (batch,1)", tuple(y.shape) == (4, 1),
          f"got {tuple(y.shape)}")

    # (3) ANDNESS: is it learned?  (the key 'is alternating needed' question)
    andness = torch.tensor(agg.andness_seen)
    fixed_half = bool(torch.allclose(andness, torch.full_like(andness, 0.5)))
    check("CLARIFY: AlternatingTree uses FIXED andness=0.5 (does NOT learn andness)",
          fixed_half,
          f"unique andness values={sorted(set(round(a,3) for a in agg.andness_seen))}")

    # (4) Coefficients ARE learnable (separate weight layers)
    has_coeff_params = any(
        c.log_coefficients.requires_grad for c in t.coeff_layers
    )
    check("coefficients (weights) are learnable in dedicated layers",
          has_coeff_params)

    # (5) Egress=1 after harden (straight-through routing, one parent per source)
    t.harden()
    first = t.agg_layers[0]
    e = first.get_edge_weights()
    binary = bool(torch.all((e == 0) | (e == 1)))
    egress_one = bool(torch.all(e.sum(dim=1) == 1.0))
    check("first agg layer routing is binary after harden", binary)
    check("EGRESS<=1: each source routes to exactly one parent", egress_one)


# --------------------------------------------------------------------------- #
def test_vectorized_routing_collapse():
    print("\n=== VectorLogicHead (vectorized head used by CUB) ===")
    torch.manual_seed(0)
    n, H, B = 5, 3, 4
    x = torch.rand(B, n)
    perm = torch.randperm(n)

    # 'full' layout: static GL blend -> PERMUTATION INVARIANT (no real routing)
    hf = VectorLogicHead(input_size=n, num_heads=H, layout="full")
    hf.eval()
    with torch.no_grad():
        y1 = hf(x)
        y2 = hf(x[:, perm])
    check("layout='full' is PERMUTATION-INVARIANT (routing collapses to 1 blend)",
          torch.allclose(y1, y2, atol=1e-6),
          f"max|dy|={float((y1-y2).abs().max()):.2e}")

    # 'left' layout: pairwise fold -> order-dependent (NOT invariant)
    hl = VectorLogicHead(input_size=n, num_heads=H, layout="left")
    hl.eval()
    with torch.no_grad():
        z1 = hl(x)
        z2 = hl(x[:, perm])
    check("layout='left' IS order-dependent (why a permutation layer is needed)",
          not torch.allclose(z1, z2, atol=1e-4),
          f"max|dz|={float((z1-z2).abs().max()):.2e}")

    # 'full' + per-concept relevance gates -> heads can differ (real selection)
    hw = VectorLogicHead(input_size=n, num_heads=H, layout="full",
                         use_input_weights=True)
    with torch.no_grad():
        hw.weight_logits.copy_(torch.randn(H, n) * 3.0)  # distinct gates per head
        yw = hw(x)
    head_spread = float(yw.std(dim=1).mean())
    check("layout='full'+use_input_weights gives per-concept gates (heads differ)",
          head_spread > 1e-3, f"mean per-sample head std={head_spread:.3f}")


def main():
    torch.manual_seed(0)
    test_fully_connected_tree()
    test_alternating_tree()
    test_vectorized_routing_collapse()

    n_pass = sum(1 for _, ok, _ in _results if ok)
    n_tot = len(_results)
    print(f"\n{'='*60}\nSUMMARY: {n_pass}/{n_tot} checks passed")
    failed = [name for name, ok, _ in _results if not ok]
    if failed:
        print("FAILED:", failed)
    return 0 if n_pass == n_tot else 1


if __name__ == "__main__":
    raise SystemExit(main())
