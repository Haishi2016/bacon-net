"""Unit tests for GenericFullWeightAggregator (combined full_weight + gl.generic).

Verifies the generic aggregator:
  * reduces EXACTLY to full_weight (lsp_power_mean) at init (static core);
  * spans continuous andness [-1, 2] incl. hard-AND / hard-OR (gl.generic can't);
  * value-based gating: andness/weights respond to input values once trained;
  * conditional gating: output responds to external context c;
  * partial absorption: R starts at identity, can mix inputs, has a reg loss;
  * gradients reach all components (so training can SELECT the mode).

Run directly:
    py -3 tests/unit/test_generic_full_weight.py
"""

from __future__ import annotations

import os
import sys

import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
sys.path.insert(0, _ROOT)

from bacon.aggregators.lsp.full_weight import lsp_power_mean          # noqa: E402
from bacon.aggregators.lsp.generic_full_weight import (               # noqa: E402
    GenericFullWeightAggregator,
)

_results = []


def check(name, cond, detail=""):
    ok = bool(cond)
    _results.append((name, ok))
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  -- {detail}" if detail else ""))
    return ok


def test_reduces_to_full_weight():
    print("\n=== reduces to full_weight at init (static core) ===")
    torch.manual_seed(0)
    N, B = 4, 8
    X = torch.rand(N, B)

    # generic mode: all gates + transform ON, but init -> gates 0, R ~ identity
    agg = GenericFullWeightAggregator(N, weight_mode="generic", init_andness=1.3)
    static = GenericFullWeightAggregator(N, weight_mode="static", init_andness=1.3)
    agg.eval()
    with torch.no_grad():
        out = agg(X)
        out_static = static(X)
        u, a, w = agg.effective_andness_weights(X)
    check("generic starts CLOSE to full_weight (R~=I, gates=0)",
          torch.allclose(out, out_static, atol=5e-2),
          f"max|d|={float((out-out_static).abs().max()):.2e}")
    # gates are zero at init -> andness is (batch-)constant, not value-varying
    check("andness ~constant across batch at init (gates zero)",
          float(a.std()) < 1e-6, f"a.std={float(a.std()):.2e}")
    # R ~ identity at init -> u ~= X
    check("coordinate transform R ~= identity at init (u ~= X)",
          torch.allclose(u, X, atol=2e-2), f"max|u-X|={float((u-X).abs().max()):.2e}")

    # static weight_mode must equal a bare full_weight power mean EXACTLY
    st = GenericFullWeightAggregator(N, weight_mode="static", init_andness=0.9)
    with torch.no_grad():
        _, a_s, w_s = st.effective_andness_weights(X)
        out_s = st(X)
        ref_s = lsp_power_mean(X, a_s, w_s)
    check("weight_mode='static' == full_weight exactly",
          torch.allclose(out_s, ref_s, atol=1e-6),
          f"max|d|={float((out_s-ref_s).abs().max()):.2e}")


def test_continuous_andness_range():
    print("\n=== continuous andness incl. hard-AND / hard-OR ===")
    N, B = 3, 16
    X = torch.rand(N, B)
    # hard-AND (a=2): output <= min of inputs (conjunction); hard-OR (a=-1) >= max
    hard_and = GenericFullWeightAggregator(N, weight_mode="static", init_andness=2.0)
    hard_or = GenericFullWeightAggregator(N, weight_mode="static", init_andness=-1.0)
    neutral = GenericFullWeightAggregator(N, weight_mode="static", init_andness=0.5)
    with torch.no_grad():
        y_and = hard_and(X)
        y_or = hard_or(X)
        y_mid = neutral(X)
        xmin = X.min(0).values
        xmax = X.max(0).values
        xmean = X.mean(0)
    check("hard-AND (a=2) <= min(inputs)+tol", bool((y_and <= xmin + 1e-3).all()),
          f"mean(y_and-min)={float((y_and-xmin).mean()):.3f}")
    check("hard-OR (a=-1) >= max(inputs)-tol", bool((y_or >= xmax - 1e-3).all()),
          f"mean(y_or-max)={float((y_or-xmax).mean()):.3f}")
    check("neutral (a=0.5) == arithmetic mean", torch.allclose(y_mid, xmean, atol=1e-3),
          f"max|y-mean|={float((y_mid-xmean).abs().max()):.2e}")


def test_value_dependent_gating():
    print("\n=== value-based gating (andness responds to inputs) ===")
    torch.manual_seed(1)
    N, B = 4, 32
    agg = GenericFullWeightAggregator(N, weight_mode="value_dependent")
    # force the andness gate to be non-trivial (simulate a trained gate)
    with torch.no_grad():
        for p in agg.andness_gate.parameters():
            p.copy_(torch.randn_like(p))
    # two very different input distributions -> different effective andness
    X_low = torch.rand(N, B) * 0.3
    X_high = 0.7 + torch.rand(N, B) * 0.3
    with torch.no_grad():
        _, a_low, _ = agg.effective_andness_weights(X_low)
        _, a_high, _ = agg.effective_andness_weights(X_high)
    check("value gate makes andness input-dependent (a varies with inputs)",
          float((a_low.mean() - a_high.mean()).abs()) > 1e-2,
          f"a_low={float(a_low.mean()):.3f} a_high={float(a_high.mean()):.3f}")
    check("value-dependent andness is per-sample (shape [batch])",
          a_low.shape == (B,), f"shape={tuple(a_low.shape)}")


def test_conditional_gating():
    print("\n=== conditional gating (output responds to context c) ===")
    torch.manual_seed(2)
    N, B, C = 4, 16, 3
    agg = GenericFullWeightAggregator(N, weight_mode="conditional", context_dim=C)
    with torch.no_grad():
        for p in agg.andness_gate.parameters():
            p.copy_(torch.randn_like(p))
        for p in agg.weight_gate.parameters():
            p.copy_(torch.randn_like(p))
    X = torch.rand(N, B)
    c1 = torch.zeros(B, C)
    c2 = torch.ones(B, C) * 2.0
    with torch.no_grad():
        y1 = agg(X, context=c1)
        y2 = agg(X, context=c2)
    check("different context c changes the output",
          float((y1 - y2).abs().max()) > 1e-3, f"max|dy|={float((y1-y2).abs().max()):.3e}")


def test_partial_absorption_transform():
    print("\n=== partial absorption via coordinate transform R ===")
    N = 3
    agg = GenericFullWeightAggregator(N, weight_mode="generic")
    check("R exists in generic mode", agg.r_logits is not None)
    # identity at init -> reg loss ~ 0
    reg0 = float(agg.transform_regularization())
    check("R reg loss ~ 0 at identity init", reg0 < 1e-4, f"reg={reg0:.2e}")
    # move R off identity -> u mixes inputs, reg loss > 0
    with torch.no_grad():
        agg.r_logits.copy_(torch.randn(N, N) * 3.0)
    X = torch.rand(N, 8)
    with torch.no_grad():
        u, _, _ = agg.effective_andness_weights(X)
    check("R off-identity mixes inputs (u != X)",
          not torch.allclose(u, X, atol=1e-3))
    check("R reg loss > 0 when R != I", float(agg.transform_regularization()) > 1e-4)


def test_gradients_flow():
    print("\n=== gradients reach all components (training can select mode) ===")
    torch.manual_seed(3)
    N, B, C = 4, 8, 2
    agg = GenericFullWeightAggregator(N, weight_mode="generic", context_dim=C)
    X = torch.rand(N, B, requires_grad=True)
    c = torch.rand(B, C)
    out = agg(X, context=c).sum()
    out.backward()
    grads = {
        "andness_logit": agg.andness_logit.grad,
        "weight_logits": agg.weight_logits.grad,
        "r_logits": agg.r_logits.grad,
        "andness_gate": agg.andness_gate[0].weight.grad,
        "weight_gate": agg.weight_gate[0].weight.grad,
    }
    for name, g in grads.items():
        check(f"gradient flows to {name}", g is not None and torch.isfinite(g).all())


def main():
    test_reduces_to_full_weight()
    test_continuous_andness_range()
    test_value_dependent_gating()
    test_conditional_gating()
    test_partial_absorption_transform()
    test_gradients_flow()
    n_pass = sum(1 for _, ok in _results if ok)
    n_tot = len(_results)
    print(f"\n{'='*60}\nSUMMARY: {n_pass}/{n_tot} checks passed")
    return 0 if n_pass == n_tot else 1


if __name__ == "__main__":
    raise SystemExit(main())
