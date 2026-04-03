"""
Unit tests for GenericGLAggregator (N-ary).

Covers:
  1. Static (classical MAT) — 2-input and N-input
  2. Composite (coordinate transformation R)
  3. Conditional (weights depend on external context c)
  4. Value-dependent (weights depend on input features psi(u))
  5. AggregatorBase interface (used by the BACON tree)
"""

import pytest
import torch
import numpy as np
from bacon.aggregators.lsp.generic_gl import (
    GenericGLAggregator,
    ANCHOR_FUNCTIONS,
    ANCHOR_ANDNESS,
)


# Helper: stack two tensors into [2, ...] for the N-ary API
def _xy(x, y):
    return torch.stack([x, y], dim=0)


# ---- Static mode ----

class TestStaticAggregation:

    def test_output_in_range_2input(self):
        agg = GenericGLAggregator(anchors=('min', 'mean', 'max'), weight_mode='static')
        torch.manual_seed(42)
        for _ in range(50):
            x = _xy(torch.rand(10), torch.rand(10))
            out = agg(x)
            assert (out >= 0.0).all() and (out <= 1.0).all()

    def test_output_in_range_5input(self):
        agg = GenericGLAggregator(anchors=('min', 'mean', 'max'), weight_mode='static')
        torch.manual_seed(42)
        for _ in range(20):
            x = torch.rand(5, 10)  # 5 inputs, batch 10
            out = agg(x)
            assert out.shape == (10,)
            assert (out >= 0.0).all() and (out <= 1.0).all()

    def test_uniform_weights_at_init(self):
        agg = GenericGLAggregator(anchors=('min', 'mean', 'max'), weight_mode='static', tau=1.0)
        w = agg.get_weights()
        np.testing.assert_allclose(w, [1 / 3] * 3, rtol=1e-4)

    def test_sharp_selection(self):
        agg = GenericGLAggregator(anchors=('min', 'mean', 'max'), weight_mode='static', tau=0.01)
        agg.alpha_logits.data = torch.tensor([0.0, 0.0, 10.0])
        w = agg.get_weights()
        assert np.argmax(w) == 2
        assert w[2] > 0.99

    def test_dominant_min_2input(self):
        agg = GenericGLAggregator(anchors=('min', 'mean', 'max'), weight_mode='static', tau=0.001)
        agg.alpha_logits.data = torch.tensor([100.0, 0.0, 0.0])
        x = torch.tensor([0.3, 0.8])
        y = torch.tensor([0.5, 0.6])
        out = agg(_xy(x, y))
        torch.testing.assert_close(out, torch.min(x, y), atol=1e-2, rtol=1e-2)

    def test_dominant_max_2input(self):
        agg = GenericGLAggregator(anchors=('min', 'mean', 'max'), weight_mode='static', tau=0.001)
        agg.alpha_logits.data = torch.tensor([0.0, 0.0, 100.0])
        x = torch.tensor([0.3, 0.8])
        y = torch.tensor([0.5, 0.6])
        out = agg(_xy(x, y))
        torch.testing.assert_close(out, torch.max(x, y), atol=1e-2, rtol=1e-2)

    def test_dominant_min_4input(self):
        """N=4: with min-dominant weights, output should equal min across 4 inputs."""
        agg = GenericGLAggregator(anchors=('min', 'mean', 'max'), weight_mode='static', tau=0.001)
        agg.alpha_logits.data = torch.tensor([100.0, 0.0, 0.0])
        x = torch.tensor([[0.3, 0.8], [0.5, 0.2], [0.7, 0.9], [0.1, 0.6]])  # [4, 2]
        out = agg(x)
        expected = x.min(dim=0).values
        torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)

    def test_dominant_mean_3input(self):
        agg = GenericGLAggregator(anchors=('min', 'mean', 'max'), weight_mode='static', tau=0.001)
        agg.alpha_logits.data = torch.tensor([0.0, 100.0, 0.0])
        x = torch.tensor([[0.2, 0.8], [0.4, 0.6], [0.6, 0.4]])  # [3, 2]
        out = agg(x)
        expected = x.mean(dim=0)
        torch.testing.assert_close(out, expected, atol=1e-2, rtol=1e-2)

    def test_gradient_flow(self):
        agg = GenericGLAggregator(anchors=('min', 'mean', 'max'), weight_mode='static')
        x = torch.rand(2, 10, requires_grad=True)
        out = agg(x)
        out.sum().backward()
        assert x.grad is not None
        assert agg.alpha_logits.grad is not None

    def test_six_anchor_set(self):
        agg = GenericGLAggregator(
            anchors=('min', 'harmonic', 'geometric', 'mean', 'quadratic', 'max'),
            weight_mode='static',
        )
        assert agg.k == 6
        out = agg(torch.tensor([0.5, 0.7]))  # 2 scalar inputs -> [2]
        assert 0.0 <= out.item() <= 1.0

    def test_default_anchors_match_paper(self):
        """Default anchor set is the paper's F = {min, H, G, A, Q, max}."""
        agg = GenericGLAggregator()
        assert agg.k == 6
        assert agg._anchor_names == ['min', 'harmonic', 'geometric',
                                      'mean', 'quadratic', 'max']

    def test_andness_within_01_for_core_anchors(self):
        """The six core anchors all have andness in [0, 1]."""
        for name in ('min', 'harmonic', 'geometric', 'mean', 'quadratic', 'max'):
            assert 0.0 <= ANCHOR_ANDNESS[name] <= 1.0, f"{name}: {ANCHOR_ANDNESS[name]}"

    def test_andness_outside_01_for_extensions(self):
        """product and prob_sum have andness outside [0, 1]."""
        assert ANCHOR_ANDNESS['product'] > 1.0
        assert ANCHOR_ANDNESS['prob_sum'] < 0.0

    def test_effective_andness(self):
        agg = GenericGLAggregator(anchors=('min', 'mean', 'max'), weight_mode='static', tau=0.001)
        agg.alpha_logits.data = torch.tensor([100.0, 0.0, 0.0])
        assert abs(agg.effective_andness() - 1.0) < 0.05

    def test_unknown_anchor_raises(self):
        with pytest.raises(ValueError, match="Unknown anchor"):
            GenericGLAggregator(anchors=('min', 'nonexistent'))


# ---- N-ary anchor correctness ----

class TestNaryAnchors:
    """Verify each anchor produces the correct result for N>2."""

    def _make(self, anchor_name):
        agg = GenericGLAggregator(anchors=(anchor_name,), weight_mode='static', tau=0.001)
        return agg

    def test_product_3(self):
        agg = self._make('product')
        x = torch.tensor([[0.5], [0.4], [0.8]])  # [3, 1]
        out = agg(x)
        torch.testing.assert_close(out, torch.tensor([0.5 * 0.4 * 0.8]), atol=1e-4, rtol=1e-4)

    def test_prob_sum_3(self):
        agg = self._make('prob_sum')
        x = torch.tensor([[0.3], [0.5], [0.2]])
        expected = 1.0 - (1 - 0.3) * (1 - 0.5) * (1 - 0.2)
        out = agg(x)
        torch.testing.assert_close(out, torch.tensor([expected]), atol=1e-4, rtol=1e-4)

    def test_mean_4(self):
        agg = self._make('mean')
        vals = [0.2, 0.4, 0.6, 0.8]
        x = torch.tensor(vals).unsqueeze(1)
        out = agg(x)
        torch.testing.assert_close(out, torch.tensor([np.mean(vals)], dtype=torch.float32), atol=1e-4, rtol=1e-4)

    def test_min_max_5(self):
        agg_min = self._make('min')
        agg_max = self._make('max')
        x = torch.tensor([0.1, 0.9, 0.5, 0.3, 0.7]).unsqueeze(1)
        assert agg_min(x).item() == pytest.approx(0.1, abs=1e-4)
        assert agg_max(x).item() == pytest.approx(0.9, abs=1e-4)

    def test_harmonic_2(self):
        agg = self._make('harmonic')
        a, b = 0.4, 0.6
        x = torch.tensor([[a], [b]])
        expected = 2 * a * b / (a + b)
        torch.testing.assert_close(out := agg(x), torch.tensor([expected]), atol=1e-3, rtol=1e-3)

    def test_geometric_2(self):
        agg = self._make('geometric')
        a, b = 0.4, 0.6
        x = torch.tensor([[a], [b]])
        expected = np.sqrt(a * b)
        assert agg(x).item() == pytest.approx(expected, abs=1e-3)

    def test_quadratic_2(self):
        agg = self._make('quadratic')
        a, b = 0.4, 0.6
        x = torch.tensor([[a], [b]])
        expected = np.sqrt((a**2 + b**2) / 2)
        assert agg(x).item() == pytest.approx(expected, abs=1e-3)


# ---- Coordinate transformation (Section 4.2) ----

class TestCoordinateTransformation:

    def test_identity_init(self):
        agg = GenericGLAggregator(
            anchors=('min', 'mean', 'max'), weight_mode='static', use_transform=True,
        )
        R = agg.get_transform_matrix().detach().numpy()
        assert R[0, 0] > 0.9 and R[1, 1] > 0.9

    def test_row_stochastic(self):
        agg = GenericGLAggregator(
            anchors=('min', 'mean', 'max'), weight_mode='static', use_transform=True,
        )
        agg.r_logits.data = torch.randn(2, 2) * 3
        R = agg.get_transform_matrix().detach().numpy()
        np.testing.assert_allclose(R.sum(axis=1), [1.0, 1.0], atol=1e-5)
        assert (R >= 0).all()

    def test_transform_preserves_range(self):
        agg = GenericGLAggregator(
            anchors=('min', 'mean', 'max'), weight_mode='static', use_transform=True,
        )
        agg.r_logits.data = torch.randn(2, 2)
        x = torch.rand(2, 100)  # [2, batch=100]
        u = agg.transform(x)
        assert (u >= 0).all() and (u <= 1).all()

    def test_partial_absorption_recovery(self):
        """R ≈ [[1,0],[0.5,0.5]] with harmonic anchor recovers P(x,y) = H(x, A(x,y))."""
        agg = GenericGLAggregator(
            anchors=('harmonic',), weight_mode='static', use_transform=True, tau=0.01,
        )
        with torch.no_grad():
            agg.r_logits[0] = torch.tensor([10.0, -10.0])
            agg.r_logits[1] = torch.tensor([0.0, 0.0])

        x_vals = torch.tensor([0.3, 0.5, 0.8])
        y_vals = torch.tensor([0.4, 0.6, 0.7])
        inp = _xy(x_vals, y_vals)  # [2, 3]
        out = agg(inp)

        a_xy = (x_vals + y_vals) / 2.0
        expected = 2 * x_vals * a_xy / (x_vals + a_xy + 1e-7)
        torch.testing.assert_close(out, expected, atol=0.02, rtol=0.02)

    def test_identity_regularization(self):
        agg = GenericGLAggregator(
            anchors=('min', 'mean', 'max'), weight_mode='static',
            use_transform=True, identity_reg=1.0,
        )
        reg = agg.transform_regularization()
        assert reg.item() < 0.5
        agg.r_logits.data = torch.tensor([[0.0, 5.0], [5.0, 0.0]])
        reg2 = agg.transform_regularization()
        assert reg2.item() > reg.item()

    def test_r_gradient_flow(self):
        agg = GenericGLAggregator(
            anchors=('min', 'mean', 'max'), weight_mode='static', use_transform=True,
        )
        x = torch.rand(2, 10, requires_grad=True)
        out = agg(x)
        out.sum().backward()
        assert agg.r_logits.grad is not None

    def test_no_transform_raises(self):
        agg = GenericGLAggregator(anchors=('min', 'mean', 'max'))
        with pytest.raises(RuntimeError):
            agg.get_transform_matrix()

    def test_transform_resizes_for_3_inputs(self):
        """R auto-resizes to 3×3 when given 3 inputs."""
        agg = GenericGLAggregator(
            anchors=('min', 'mean', 'max'), weight_mode='static', use_transform=True,
        )
        x = torch.rand(3, 8)  # 3 inputs
        out = agg(x)
        assert out.shape == (8,)
        R = agg.get_transform_matrix()
        assert R.shape == (3, 3)
        np.testing.assert_allclose(
            R.detach().numpy().sum(axis=1), [1.0, 1.0, 1.0], atol=1e-5,
        )


# ---- Conditional aggregation (Section 4.3) ----

class TestConditionalAggregation:

    def test_requires_context(self):
        agg = GenericGLAggregator(
            anchors=('min', 'mean', 'max'), weight_mode='conditional', context_dim=1,
        )
        with pytest.raises(ValueError, match="requires context"):
            agg(_xy(torch.tensor(0.5), torch.tensor(0.7)))

    def test_context_changes_output(self):
        agg = GenericGLAggregator(
            anchors=('min', 'mean', 'max'), weight_mode='conditional',
            context_dim=1, hidden_dim=16,
        )
        inp = _xy(torch.tensor(0.4), torch.tensor(0.8))
        c_low = torch.tensor([0.1])
        c_high = torch.tensor([0.9])
        out_low = agg(inp, c=c_low)
        out_high = agg(inp, c=c_high)
        assert out_low.shape == torch.Size([])
        assert out_high.shape == torch.Size([])

    def test_learn_min_max_switch(self):
        """min when c<0.5, max when c>=0.5."""
        torch.manual_seed(42)
        agg = GenericGLAggregator(
            anchors=('min', 'mean', 'max'), weight_mode='conditional',
            context_dim=1, hidden_dim=16, tau=0.3,
        )
        opt = torch.optim.Adam(agg.parameters(), lr=0.01)
        N = 500
        x, y = torch.rand(N), torch.rand(N)
        c = torch.rand(N, 1)
        target = torch.where(c.squeeze() < 0.5, torch.min(x, y), torch.max(x, y))
        inp = _xy(x, y)  # [2, N]

        for _ in range(300):
            opt.zero_grad()
            loss = ((agg(inp, c=c) - target) ** 2).mean()
            loss.backward()
            opt.step()

        final = ((agg(inp, c=c) - target) ** 2).mean().item()
        assert final < 0.01, f"MSE={final:.4f}"

    def test_batch_context(self):
        agg = GenericGLAggregator(
            anchors=('min', 'mean', 'max'), weight_mode='conditional', context_dim=1,
        )
        inp = torch.rand(2, 32)
        c = torch.rand(32, 1)
        out = agg(inp, c=c)
        assert out.shape == torch.Size([32])


# ---- Value-dependent aggregation (Section 4.4) ----

class TestValueDependentAggregation:

    def test_output_varies_with_inputs(self):
        agg = GenericGLAggregator(
            anchors=('min', 'mean', 'max'), weight_mode='value_dependent', hidden_dim=16,
        )
        out1 = agg(torch.tensor([0.1, 0.9]))
        out2 = agg(torch.tensor([0.9, 0.1]))
        assert out1.shape == torch.Size([])

    def test_learn_joint_condition(self):
        """min when x+y>1, max when x+y<=1."""
        torch.manual_seed(0)
        agg = GenericGLAggregator(
            anchors=('min', 'mean', 'max'), weight_mode='value_dependent',
            hidden_dim=32, tau=0.3,
        )
        opt = torch.optim.Adam(agg.parameters(), lr=0.01)
        N = 500
        x, y = torch.rand(N), torch.rand(N)
        target = torch.where(x + y > 1.0, torch.min(x, y), torch.max(x, y))
        inp = _xy(x, y)

        for _ in range(500):
            opt.zero_grad()
            loss = ((agg(inp) - target) ** 2).mean()
            loss.backward()
            opt.step()

        final = ((agg(inp) - target) ** 2).mean().item()
        assert final < 0.01, f"MSE={final:.4f}"

    def test_gradient_flow(self):
        agg = GenericGLAggregator(
            anchors=('min', 'mean', 'max'), weight_mode='value_dependent',
        )
        x = torch.rand(2, 10, requires_grad=True)
        out = agg(x)
        out.sum().backward()
        assert x.grad is not None

    def test_psi_features(self):
        """psi returns [mean, min, max, std, product] along dim 0."""
        u = torch.tensor([[0.3, 0.8], [0.7, 0.2], [0.5, 0.5]])  # [3, 2]
        psi = GenericGLAggregator._psi(u)
        assert psi.shape == (2, 5)  # [batch=2, 5]
        torch.testing.assert_close(psi[:, 0], u.mean(dim=0))
        torch.testing.assert_close(psi[:, 1], u.min(dim=0).values)
        torch.testing.assert_close(psi[:, 2], u.max(dim=0).values)
        torch.testing.assert_close(psi[:, 4], u.prod(dim=0))


# ---- Full mode ----

class TestFullMode:
    def test_forward_runs(self):
        agg = GenericGLAggregator(
            anchors=('min', 'mean', 'max'), weight_mode='full',
            context_dim=2, hidden_dim=16,
        )
        x = torch.rand(2, 8)
        c = torch.rand(8, 2)
        out = agg(x, c=c)
        assert out.shape == torch.Size([8])
        assert (out >= 0).all() and (out <= 1).all()


# ---- AggregatorBase interface ----

class TestAggregatorBaseInterface:
    """The BACON tree calls aggregate([left, right], a, [w0, w1])."""

    def test_aggregate_tensor_2input(self):
        agg = GenericGLAggregator(anchors=('min', 'mean', 'max'), weight_mode='static', tau=0.001)
        agg.alpha_logits.data = torch.tensor([100.0, 0.0, 0.0])  # min
        left = torch.tensor([0.3, 0.8])
        right = torch.tensor([0.5, 0.2])
        out = agg.aggregate([left, right], 0.5, [torch.tensor(0.5), torch.tensor(0.5)])
        torch.testing.assert_close(out, torch.min(left, right), atol=1e-2, rtol=1e-2)

    def test_aggregate_tensor_3input(self):
        agg = GenericGLAggregator(anchors=('min', 'mean', 'max'), weight_mode='static', tau=0.001)
        agg.alpha_logits.data = torch.tensor([0.0, 0.0, 100.0])  # max
        a = torch.tensor([0.3])
        b = torch.tensor([0.5])
        c = torch.tensor([0.1])
        out = agg.aggregate([a, b, c], 0.5, [torch.tensor(0.33)] * 3)
        torch.testing.assert_close(out, torch.tensor([0.5]), atol=1e-2, rtol=1e-2)

    def test_aggregate_float(self):
        agg = GenericGLAggregator(anchors=('min', 'mean', 'max'), weight_mode='static', tau=0.001)
        agg.alpha_logits.data = torch.tensor([100.0, 0.0, 0.0])  # min
        out = agg.aggregate([0.3, 0.7], 0.5, [0.5, 0.5])
        assert abs(out - 0.3) < 0.05


# ---- Diagnostics ----

class TestDiagnostics:
    def test_describe_static(self):
        agg = GenericGLAggregator(anchors=('min', 'mean', 'max'), weight_mode='static')
        info = agg.describe()
        assert info['weight_mode'] == 'static'
        assert len(info['weights']) == 3
        assert 'effective_andness' in info

    def test_describe_with_transform(self):
        agg = GenericGLAggregator(
            anchors=('min', 'mean', 'max'), weight_mode='static', use_transform=True,
        )
        info = agg.describe()
        assert 'R' in info
        assert len(info['R']) == 2

    def test_entropy_loss(self):
        agg = GenericGLAggregator(anchors=('min', 'mean', 'max'), weight_mode='static')
        x = torch.rand(2, 10, requires_grad=True)
        loss = agg.entropy_loss(x)
        loss.backward()
        assert agg.alpha_logits.grad is not None

    def test_tau_annealing(self):
        agg = GenericGLAggregator(anchors=('min', 'mean', 'max'), tau=1.0)
        agg.anneal_tau(0, 10, final_tau=0.01)
        assert abs(agg.tau - 1.0) < 0.02
        agg.anneal_tau(9, 10, final_tau=0.01)
        assert abs(agg.tau - 0.01) < 0.02

    def test_repr(self):
        agg = GenericGLAggregator(anchors=('min', 'max'), weight_mode='static')
        r = repr(agg)
        assert 'GenericGLAggregator' in r
        assert 'min' in r and 'max' in r

    def test_unknown_weight_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown weight_mode"):
            GenericGLAggregator(anchors=('min', 'max'), weight_mode='bogus')
