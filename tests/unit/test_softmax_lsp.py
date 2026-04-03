"""
Unit tests for LspSoftmaxAggregator and PerNodeLspSoftmaxAggregator.

Converted from the standalone run_tests_softmax_lsp.py into pytest format.
"""

import pytest
import torch
import numpy as np
from bacon.aggregators.lsp.softmax_lsp import (
    LspSoftmaxAggregator,
    PerNodeLspSoftmaxAggregator,
)


# ---- LspSoftmaxAggregator (andness-based) ----

class TestLspSoftmaxOperatorSelection:
    """Andness parameter selects the correct dominant operator."""

    @pytest.mark.parametrize("andness, expected_idx, name", [
        (1.5, 0, "product"),
        (1.0, 1, "min"),
        (0.5, 2, "avg"),
        (0.0, 3, "max"),
        (-0.5, 4, "prob_sum"),
    ])
    def test_andness_selects_operator(self, andness, expected_idx, name):
        agg = LspSoftmaxAggregator(tau=0.01)
        w = agg.get_weights_for_andness(andness)
        assert np.argmax(w) == expected_idx, (
            f"Expected {name} (index {expected_idx}) at a={andness}, "
            f"got argmax={np.argmax(w)}"
        )
        assert w[expected_idx] > 0.9


class TestLspSoftmaxOutputs:
    """Output correctness and range."""

    def test_output_in_valid_range(self):
        agg = LspSoftmaxAggregator(tau=0.5)
        torch.manual_seed(42)
        for _ in range(100):
            x = torch.rand(10)
            y = torch.rand(10)
            a = torch.rand(1) * 2.0 - 0.5
            out = agg(x, y, a)
            assert (out >= 0.0).all() and (out <= 1.0).all()

    @pytest.mark.parametrize("andness, expected_fn", [
        (1.5, lambda x, y: x * y),
        (1.0, lambda x, y: torch.min(x, y)),
        (0.5, lambda x, y: (x + y) / 2.0),
        (0.0, lambda x, y: torch.max(x, y)),
        (-0.5, lambda x, y: x + y - x * y),
    ])
    def test_extreme_anchors(self, andness, expected_fn):
        agg = LspSoftmaxAggregator(tau=0.01)
        x = torch.tensor([0.3, 0.5, 0.8])
        y = torch.tensor([0.4, 0.6, 0.7])
        a = torch.tensor(andness)
        out = agg(x, y, a)
        expected = expected_fn(x, y)
        torch.testing.assert_close(out, expected, rtol=1e-1, atol=1e-1)

    def test_batch_1d(self):
        agg = LspSoftmaxAggregator(tau=0.5)
        x, y = torch.rand(100), torch.rand(100)
        out = agg(x, y, torch.tensor(0.5))
        assert out.shape == x.shape

    def test_scalar(self):
        agg = LspSoftmaxAggregator(tau=0.5)
        out = agg(torch.tensor(0.5), torch.tensor(0.7), torch.tensor(0.5))
        assert out.shape == torch.Size([])


class TestLspSoftmaxPruning:
    def test_prune_y_returns_x(self):
        agg = LspSoftmaxAggregator(tau=0.5)
        x = torch.tensor([0.3, 0.5, 0.8])
        y = torch.tensor([0.7, 0.2, 0.4])
        out = agg.forward(x, y, torch.tensor(0.5), torch.tensor(1.0), torch.tensor(0.0))
        torch.testing.assert_close(out, x, rtol=1e-5, atol=1e-5)

    def test_prune_x_returns_y(self):
        agg = LspSoftmaxAggregator(tau=0.5)
        x = torch.tensor([0.3, 0.5, 0.8])
        y = torch.tensor([0.7, 0.2, 0.4])
        out = agg.forward(x, y, torch.tensor(0.5), torch.tensor(0.0), torch.tensor(1.0))
        torch.testing.assert_close(out, y, rtol=1e-5, atol=1e-5)


class TestLspSoftmaxTau:
    def test_tau_update_sharpens(self):
        agg = LspSoftmaxAggregator(tau=1.0)
        w1 = agg.get_weights_for_andness(1.5)
        agg.set_tau(0.1)
        w2 = agg.get_weights_for_andness(1.5)
        assert w2[0] > w1[0]

    def test_tau_minimum_clamp(self):
        agg = LspSoftmaxAggregator(tau=0.0)
        assert agg.tau >= 1e-4
        agg.set_tau(-1.0)
        assert agg.tau >= 1e-4

    def test_exponential_annealing(self):
        agg = LspSoftmaxAggregator(tau=1.0)
        agg.anneal_tau(0, 10, final_tau=0.01, schedule="exponential")
        assert abs(agg.tau - 1.0) < 0.01
        agg.anneal_tau(9, 10, final_tau=0.01, schedule="exponential")
        assert abs(agg.tau - 0.01) < 0.01

    def test_linear_annealing(self):
        agg = LspSoftmaxAggregator(tau=1.0)
        agg.anneal_tau(5, 10, final_tau=0.0, schedule="linear")
        assert 0.4 < agg.tau < 0.6


class TestLspSoftmaxEntropy:
    def test_entropy_at_center_lower(self):
        sharp = LspSoftmaxAggregator(tau=0.1)
        wide = LspSoftmaxAggregator(tau=0.5)
        assert wide.entropy(a=0.75) > sharp.entropy(a=2.0)

    def test_entropy_loss_gradient(self):
        agg = LspSoftmaxAggregator(tau=0.5)
        a = torch.tensor(1.5, requires_grad=True)
        loss = agg.entropy_loss(a)
        loss.backward()
        assert a.grad is not None


class TestLspSoftmaxMisc:
    def test_gradient_flow(self):
        agg = LspSoftmaxAggregator(tau=0.5)
        x = torch.rand(10, requires_grad=True)
        y = torch.rand(10, requires_grad=True)
        a = torch.tensor(0.5, requires_grad=True)
        out = agg(x, y, a)
        out.sum().backward()
        assert x.grad is not None and y.grad is not None and a.grad is not None

    def test_describe(self):
        agg = LspSoftmaxAggregator(tau=0.5)
        info = agg.describe(a=0.5)
        assert info["tau"] == 0.5
        assert len(info["weights"]) == 5

    def test_custom_centers(self):
        custom = [1.25, 1.0, 0.5, 0.0, -0.25]
        agg = LspSoftmaxAggregator(tau=0.01, centers=custom)
        assert np.argmax(agg.get_weights_for_andness(1.25)) == 0
        assert np.argmax(agg.get_weights_for_andness(-0.25)) == 4
        assert agg.describe(a=0.5)["centers"] == custom

    def test_weights_smooth_transition(self):
        agg = LspSoftmaxAggregator(tau=0.5)
        w = agg.get_weights_for_andness(1.5)
        assert w[0] > 0.1 and w[1] > 0.1


# ---- PerNodeLspSoftmaxAggregator ----

class TestPerNodeLspSoftmax:
    def test_attach_to_tree(self):
        agg = PerNodeLspSoftmaxAggregator(tau=0.5)
        agg.attach_to_tree(num_layers=7)
        assert agg.num_layers == 7
        assert len(agg.op_logits_per_node) == 7
        for logits in agg.op_logits_per_node:
            assert logits.shape == torch.Size([5])

    def test_pointer_reset(self):
        agg = PerNodeLspSoftmaxAggregator(tau=0.5)
        agg.attach_to_tree(num_layers=5)
        x, y = torch.rand(10), torch.rand(10)
        a = torch.tensor(0.5)
        agg.aggregate(x, y, a, torch.tensor(0.5), torch.tensor(0.5))
        agg.aggregate(x, y, a, torch.tensor(0.5), torch.tensor(0.5))
        assert agg._node_ptr == 2
        agg.start_forward()
        assert agg._node_ptr == 0

    def test_error_without_attach(self):
        agg = PerNodeLspSoftmaxAggregator(tau=0.5)
        with pytest.raises(RuntimeError):
            agg.aggregate(
                torch.rand(10), torch.rand(10),
                torch.tensor(0.5), torch.tensor(0.5), torch.tensor(0.5),
            )
