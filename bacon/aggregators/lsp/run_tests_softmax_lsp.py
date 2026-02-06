"""
Standalone tests for LspSoftmaxAggregator (no pytest required)

Run with: python run_tests_softmax_lsp.py

The LspSoftmaxAggregator uses the tree's 'a' (andness) parameter to select
which operator to use, with softmax-weighted mixing based on distance from
canonical centers (default evenly-spaced):
  - a = 1.5: product (x*y) - pure t-norm
  - a = 1.0: min
  - a = 0.5: avg
  - a = 0.0: max
  - a = -0.5: prob_sum (x+y-xy) - pure t-conorm
"""

import sys
sys.path.insert(0, '.')

import torch
import numpy as np
from bacon.aggregators.lsp.softmax_lsp import LspSoftmaxAggregator, PerNodeLspSoftmaxAggregator


def test_andness_selects_product():
    """With a=1.5 (product center), product operator should dominate."""
    agg = LspSoftmaxAggregator(tau=0.01)
    w = agg.get_weights_for_andness(1.5)
    
    # Product (index 0) should have highest weight at a=1.5
    assert np.argmax(w) == 0, f"Expected product (index 0) at a=1.5, got argmax={np.argmax(w)}"
    assert w[0] > 0.9, f"Expected product weight > 0.9, got {w[0]}"
    print("✓ test_andness_selects_product passed")


def test_andness_selects_min():
    """With a=1.0 (min center), min operator should dominate."""
    agg = LspSoftmaxAggregator(tau=0.01)
    w = agg.get_weights_for_andness(1.0)
    
    # Min (index 1) should have highest weight at a=1.0
    assert np.argmax(w) == 1, f"Expected min (index 1) at a=1.0, got argmax={np.argmax(w)}"
    assert w[1] > 0.9, f"Expected min weight > 0.9, got {w[1]}"
    print("✓ test_andness_selects_min passed")


def test_andness_selects_avg():
    """With a=0.5 (avg center), avg operator should dominate."""
    agg = LspSoftmaxAggregator(tau=0.01)
    w = agg.get_weights_for_andness(0.5)
    
    # Avg (index 2) should have highest weight at a=0.5
    assert np.argmax(w) == 2, f"Expected avg (index 2) at a=0.5, got argmax={np.argmax(w)}"
    assert w[2] > 0.9, f"Expected avg weight > 0.9, got {w[2]}"
    print("✓ test_andness_selects_avg passed")


def test_andness_selects_max():
    """With a=0.0 (max center), max operator should dominate."""
    agg = LspSoftmaxAggregator(tau=0.01)
    w = agg.get_weights_for_andness(0.0)
    
    # Max (index 3) should have highest weight at a=0.0
    assert np.argmax(w) == 3, f"Expected max (index 3) at a=0.0, got argmax={np.argmax(w)}"
    assert w[3] > 0.9, f"Expected max weight > 0.9, got {w[3]}"
    print("✓ test_andness_selects_max passed")


def test_andness_selects_probsum():
    """With a=-0.5 (prob_sum center), prob_sum operator should dominate."""
    agg = LspSoftmaxAggregator(tau=0.01)
    w = agg.get_weights_for_andness(-0.5)
    
    # Prob sum (index 4) should have highest weight at a=-0.5
    assert np.argmax(w) == 4, f"Expected prob_sum (index 4) at a=-0.5, got argmax={np.argmax(w)}"
    assert w[4] > 0.9, f"Expected prob_sum weight > 0.9, got {w[4]}"
    print("✓ test_andness_selects_probsum passed")


def test_output_in_valid_range():
    """Output should be in [0, 1] for random inputs in [0, 1]."""
    agg = LspSoftmaxAggregator(tau=0.5)
    
    torch.manual_seed(42)
    for _ in range(100):
        x = torch.rand(10)
        y = torch.rand(10)
        a = torch.rand(1) * 2.0 - 0.5  # Random a in [-0.5, 1.5]
        out = agg(x, y, a)
        
        assert (out >= 0.0).all(), f"Output below 0: {out.min()}"
        assert (out <= 1.0).all(), f"Output above 1: {out.max()}"
    print("✓ test_output_in_valid_range passed")


def test_extreme_anchors():
    """Test that extreme andness values produce expected operator outputs."""
    agg = LspSoftmaxAggregator(tau=0.01)  # Very sharp selection
    
    x = torch.tensor([0.3, 0.5, 0.8])
    y = torch.tensor([0.4, 0.6, 0.7])
    
    tests = [
        (1.5, lambda x, y: x * y, "product (a=1.5)"),
        (1.0, lambda x, y: torch.min(x, y), "min (a=1.0)"),
        (0.5, lambda x, y: (x + y) / 2.0, "avg (a=0.5)"),
        (0.0, lambda x, y: torch.max(x, y), "max (a=0.0)"),
        (-0.5, lambda x, y: x + y - x * y, "prob_sum (a=-0.5)"),
    ]
    
    for andness, expected_fn, name in tests:
        a = torch.tensor(andness)
        out = agg(x, y, a)
        expected = expected_fn(x, y)
        
        assert torch.allclose(out, expected, rtol=1e-1, atol=1e-1), \
            f"Failed for {name}: expected {expected}, got {out}"
    
    print("✓ test_extreme_anchors passed (all 5 operators)")


def test_tau_update():
    """Test that tau can be updated and affects weight sharpness."""
    agg = LspSoftmaxAggregator(tau=1.0)
    
    # At a=1.5, product should be selected
    w1 = agg.get_weights_for_andness(1.5)
    
    agg.set_tau(0.1)
    w2 = agg.get_weights_for_andness(1.5)
    
    # With lower tau, distribution should be sharper (more peaked at product)
    assert w2[0] > w1[0], "Lower tau should make distribution sharper"
    print("✓ test_tau_update passed")


def test_tau_minimum_clamp():
    """Test that tau is clamped to minimum value."""
    agg = LspSoftmaxAggregator(tau=0.0)
    assert agg.tau >= 1e-4, f"tau should be >= 1e-4, got {agg.tau}"
    
    agg.set_tau(-1.0)
    assert agg.tau >= 1e-4, f"tau should be >= 1e-4, got {agg.tau}"
    print("✓ test_tau_minimum_clamp passed")


def test_describe():
    """Test describe() returns correct info."""
    agg = LspSoftmaxAggregator(tau=0.5)
    info = agg.describe(a=0.5)
    
    assert "tau" in info
    assert "weights" in info
    assert "operators" in info
    assert "andness" in info
    assert "entropy" in info
    assert "dominant_op" in info
    
    assert info["tau"] == 0.5
    assert info["andness"] == 0.5
    assert len(info["weights"]) == 5
    assert len(info["operators"]) == 5
    print("✓ test_describe passed")


def test_batch_support():
    """Test that batched inputs work correctly."""
    agg = LspSoftmaxAggregator(tau=0.5)
    
    # 1D batch
    x = torch.rand(100)
    y = torch.rand(100)
    a = torch.tensor(0.5)
    out = agg(x, y, a)
    assert out.shape == x.shape, f"Shape mismatch: expected {x.shape}, got {out.shape}"
    
    # Scalar
    x = torch.tensor(0.5)
    y = torch.tensor(0.7)
    a = torch.tensor(0.5)
    out = agg(x, y, a)
    assert out.shape == torch.Size([]), f"Shape mismatch: expected [], got {out.shape}"
    print("✓ test_batch_support passed")


def test_pruning_support():
    """Test that pruning weights (w0, w1) work correctly."""
    agg = LspSoftmaxAggregator(tau=0.5)
    
    x = torch.tensor([0.3, 0.5, 0.8])
    y = torch.tensor([0.7, 0.2, 0.4])
    a = torch.tensor(0.5)
    
    # Prune y (w0=1, w1=0): should return x
    out_x_only = agg.forward(x, y, a, torch.tensor(1.0), torch.tensor(0.0))
    torch.testing.assert_close(out_x_only, x, rtol=1e-5, atol=1e-5)
    
    # Prune x (w0=0, w1=1): should return y
    out_y_only = agg.forward(x, y, a, torch.tensor(0.0), torch.tensor(1.0))
    torch.testing.assert_close(out_y_only, y, rtol=1e-5, atol=1e-5)
    
    print("✓ test_pruning_support passed")


def test_aggregator_base_interface():
    """Test compatibility with AggregatorBase interface."""
    agg = LspSoftmaxAggregator(tau=0.5)
    
    x = torch.tensor([0.3, 0.5])
    y = torch.tensor([0.4, 0.6])
    a = torch.tensor(0.5)
    w0 = torch.tensor([0.5, 0.5])
    w1 = torch.tensor([0.5, 0.5])
    
    out = agg.aggregate_tensor(x, y, a, w0, w1)
    assert out.shape == x.shape
    
    out_float = agg.aggregate_float(0.3, 0.4, 0.5, 0.5, 0.5)
    assert isinstance(out_float, float)
    assert 0.0 <= out_float <= 1.0
    print("✓ test_aggregator_base_interface passed")


def test_gradient_flow():
    """Test that gradients flow through the aggregator."""
    agg = LspSoftmaxAggregator(tau=0.5)
    
    x = torch.rand(10, requires_grad=True)
    y = torch.rand(10, requires_grad=True)
    a = torch.tensor(0.5, requires_grad=True)
    
    out = agg(x, y, a)
    loss = out.sum()
    loss.backward()
    
    assert x.grad is not None, "x.grad is None"
    assert y.grad is not None, "y.grad is None"
    assert a.grad is not None, "a.grad is None"
    print("✓ test_gradient_flow passed")


def test_per_node_attach_to_tree():
    """Test tree attachment creates correct number of logits."""
    agg = PerNodeLspSoftmaxAggregator(tau=0.5)
    agg.attach_to_tree(num_layers=7)
    
    assert agg.num_layers == 7
    assert len(agg.op_logits_per_node) == 7
    
    for logits in agg.op_logits_per_node:
        assert logits.shape == torch.Size([5])
    print("✓ test_per_node_attach_to_tree passed")


def test_per_node_pointer_reset():
    """Test that start_forward() resets node pointer."""
    agg = PerNodeLspSoftmaxAggregator(tau=0.5)
    agg.attach_to_tree(num_layers=5)
    
    x = torch.rand(10)
    y = torch.rand(10)
    a = torch.tensor(0.5)
    agg.aggregate(x, y, a, torch.tensor(0.5), torch.tensor(0.5))
    agg.aggregate(x, y, a, torch.tensor(0.5), torch.tensor(0.5))
    
    assert agg._node_ptr == 2
    
    agg.start_forward()
    assert agg._node_ptr == 0
    print("✓ test_per_node_pointer_reset passed")


def test_per_node_error_without_attach():
    """Test that aggregate() raises error if attach_to_tree() not called."""
    agg = PerNodeLspSoftmaxAggregator(tau=0.5)
    
    x = torch.rand(10)
    y = torch.rand(10)
    a = torch.tensor(0.5)
    
    try:
        agg.aggregate(x, y, a, torch.tensor(0.5), torch.tensor(0.5))
        assert False, "Should have raised RuntimeError"
    except RuntimeError:
        pass
    print("✓ test_per_node_error_without_attach passed")


def test_weights_smooth_transition():
    """Test that weights transition smoothly between operator centers."""
    agg = LspSoftmaxAggregator(tau=0.5)
    
    # Test intermediate values between centers
    w_1_5 = agg.get_weights_for_andness(1.5)  # Between product (2.0) and min (1.0)
    
    # Both product and min should have significant weight
    assert w_1_5[0] > 0.1, f"Expected product weight > 0.1 at a=1.5, got {w_1_5[0]}"
    assert w_1_5[1] > 0.1, f"Expected min weight > 0.1 at a=1.5, got {w_1_5[1]}"
    
    # Others should have small weight
    assert w_1_5[2] < w_1_5[0], "Avg should have less weight than product at a=1.5"
    assert w_1_5[3] < w_1_5[1], "Max should have less weight than min at a=1.5"
    
    print("✓ test_weights_smooth_transition passed")


def test_entropy():
    """Test entropy calculation for different andness values."""
    agg = LspSoftmaxAggregator(tau=0.1)
    
    # At a canonical center, entropy should be low (sharp distribution)
    entropy_at_center = agg.entropy(a=2.0)
    
    # At a point between centers, entropy should be higher
    agg_wide = LspSoftmaxAggregator(tau=0.5)
    entropy_between = agg_wide.entropy(a=0.75)  # Between min and avg
    
    # With wider tau, entropy should be higher
    assert entropy_between > entropy_at_center, \
        f"Expected entropy between centers ({entropy_between}) > at center ({entropy_at_center})"
    
    print("✓ test_entropy passed")


def test_tau_annealing():
    """Test tau annealing schedule."""
    agg = LspSoftmaxAggregator(tau=1.0)
    
    # Test exponential annealing
    agg.anneal_tau(0, 10, final_tau=0.01, schedule="exponential")
    assert abs(agg.tau - 1.0) < 0.01, f"Epoch 0 should have initial tau, got {agg.tau}"
    
    agg.anneal_tau(9, 10, final_tau=0.01, schedule="exponential")
    assert abs(agg.tau - 0.01) < 0.01, f"Final epoch should have final tau, got {agg.tau}"
    
    # Test linear annealing
    agg2 = LspSoftmaxAggregator(tau=1.0)
    agg2.anneal_tau(5, 10, final_tau=0.0, schedule="linear")
    assert 0.4 < agg2.tau < 0.6, f"Mid-point linear should be ~0.5, got {agg2.tau}"
    
    print("✓ test_tau_annealing passed")


def test_entropy_loss():
    """Test differentiable entropy loss."""
    agg = LspSoftmaxAggregator(tau=0.5)
    
    # At canonical center, entropy should be low
    a_center = torch.tensor(1.5, requires_grad=True)
    loss_center = agg.entropy_loss(a_center)
    
    # Between centers, entropy should be higher
    a_between = torch.tensor(0.75, requires_grad=True)
    loss_between = agg.entropy_loss(a_between)
    
    assert loss_between > loss_center, \
        f"Entropy between centers ({loss_between}) should be > at center ({loss_center})"
    
    # Test gradient flows
    loss_center.backward()
    assert a_center.grad is not None, "Gradient should flow to a"
    
    print("✓ test_entropy_loss passed")


def test_custom_centers():
    """Test custom center configuration."""
    # Use asymmetric centers like before
    custom = [1.25, 1.0, 0.5, 0.0, -0.25]
    agg = LspSoftmaxAggregator(tau=0.01, centers=custom)
    
    # Check product at custom center
    w = agg.get_weights_for_andness(1.25)
    assert np.argmax(w) == 0, f"Product should be selected at 1.25"
    
    # Check prob_sum at custom center
    w = agg.get_weights_for_andness(-0.25)
    assert np.argmax(w) == 4, f"Prob_sum should be selected at -0.25"
    
    # Verify describe shows custom centers
    info = agg.describe(a=0.5)
    assert info["centers"] == custom, f"describe() should show custom centers"
    
    print("✓ test_custom_centers passed")


def run_all_tests():
    """Run all tests."""
    print("="*60)
    print("Running LspSoftmaxAggregator Tests (Andness-based)")
    print("="*60)
    
    tests = [
        test_andness_selects_product,
        test_andness_selects_min,
        test_andness_selects_avg,
        test_andness_selects_max,
        test_andness_selects_probsum,
        test_output_in_valid_range,
        test_extreme_anchors,
        test_tau_update,
        test_tau_minimum_clamp,
        test_describe,
        test_batch_support,
        test_pruning_support,
        test_aggregator_base_interface,
        test_gradient_flow,
        test_per_node_attach_to_tree,
        test_per_node_pointer_reset,
        test_per_node_error_without_attach,
        test_weights_smooth_transition,
        test_entropy,
        test_tau_annealing,
        test_entropy_loss,
        test_custom_centers,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"✗ {test.__name__} FAILED: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("="*60)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
