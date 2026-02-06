"""
Unit tests for LspSoftmaxAggregator

Run with: pytest test_softmax_lsp.py -v
"""

import pytest
import torch
import numpy as np
from bacon.aggregators.lsp.softmax_lsp import LspSoftmaxAggregator, PerNodeLspSoftmaxAggregator


class TestLspSoftmaxAggregator:
    """Tests for the single-node LspSoftmaxAggregator."""
    
    def test_uniform_weights_with_zero_alpha(self):
        """With alpha=zeros, weights should be ~[0.2, 0.2, 0.2, 0.2, 0.2]."""
        agg = LspSoftmaxAggregator(tau=1.0)
        w = agg.get_weights()
        
        expected = np.array([0.2, 0.2, 0.2, 0.2, 0.2])
        np.testing.assert_allclose(w, expected, rtol=1e-5)
    
    def test_near_one_hot_with_small_tau(self):
        """With small tau and one large alpha entry, weights become near one-hot."""
        agg = LspSoftmaxAggregator(tau=0.01)
        # Set alpha to make entry 2 (avg) dominant
        agg.alpha.data = torch.tensor([0.0, 0.0, 10.0, 0.0, 0.0])
        
        w = agg.get_weights()
        
        # Entry 2 should be near 1.0, others near 0.0
        assert w[2] > 0.99, f"Expected w[2] > 0.99, got {w[2]}"
        assert sum(w[[0, 1, 3, 4]]) < 0.01, f"Expected other weights < 0.01"
    
    def test_output_in_valid_range(self):
        """Output should be in [0, 1] for random inputs in [0, 1]."""
        agg = LspSoftmaxAggregator(tau=1.0)
        
        # Test with random inputs
        torch.manual_seed(42)
        for _ in range(100):
            x = torch.rand(10)
            y = torch.rand(10)
            out = agg(x, y)
            
            assert (out >= 0.0).all(), f"Output below 0: {out.min()}"
            assert (out <= 1.0).all(), f"Output above 1: {out.max()}"
    
    def test_extreme_anchors_product(self):
        """With weights=[1,0,0,0,0], F(x,y) = x*y."""
        agg = LspSoftmaxAggregator(tau=0.001)
        agg.alpha.data = torch.tensor([100.0, 0.0, 0.0, 0.0, 0.0])
        
        x = torch.tensor([0.3, 0.5, 0.8])
        y = torch.tensor([0.4, 0.6, 0.9])
        
        out = agg(x, y)
        expected = x * y
        
        torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)
    
    def test_extreme_anchors_min(self):
        """With weights=[0,1,0,0,0], F(x,y) = min(x,y)."""
        agg = LspSoftmaxAggregator(tau=0.001)
        agg.alpha.data = torch.tensor([0.0, 100.0, 0.0, 0.0, 0.0])
        
        x = torch.tensor([0.3, 0.5, 0.8])
        y = torch.tensor([0.4, 0.6, 0.7])
        
        out = agg(x, y)
        expected = torch.min(x, y)
        
        torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)
    
    def test_extreme_anchors_avg(self):
        """With weights=[0,0,1,0,0], F(x,y) = (x+y)/2."""
        agg = LspSoftmaxAggregator(tau=0.001)
        agg.alpha.data = torch.tensor([0.0, 0.0, 100.0, 0.0, 0.0])
        
        x = torch.tensor([0.3, 0.5, 0.8])
        y = torch.tensor([0.4, 0.6, 0.7])
        
        out = agg(x, y)
        expected = (x + y) / 2.0
        
        torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)
    
    def test_extreme_anchors_max(self):
        """With weights=[0,0,0,1,0], F(x,y) = max(x,y)."""
        agg = LspSoftmaxAggregator(tau=0.001)
        agg.alpha.data = torch.tensor([0.0, 0.0, 0.0, 100.0, 0.0])
        
        x = torch.tensor([0.3, 0.5, 0.8])
        y = torch.tensor([0.4, 0.6, 0.7])
        
        out = agg(x, y)
        expected = torch.max(x, y)
        
        torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)
    
    def test_extreme_anchors_prob_sum(self):
        """With weights=[0,0,0,0,1], F(x,y) = x+y-x*y."""
        agg = LspSoftmaxAggregator(tau=0.001)
        agg.alpha.data = torch.tensor([0.0, 0.0, 0.0, 0.0, 100.0])
        
        x = torch.tensor([0.3, 0.5, 0.8])
        y = torch.tensor([0.4, 0.6, 0.7])
        
        out = agg(x, y)
        expected = x + y - x * y
        
        torch.testing.assert_close(out, expected, rtol=1e-2, atol=1e-2)
    
    def test_tau_update(self):
        """Test that tau can be updated and affects weights."""
        agg = LspSoftmaxAggregator(tau=1.0)
        agg.alpha.data = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
        
        w1 = agg.get_weights()
        
        agg.set_tau(0.1)
        w2 = agg.get_weights()
        
        # With lower tau, distribution should be sharper (more peaked at index 0)
        assert w2[0] > w1[0], "Lower tau should make distribution sharper"
    
    def test_tau_minimum_clamp(self):
        """Test that tau is clamped to minimum value."""
        agg = LspSoftmaxAggregator(tau=0.0)  # Should be clamped
        assert agg.tau >= 1e-4
        
        agg.set_tau(-1.0)  # Should be clamped
        assert agg.tau >= 1e-4
    
    def test_entropy(self):
        """Test entropy calculation."""
        agg = LspSoftmaxAggregator(tau=1.0)
        
        # Uniform distribution should have max entropy
        uniform_entropy = agg.entropy()
        
        # Sharp distribution should have lower entropy
        agg.alpha.data = torch.tensor([10.0, 0.0, 0.0, 0.0, 0.0])
        agg.set_tau(0.1)
        sharp_entropy = agg.entropy()
        
        assert sharp_entropy < uniform_entropy, "Sharp distribution should have lower entropy"
    
    def test_describe(self):
        """Test describe() returns correct info."""
        agg = LspSoftmaxAggregator(tau=0.5)
        info = agg.describe()
        
        assert "tau" in info
        assert "weights" in info
        assert "operators" in info
        assert "alpha" in info
        assert "entropy" in info
        assert "dominant_op" in info
        
        assert info["tau"] == 0.5
        assert len(info["weights"]) == 5
        assert len(info["operators"]) == 5
    
    def test_batch_support(self):
        """Test that batched inputs work correctly."""
        agg = LspSoftmaxAggregator(tau=1.0)
        
        # 2D batch
        x = torch.rand(32, 10)
        y = torch.rand(32, 10)
        out = agg(x, y)
        assert out.shape == x.shape
        
        # 1D batch
        x = torch.rand(100)
        y = torch.rand(100)
        out = agg(x, y)
        assert out.shape == x.shape
        
        # Scalar (0D)
        x = torch.tensor(0.5)
        y = torch.tensor(0.7)
        out = agg(x, y)
        assert out.shape == torch.Size([])
    
    def test_aggregator_base_interface(self):
        """Test compatibility with AggregatorBase interface."""
        agg = LspSoftmaxAggregator(tau=1.0)
        
        # aggregate_tensor (a, w0, w1 should be ignored)
        x = torch.tensor([0.3, 0.5])
        y = torch.tensor([0.4, 0.6])
        a = torch.tensor([0.5, 0.5])  # ignored
        w0 = torch.tensor([0.5, 0.5])  # ignored
        w1 = torch.tensor([0.5, 0.5])  # ignored
        
        out = agg.aggregate_tensor(x, y, a, w0, w1)
        assert out.shape == x.shape
        
        # aggregate_float
        out_float = agg.aggregate_float(0.3, 0.4, 0.5, 0.5, 0.5)
        assert isinstance(out_float, float)
        assert 0.0 <= out_float <= 1.0
    
    def test_gradient_flow(self):
        """Test that gradients flow through the aggregator."""
        agg = LspSoftmaxAggregator(tau=1.0)
        
        x = torch.rand(10, requires_grad=True)
        y = torch.rand(10, requires_grad=True)
        
        out = agg(x, y)
        loss = out.sum()
        loss.backward()
        
        # Check gradients exist
        assert x.grad is not None
        assert y.grad is not None
        assert agg.alpha.grad is not None


class TestPerNodeLspSoftmaxAggregator:
    """Tests for the per-node aggregator (for BACON integration)."""
    
    def test_attach_to_tree(self):
        """Test tree attachment creates correct number of logits."""
        agg = PerNodeLspSoftmaxAggregator(tau=1.0)
        agg.attach_to_tree(num_layers=7)
        
        assert agg.num_layers == 7
        assert len(agg.op_logits_per_node) == 7
        
        for logits in agg.op_logits_per_node:
            assert logits.shape == torch.Size([5])
    
    def test_node_pointer_reset(self):
        """Test that start_forward() resets node pointer."""
        agg = PerNodeLspSoftmaxAggregator(tau=1.0)
        agg.attach_to_tree(num_layers=5)
        
        # Simulate some aggregations
        x = torch.rand(10)
        y = torch.rand(10)
        agg.aggregate(x, y, torch.tensor(0.5), torch.tensor(0.5), torch.tensor(0.5))
        agg.aggregate(x, y, torch.tensor(0.5), torch.tensor(0.5), torch.tensor(0.5))
        
        assert agg._node_ptr == 2
        
        agg.start_forward()
        assert agg._node_ptr == 0
    
    def test_per_node_weights(self):
        """Test that each node can have different weights."""
        agg = PerNodeLspSoftmaxAggregator(tau=0.001)
        agg.attach_to_tree(num_layers=3)
        
        # Set different alphas for each node
        agg.op_logits_per_node[0].data = torch.tensor([100.0, 0.0, 0.0, 0.0, 0.0])  # product
        agg.op_logits_per_node[1].data = torch.tensor([0.0, 0.0, 100.0, 0.0, 0.0])  # avg
        agg.op_logits_per_node[2].data = torch.tensor([0.0, 0.0, 0.0, 0.0, 100.0])  # prob sum
        
        w0 = agg.get_weights(0)
        w1 = agg.get_weights(1)
        w2 = agg.get_weights(2)
        
        # Node 0 should be product-dominant
        assert np.argmax(w0) == 0
        # Node 1 should be avg-dominant
        assert np.argmax(w1) == 2
        # Node 2 should be prob-sum-dominant
        assert np.argmax(w2) == 4
    
    def test_error_without_attach(self):
        """Test that aggregate() raises error if attach_to_tree() not called."""
        agg = PerNodeLspSoftmaxAggregator(tau=1.0)
        
        x = torch.rand(10)
        y = torch.rand(10)
        
        with pytest.raises(RuntimeError):
            agg.aggregate(x, y, torch.tensor(0.5), torch.tensor(0.5), torch.tensor(0.5))
    
    def test_describe_all_nodes(self):
        """Test describe() for all nodes."""
        agg = PerNodeLspSoftmaxAggregator(tau=1.0)
        agg.attach_to_tree(num_layers=3)
        
        # Describe single node
        info0 = agg.describe(node_index=0)
        assert "node" in info0
        assert info0["node"] == 0
        
        # Describe all nodes
        all_info = agg.describe()
        assert len(all_info) == 3


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
