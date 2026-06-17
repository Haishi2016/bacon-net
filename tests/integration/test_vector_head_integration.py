"""Integration tests for the ``head_type='vector'`` path.

These check that:
  1. The default ``head_type='binary'`` path is completely unchanged
     (byte-identical state + identical forward output) when the new
     parameters are left at their defaults.
  2. The vectorized path in ``binaryTreeLogicNet`` reproduces the binary
     path for ``num_heads=1`` (the head reuses the same GL math).
  3. ``baconNet`` passes the new arguments through and produces a
     ``(batch, num_heads)`` output in vector mode.
"""

import pytest
import torch

from bacon.binaryTreeLogicNet import binaryTreeLogicNet
from bacon.baconNet import baconNet
from bacon.aggregators.lsp.generic_gl import GenericGLAggregator
from bacon.vectorizedLogicHead import DEFAULT_ANCHORS


def _make_binary_tree(input_size, seed=0):
    torch.manual_seed(seed)
    agg = GenericGLAggregator(anchors=DEFAULT_ANCHORS, weight_mode="static", tau=0.5)
    return binaryTreeLogicNet(
        input_size=input_size,
        aggregator=agg,
        tree_layout="left",
        weight_mode="trainable",
        normalize_andness=True,
        use_permutation_layer=False,
        device=torch.device("cpu"),
    )


class TestBinaryPathUnchanged:

    def test_defaults_select_binary_head(self):
        tree = _make_binary_tree(6)
        assert tree.head_type == "binary"
        assert tree.num_heads == 1
        assert tree.vector_head is None

    def test_explicit_binary_matches_default(self):
        # Building with explicit head_type='binary' must match the default
        # build exactly (same parameters, same forward output).
        a = _make_binary_tree(6, seed=42)
        torch.manual_seed(0)
        agg = GenericGLAggregator(anchors=DEFAULT_ANCHORS, weight_mode="static", tau=0.5)
        torch.manual_seed(42)
        b = binaryTreeLogicNet(
            input_size=6,
            aggregator=agg,
            tree_layout="left",
            weight_mode="trainable",
            normalize_andness=True,
            use_permutation_layer=False,
            device=torch.device("cpu"),
            head_type="binary",
            num_heads=1,
        )
        a.eval(); b.eval()
        x = torch.rand(8, 6)
        with torch.no_grad():
            assert torch.allclose(a(x), b(x), atol=1e-6)


class TestVectorPathMatchesBinary:

    def test_vector_num_heads_one_matches_binary(self):
        input_size = 7
        binary = _make_binary_tree(input_size, seed=7)
        binary.eval()

        # Vector tree sharing the same aggregator anchor logits.
        torch.manual_seed(7)
        agg = GenericGLAggregator(anchors=DEFAULT_ANCHORS, weight_mode="static", tau=0.5)
        vector = binaryTreeLogicNet(
            input_size=input_size,
            aggregator=agg,
            tree_layout="left",
            weight_mode="trainable",
            normalize_andness=True,
            use_permutation_layer=False,
            device=torch.device("cpu"),
            head_type="vector",
            num_heads=1,
        )
        vector.eval()
        with torch.no_grad():
            vector.vector_head.alpha_logits.copy_(
                binary.aggregator.alpha_logits.detach().unsqueeze(0)
            )

        x = torch.rand(10, input_size)
        with torch.no_grad():
            ref = binary(x)          # (10, 1)
            got = vector(x)          # (10, 1)
        assert torch.allclose(ref, got, atol=1e-5), (ref - got).abs().max()


class TestVectorLayoutsMatchBinary:
    """Vector mode reproduces the binary tree for full and alternating layouts."""

    def _make_pair(self, input_size, layout, seed):
        torch.manual_seed(seed)
        agg_b = GenericGLAggregator(anchors=DEFAULT_ANCHORS, weight_mode="static", tau=0.5)
        binary = binaryTreeLogicNet(
            input_size=input_size, aggregator=agg_b, tree_layout=layout,
            weight_mode="trainable", normalize_andness=True,
            use_permutation_layer=False, device=torch.device("cpu"),
        )
        binary.eval()
        torch.manual_seed(seed)
        agg_v = GenericGLAggregator(anchors=DEFAULT_ANCHORS, weight_mode="static", tau=0.5)
        vector = binaryTreeLogicNet(
            input_size=input_size, aggregator=agg_v, tree_layout=layout,
            weight_mode="trainable", normalize_andness=True,
            use_permutation_layer=False, device=torch.device("cpu"),
            head_type="vector", num_heads=1,
        )
        vector.eval()
        return binary, vector

    def test_full_layout_matches(self):
        binary, vector = self._make_pair(6, "full", seed=3)
        with torch.no_grad():
            vector.vector_head.alpha_logits.copy_(
                binary.aggregator.alpha_logits.detach().unsqueeze(0)
            )
        x = torch.rand(10, 6)
        with torch.no_grad():
            assert torch.allclose(binary(x), vector(x), atol=1e-5)

    def test_alternating_layout_matches(self):
        binary, vector = self._make_pair(5, "alternating", seed=4)
        with torch.no_grad():
            # Randomize binary coefficients, then copy anchor logits + coeffs.
            for cl in binary.alternating_tree.coeff_layers:
                torch.nn.init.normal_(cl.log_coefficients, std=0.5)
            vector.vector_head.alpha_logits.copy_(
                binary.aggregator.alpha_logits.detach().unsqueeze(0)
            )
            for hp, cl in zip(vector.vector_head.coeff_log,
                              binary.alternating_tree.coeff_layers):
                hp.copy_(cl.log_coefficients.detach().unsqueeze(0))
        x = torch.rand(10, 5)
        with torch.no_grad():
            assert torch.allclose(binary(x), vector(x), atol=1e-5)


class TestBaconNetPassthrough:

    def test_vector_mode_output_shape(self):
        net = baconNet(
            input_size=8,
            aggregator="gl.generic",
            tree_layout="left",
            use_permutation_layer=False,
            head_type="vector",
            num_heads=16,
        )
        net.eval()
        device = net.assembler.device
        with torch.no_grad():
            out = net(torch.rand(4, 8, device=device))
        assert out.shape == (4, 16)

    @pytest.mark.parametrize("layout", ["left", "full", "alternating"])
    def test_vector_mode_all_layouts(self, layout):
        net = baconNet(
            input_size=8,
            aggregator="gl.generic",
            tree_layout=layout,
            use_permutation_layer=False,
            head_type="vector",
            num_heads=12,
        )
        net.eval()
        device = net.assembler.device
        with torch.no_grad():
            out = net(torch.rand(4, 8, device=device))
        assert out.shape == (4, 12)

    def test_binary_mode_default_shape(self):
        net = baconNet(
            input_size=8,
            aggregator="gl.generic",
            tree_layout="left",
            use_permutation_layer=False,
        )
        net.eval()
        device = net.assembler.device
        with torch.no_grad():
            out = net(torch.rand(4, 8, device=device))
        assert out.shape == (4, 1)

