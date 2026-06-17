"""Unit tests for :class:`bacon.vectorizedLogicHead.VectorLogicHead`.

Covers:
  1. Output shape and range (shared and per-head inputs).
  2. Per-head independence (each head computes its own rule).
  3. Anchor-weight selection (sharp tau -> pure anchor behavior).
  4. Validation of bad arguments.
  5. Equivalence with a left-associated ``binaryTreeLogicNet`` using the
     static ``GenericGLAggregator`` (the math the head reproduces).
"""

import pytest
import torch

from bacon.vectorizedLogicHead import VectorLogicHead, VectorTreeLogicHead, DEFAULT_ANCHORS


class TestShapeAndRange:

    def test_shared_inputs_shape(self):
        head = VectorLogicHead(input_size=8, num_heads=5)
        out = head(torch.rand(4, 8))
        assert out.shape == (4, 5)

    def test_per_head_inputs_shape(self):
        head = VectorLogicHead(input_size=8, num_heads=5)
        out = head(torch.rand(4, 5, 8))
        assert out.shape == (4, 5)

    def test_output_in_unit_interval(self):
        head = VectorLogicHead(input_size=12, num_heads=7)
        torch.manual_seed(0)
        out = head(torch.rand(16, 12))
        assert (out > 0.0).all() and (out < 1.0).all()

    def test_single_input_passthrough(self):
        # With one leaf there is nothing to fold; output tracks that input.
        head = VectorLogicHead(input_size=1, num_heads=3)
        x = torch.rand(4, 1)
        out = head(x)
        assert torch.allclose(out, x.expand(4, 3).clamp(head.eps, 1 - head.eps))


class TestHeadIndependence:

    def test_distinct_weights_give_distinct_outputs(self):
        head = VectorLogicHead(input_size=6, num_heads=2, tau=0.1)
        # Head 0 -> pure min (andness high); head 1 -> pure max (andness low).
        with torch.no_grad():
            head.alpha_logits.zero_()
            head.alpha_logits[0, 0] = 10.0   # 'min'
            head.alpha_logits[1, -1] = 10.0  # 'max'
        x = torch.rand(8, 6)
        out = head(x)
        # Left fold of pairwise-min == global min; same for max.
        assert torch.allclose(out[:, 0], x.min(dim=1).values, atol=1e-4)
        assert torch.allclose(out[:, 1], x.max(dim=1).values, atol=1e-4)


class TestAnchorWeights:

    def test_weights_sum_to_one(self):
        head = VectorLogicHead(input_size=4, num_heads=3)
        w = head.anchor_weights()
        assert w.shape == (3, len(DEFAULT_ANCHORS))
        assert torch.allclose(w.sum(dim=-1), torch.ones(3), atol=1e-5)

    def test_uniform_at_init(self):
        head = VectorLogicHead(input_size=4, num_heads=2, tau=1.0)
        w = head.anchor_weights()
        expected = 1.0 / len(DEFAULT_ANCHORS)
        assert torch.allclose(w, torch.full_like(w, expected), atol=1e-5)


class TestValidation:

    def test_bad_layout(self):
        with pytest.raises(NotImplementedError):
            VectorLogicHead(input_size=4, num_heads=2, layout="balanced")

    def test_bad_anchor(self):
        with pytest.raises(ValueError):
            VectorLogicHead(input_size=4, num_heads=2, anchors=("bogus",))

    def test_bad_sizes(self):
        with pytest.raises(ValueError):
            VectorLogicHead(input_size=0, num_heads=2)
        with pytest.raises(ValueError):
            VectorLogicHead(input_size=4, num_heads=0)

    def test_wrong_input_width(self):
        head = VectorLogicHead(input_size=5, num_heads=2)
        with pytest.raises(ValueError):
            head(torch.rand(3, 4))

    def test_gradients_flow(self):
        head = VectorLogicHead(input_size=6, num_heads=4)
        out = head(torch.rand(8, 6))
        out.sum().backward()
        assert head.alpha_logits.grad is not None
        assert not torch.isnan(head.alpha_logits.grad).any()


class TestEquivalenceWithBinaryTree:
    """The head reproduces a left-associated tree with a static GL aggregator."""

    def test_matches_binary_tree_logic_net(self):
        from bacon.binaryTreeLogicNet import binaryTreeLogicNet
        from bacon.aggregators.lsp.generic_gl import GenericGLAggregator

        torch.manual_seed(123)
        input_size = 7
        agg = GenericGLAggregator(anchors=DEFAULT_ANCHORS, weight_mode="static",
                                  tau=0.5)
        tree = binaryTreeLogicNet(
            input_size=input_size,
            aggregator=agg,
            tree_layout="left",
            weight_mode="trainable",
            normalize_andness=True,
            use_permutation_layer=False,  # identity input ordering
            device=torch.device("cpu"),
        )
        tree.eval()

        head = VectorLogicHead(input_size=input_size, num_heads=1, tau=0.5)
        # Copy the aggregator's learned anchor logits into the head.
        with torch.no_grad():
            head.alpha_logits.copy_(agg.alpha_logits.detach().unsqueeze(0))

        x = torch.rand(10, input_size)
        with torch.no_grad():
            ref = tree(x)            # (10, 1)
            got = head(x)            # (10, 1)
        assert torch.allclose(ref, got, atol=1e-5), (ref - got).abs().max()


def _build_static_tree(input_size, layout, seed=0):
    """A left/full/alternating binaryTreeLogicNet with a static GL aggregator."""
    from bacon.binaryTreeLogicNet import binaryTreeLogicNet
    from bacon.aggregators.lsp.generic_gl import GenericGLAggregator

    torch.manual_seed(seed)
    agg = GenericGLAggregator(anchors=DEFAULT_ANCHORS, weight_mode="static", tau=0.5)
    tree = binaryTreeLogicNet(
        input_size=input_size,
        aggregator=agg,
        tree_layout=layout,
        weight_mode="trainable",
        normalize_andness=True,
        use_permutation_layer=False,
        device=torch.device("cpu"),
    )
    tree.eval()
    return tree, agg


class TestLayoutShapes:

    @pytest.mark.parametrize("layout", ["left", "full", "alternating"])
    def test_shape_and_range(self, layout):
        head = VectorLogicHead(input_size=9, num_heads=5, layout=layout)
        torch.manual_seed(0)
        out = head(torch.rand(7, 9))
        assert out.shape == (7, 5)
        assert (out > 0.0).all() and (out < 1.0).all()

    def test_unsupported_layout_raises(self):
        with pytest.raises(NotImplementedError):
            VectorLogicHead(input_size=4, num_heads=2, layout="balanced")

    def test_alternating_has_per_head_coeffs(self):
        head = VectorLogicHead(input_size=5, num_heads=3, layout="alternating")
        # One coefficient layer per aggregation: widths 5, 4, 3, 2.
        assert [p.shape for p in head.coeff_log] == [
            (3, 5), (3, 4), (3, 3), (3, 2)
        ]

    def test_left_and_full_have_no_coeffs(self):
        assert VectorLogicHead(4, 2, layout="left").coeff_log is None
        assert VectorLogicHead(4, 2, layout="full").coeff_log is None


class TestFullLayoutEquivalence:
    """Full layout reduces to a single N-ary GL blend over all inputs."""

    def test_matches_binary_tree_full(self):
        input_size = 6
        tree, agg = _build_static_tree(input_size, "full", seed=11)
        head = VectorLogicHead(input_size=input_size, num_heads=1,
                               layout="full", tau=0.5)
        with torch.no_grad():
            head.alpha_logits.copy_(agg.alpha_logits.detach().unsqueeze(0))

        x = torch.rand(10, input_size)
        with torch.no_grad():
            ref = tree(x)
            got = head(x)
        assert torch.allclose(ref, got, atol=1e-5), (ref - got).abs().max()


class TestAlternatingLayoutEquivalence:
    """Alternating layout: coefficient scaling interleaved with N-ary blends."""

    def test_matches_binary_tree_alternating(self):
        input_size = 5
        tree, agg = _build_static_tree(input_size, "alternating", seed=21)
        # Randomize the tree's coefficients so they actually matter.
        with torch.no_grad():
            for cl in tree.alternating_tree.coeff_layers:
                torch.nn.init.normal_(cl.log_coefficients, std=0.5)

        head = VectorLogicHead(input_size=input_size, num_heads=1,
                               layout="alternating", tau=0.5)
        with torch.no_grad():
            head.alpha_logits.copy_(agg.alpha_logits.detach().unsqueeze(0))
            # Copy each per-node coefficient layer into the head (head dim = 1).
            for hp, cl in zip(head.coeff_log, tree.alternating_tree.coeff_layers):
                hp.copy_(cl.log_coefficients.detach().unsqueeze(0))

        x = torch.rand(10, input_size)
        with torch.no_grad():
            ref = tree(x)
            got = head(x)
        assert torch.allclose(ref, got, atol=1e-5), (ref - got).abs().max()

    def test_single_input_alternating_matches(self):
        # n == 1 has a single trailing coefficient layer and no aggregation.
        tree, agg = _build_static_tree(1, "alternating", seed=31)
        with torch.no_grad():
            torch.nn.init.normal_(tree.alternating_tree.coeff_layers[0].log_coefficients,
                                  std=0.5)
        head = VectorLogicHead(input_size=1, num_heads=1,
                               layout="alternating", tau=0.5)
        with torch.no_grad():
            head.alpha_logits.copy_(agg.alpha_logits.detach().unsqueeze(0))
            head.coeff_log[0].copy_(
                tree.alternating_tree.coeff_layers[0].log_coefficients.detach().unsqueeze(0)
            )
        x = torch.rand(8, 1)
        with torch.no_grad():
            ref = tree(x)
            got = head(x)
        assert torch.allclose(ref, got, atol=1e-5), (ref - got).abs().max()


class TestWeightedFullLayout:
    """Per-concept relevance gates (use_input_weights) on the full layout."""

    def test_requires_full_layout(self):
        for layout in ("left", "alternating"):
            with pytest.raises(NotImplementedError):
                VectorLogicHead(input_size=6, num_heads=4, layout=layout,
                                use_input_weights=True)

    def test_rejects_non_core_anchors(self):
        with pytest.raises(ValueError):
            VectorLogicHead(input_size=6, num_heads=4, layout="full",
                            anchors=("product",), use_input_weights=True)

    def test_output_shape_and_range(self):
        head = VectorLogicHead(input_size=6, num_heads=5, layout="full",
                               use_input_weights=True)
        out = head(torch.rand(12, 6))
        assert out.shape == (12, 5)
        assert (out > 0).all() and (out < 1).all()

    def test_has_per_concept_weight_logits(self):
        head = VectorLogicHead(input_size=6, num_heads=5, layout="full",
                               use_input_weights=True)
        assert head.weight_logits is not None
        assert head.weight_logits.shape == (5, 6)


class TestVectorTreeLogicHead:
    """Faithful multi-head lift of the scalar ``left`` tree with a real LSP
    aggregator (lsp.full_weight / lsp.half_weight)."""

    @staticmethod
    def _agg(name="full"):
        from bacon.aggregators.lsp.full_weight import FullWeightAggregator
        from bacon.aggregators.lsp.half_weight import HalfWeightAggregator
        return FullWeightAggregator() if name == "full" else HalfWeightAggregator()

    @pytest.mark.parametrize("agg_name", ["full", "half"])
    @pytest.mark.parametrize("input_size", [2, 3, 5, 8])
    def test_matches_serial_left_tree(self, agg_name, input_size):
        """One head with copied parameters reproduces the scalar left tree
        exactly (same routing off, same andness, same softmax weights)."""
        from bacon.binaryTreeLogicNet import binaryTreeLogicNet
        from bacon.vectorizedLogicHead import VectorTreeLogicHead

        torch.manual_seed(7)
        serial = binaryTreeLogicNet(
            input_size=input_size, aggregator=self._agg(agg_name),
            tree_layout="left", weight_mode="trainable",
            weight_normalization="softmax", normalize_andness=True,
            use_permutation_layer=False, device=torch.device("cpu"),
        )
        serial.eval()
        head = VectorTreeLogicHead(
            input_size=input_size, num_heads=1, aggregator=self._agg(agg_name),
            normalize_andness=True, use_permutation_layer=False,
            use_transformation_layer=False,
        )
        with torch.no_grad():
            for i in range(input_size - 1):
                head.bias[0, i] = serial.biases[i].detach().squeeze()
                head.weight_logits[0, i, 0] = serial.weights[i].detach()[0]
                head.weight_logits[0, i, 1] = serial.weights[i].detach()[1]
        x = torch.rand(8, input_size)
        with torch.no_grad():
            ref = serial(x).squeeze(1)
            got = head(x)[:, 0]
        assert torch.allclose(ref, got, atol=1e-5), (ref - got).abs().max()

    def test_output_shape_and_range(self):
        head = VectorTreeLogicHead(input_size=9, num_heads=6, aggregator=self._agg())
        out = head(torch.rand(7, 9))
        assert out.shape == (7, 6)
        assert (out > 0).all() and (out < 1).all()

    def test_single_leaf_passthrough(self):
        head = VectorTreeLogicHead(input_size=1, num_heads=3, aggregator=self._agg())
        x = torch.rand(4, 1)
        out = head(x)
        assert torch.allclose(out, x.expand(4, 3).clamp(head.eps, 1 - head.eps))

    def test_heads_are_distinct(self):
        head = VectorTreeLogicHead(input_size=6, num_heads=5, aggregator=self._agg())
        out = head(torch.rand(8, 6))
        # Random per-head routing/andness/weights -> heads compute different rules.
        assert (out.std(dim=1) > 1e-6).all()

    def test_gradients_flow_to_all_parameters(self):
        head = VectorTreeLogicHead(
            input_size=6, num_heads=4, aggregator=self._agg(),
            use_permutation_layer=True, use_transformation_layer=True,
        )
        out = head(torch.rand(10, 6))
        out.sum().backward()
        for p in (head.bias, head.weight_logits, head.perm_logits,
                  head.transform_logits):
            assert p.grad is not None and p.grad.abs().sum() > 0
            assert not torch.isnan(p.grad).any()

    def test_negation_uses_absent_concepts(self):
        """With the transformation layer a head can key on a concept being OFF
        (1 - x), which a monotone aggregator alone cannot express."""
        torch.manual_seed(0)
        head = VectorTreeLogicHead(
            input_size=2, num_heads=1, aggregator=self._agg(),
            use_permutation_layer=False, use_transformation_layer=True,
        )
        with torch.no_grad():
            # Force head to negate both concepts (select the negation branch).
            head.transform_logits[..., 0] = -10.0
            head.transform_logits[..., 1] = 10.0
        low = head(torch.zeros(1, 2))[0, 0]   # both OFF -> negated -> high
        high = head(torch.ones(1, 2))[0, 0]   # both ON  -> negated -> low
        assert low > high

    def test_rejects_non_lsp_aggregator(self):
        from bacon.aggregators.lsp.generic_gl import GenericGLAggregator
        from bacon.vectorizedLogicHead import VectorTreeLogicHead
        with pytest.raises(TypeError):
            VectorTreeLogicHead(input_size=4, num_heads=2,
                                aggregator=GenericGLAggregator())


class TestBinaryTreeNetVectorDispatch:
    """head_type='vector' picks the right engine from the aggregator."""

    def test_lsp_aggregator_builds_tree_head(self):
        from bacon.binaryTreeLogicNet import binaryTreeLogicNet
        from bacon.vectorizedLogicHead import VectorTreeLogicHead
        from bacon.aggregators.lsp.full_weight import FullWeightAggregator

        net = binaryTreeLogicNet(
            input_size=10, aggregator=FullWeightAggregator(), tree_layout="left",
            normalize_andness=True, use_permutation_layer=True,
            head_type="vector", num_heads=8, device=torch.device("cpu"),
        )
        assert isinstance(net.vector_head, VectorTreeLogicHead)
        out = net(torch.rand(5, 10))
        assert out.shape == (5, 8)
        assert (out > 0).all() and (out < 1).all()

    def test_lsp_non_left_layout_raises(self):
        from bacon.binaryTreeLogicNet import binaryTreeLogicNet
        from bacon.aggregators.lsp.full_weight import FullWeightAggregator
        with pytest.raises(NotImplementedError):
            binaryTreeLogicNet(
                input_size=10, aggregator=FullWeightAggregator(),
                tree_layout="full", head_type="vector", num_heads=4,
                device=torch.device("cpu"),
            )

    def test_gl_generic_still_uses_anchor_head(self):
        from bacon.binaryTreeLogicNet import binaryTreeLogicNet
        from bacon.vectorizedLogicHead import VectorLogicHead
        from bacon.aggregators.lsp.generic_gl import GenericGLAggregator

        net = binaryTreeLogicNet(
            input_size=10, aggregator=GenericGLAggregator(), tree_layout="full",
            use_permutation_layer=True, head_type="vector", num_heads=8,
            device=torch.device("cpu"),
        )
        assert isinstance(net.vector_head, VectorLogicHead)
        out = net(torch.rand(5, 10))
        assert out.shape == (5, 8)

    def test_breaks_head_symmetry_at_init(self):
        # Shared input + plain anchors -> every head is identical (the bug that
        # pins task accuracy at chance). Relevance gates must diversify heads.
        torch.manual_seed(0)
        plain = VectorLogicHead(input_size=8, num_heads=8, layout="full")
        x = torch.rand(16, 8)
        assert plain(x).std(dim=1).mean().item() < 1e-6  # identical heads
        torch.manual_seed(0)
        weighted = VectorLogicHead(input_size=8, num_heads=8, layout="full",
                                   use_input_weights=True)
        assert weighted(x).std(dim=1).mean().item() > 0.0  # heads differ

    def test_reduces_to_plain_anchors_as_gates_saturate(self):
        # As all gates -> 1 the weighted blend equals the unweighted blend.
        torch.manual_seed(3)
        weighted = VectorLogicHead(input_size=7, num_heads=4, layout="full",
                                   use_input_weights=True)
        plain = VectorLogicHead(input_size=7, num_heads=4, layout="full")
        with torch.no_grad():
            plain.alpha_logits.copy_(weighted.alpha_logits)
            weighted.weight_logits.fill_(20.0)  # sigmoid(20) ~= 1
        x = torch.rand(10, 7)
        with torch.no_grad():
            diff = (weighted(x) - plain(x)).abs().max().item()
        assert diff < 1e-4, diff

    def test_learns_argmax_above_chance(self):
        # 8 concepts, class = argmax(concepts); plain shared-input head cannot
        # exceed chance, the weighted head must learn well above 1/8.
        torch.manual_seed(1)

        def gen(n):
            c = torch.rand(n, 8)
            return c, c.argmax(dim=1)

        head = VectorLogicHead(input_size=8, num_heads=8, layout="full",
                               use_input_weights=True)
        opt = torch.optim.Adam(head.parameters(), lr=0.05)
        lossf = torch.nn.CrossEntropyLoss()
        for _ in range(300):
            c, y = gen(256)
            p = head(c)
            loss = lossf(torch.log(p / (1 - p)), y)
            opt.zero_grad()
            loss.backward()
            opt.step()
        c, y = gen(2000)
        with torch.no_grad():
            acc = (head(c).argmax(1) == y).float().mean().item()
        assert acc > 0.5, acc

    def test_gradients_flow_to_weight_logits(self):
        head = VectorLogicHead(input_size=6, num_heads=4, layout="full",
                               use_input_weights=True)
        out = head(torch.rand(8, 6))
        out.sum().backward()
        assert head.weight_logits.grad is not None
        assert torch.isfinite(head.weight_logits.grad).all()
        assert head.weight_logits.grad.abs().sum() > 0


