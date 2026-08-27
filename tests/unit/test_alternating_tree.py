"""Unit tests that trace the AlternatingTree end-to-end and verify it actually works.

The AlternatingTree separates *coefficient* learning (scalar w*x per input) from
*structure* learning (routing n -> n-1) and *operator* learning.

Important traced behavior (see test_aggregation_passes_a_fixed_andness):
- ``FirstAggregationLayer.forward`` / ``BinaryAggregationLayer.forward`` call the
  aggregator with a HARD-CODED ``andness = 0.5``.
- Therefore operator selection MUST come from the aggregator's own per-node
  ``op_logits`` (an ``OperatorSetAggregator`` such as ``ArithmeticOperatorSet`` /
  ``BoolOperatorSet``), which the tree wires up via ``attach_to_tree``.
- Pairing the tree with a pure andness-driven aggregator (e.g. LSP power-mean)
  would freeze the operator at the arithmetic mean, which is why the operator
  set aggregators are the correct companions.
"""

import torch
import torch.nn as nn

from bacon.alternatingTree import (
    AlternatingTree,
    CoefficientLayer,
    FirstAggregationLayer,
)
from bacon.aggregators.base import AggregatorBase
from bacon.aggregators.math import ArithmeticOperatorSet
from bacon.binaryTreeLogicNet import binaryTreeLogicNet


CPU = torch.device("cpu")


# ---------------------------------------------------------------------------
# CoefficientLayer
# ---------------------------------------------------------------------------
def test_coefficient_layer_scales_each_input_independently():
    layer = CoefficientLayer(width=3, device=CPU, trainable=True)
    with torch.no_grad():
        layer.log_coefficients.copy_(torch.log(torch.tensor([1.0, 2.0, 0.5])))

    out = layer(torch.tensor([[4.0, 4.0, 4.0]]))

    torch.testing.assert_close(out, torch.tensor([[4.0, 8.0, 2.0]]))


def test_coefficient_layer_clamps_extreme_outputs():
    layer = CoefficientLayer(width=1, device=CPU, trainable=True)
    with torch.no_grad():
        layer.log_coefficients.fill_(10.0)  # exp(10) ~ 22026

    out = layer(torch.tensor([[1.0]]))

    assert out.item() <= 1e4 + 1  # clamped to [-1e4, 1e4]


# ---------------------------------------------------------------------------
# FirstAggregationLayer routing
# ---------------------------------------------------------------------------
def test_fixed_pattern_routes_each_input_to_a_single_node():
    layer = FirstAggregationLayer(in_width=4, out_width=3, learn_routing=False, device=CPU)

    edges = layer.get_edge_weights()

    assert edges.shape == (4, 3)
    # Each source routes to exactly one destination (rows are one-hot).
    torch.testing.assert_close(edges.sum(dim=1), torch.ones(4))
    # Fixed pattern: input i -> node min(i, out_width-1); last inputs pile into last node.
    assert edges[0, 0] == 1.0
    assert edges[1, 1] == 1.0
    assert edges[2, 2] == 1.0
    assert edges[3, 2] == 1.0  # piles into last node


def test_learned_hard_routing_is_one_hot_per_source():
    torch.manual_seed(0)
    layer = FirstAggregationLayer(
        in_width=5, out_width=4, learn_routing=True,
        max_egress=1, use_straight_through=True, device=CPU,
    )
    layer.eval()  # disable gumbel noise
    with torch.no_grad():
        layer.edge_logits.copy_(torch.randn(5, 4))

    edges = layer.get_edge_weights()

    assert edges.shape == (5, 4)
    # Straight-through forward values are a discrete one-hot per row.
    torch.testing.assert_close(edges.sum(dim=1), torch.ones(5))
    assert torch.all((edges == 0) | (edges == 1))


def test_soft_routing_rows_form_a_distribution():
    layer = FirstAggregationLayer(
        in_width=3, out_width=2, learn_routing=True,
        max_egress=1, use_straight_through=False, device=CPU,
    )
    layer.eval()

    edges = layer.get_edge_weights()

    # Soft softmax routing: each row sums to 1 but need not be one-hot.
    torch.testing.assert_close(edges.sum(dim=1), torch.ones(3))
    assert torch.all(edges > 0)


def test_harden_freezes_routing_to_one_hot():
    torch.manual_seed(1)
    layer = FirstAggregationLayer(
        in_width=4, out_width=3, learn_routing=True,
        max_egress=1, use_straight_through=True, device=CPU,
    )
    with torch.no_grad():
        layer.edge_logits.copy_(torch.randn(4, 3))

    layer.harden()

    assert layer.is_hardened is True
    edges = layer.get_edge_weights()
    torch.testing.assert_close(edges, layer.hard_edges)
    torch.testing.assert_close(edges.sum(dim=1), torch.ones(4))
    assert torch.all((edges == 0) | (edges == 1))


def test_routing_regularization_losses_are_finite_and_nonnegative():
    layer = FirstAggregationLayer(
        in_width=4, out_width=3, learn_routing=True, max_egress=1, device=CPU,
    )

    balance = layer.get_balance_loss()
    egress = layer.get_egress_loss()

    assert torch.isfinite(balance) and balance.item() >= 0.0
    assert torch.isfinite(egress) and egress.item() >= 0.0


# ---------------------------------------------------------------------------
# AlternatingTree structure & forward
# ---------------------------------------------------------------------------
def test_tree_builds_one_layer_per_reduction_step():
    tree = AlternatingTree(num_inputs=4, device=CPU)

    # 4 -> 3 -> 2 -> 1 : three aggregation layers, three coefficient layers.
    assert len(tree.agg_layers) == 3
    assert len(tree.coeff_layers) == 3
    assert [layer.out_width for layer in tree.agg_layers] == [3, 2, 1]
    assert tree.num_agg_nodes == 3 + 2 + 1


def test_forward_reduces_any_width_to_a_single_output():
    for n in (2, 3, 5):
        tree = AlternatingTree(num_inputs=n, use_gumbel=False, device=CPU)
        tree.eval()
        agg = ArithmeticOperatorSet(use_gumbel=False)
        agg.attach_to_tree(tree.num_agg_nodes)

        out = tree(torch.rand(6, n), aggregator=agg)

        assert out.shape == (6, 1)
        assert torch.isfinite(out).all()


def test_forward_requires_an_aggregator():
    tree = AlternatingTree(num_inputs=3, device=CPU)
    try:
        tree(torch.rand(2, 3))
        assert False, "expected ValueError when aggregator is missing"
    except ValueError:
        pass


# ---------------------------------------------------------------------------
# Gradient flow (the "actually works" core)
# ---------------------------------------------------------------------------
def test_gradients_flow_to_coefficients_routing_and_operator_logits():
    torch.manual_seed(0)
    tree = AlternatingTree(
        num_inputs=3, learn_coefficients=True, learn_first_routing=True,
        use_gumbel=False, device=CPU,
    )
    agg = ArithmeticOperatorSet(use_gumbel=False)
    agg.attach_to_tree(tree.num_agg_nodes)

    out = tree(torch.rand(8, 3), aggregator=agg)
    out.sum().backward()

    # Coefficient parameters receive gradient.
    coeff_grad = tree.coeff_layers[0].log_coefficients.grad
    assert coeff_grad is not None and torch.any(coeff_grad != 0)

    # Routing logits receive gradient (through the straight-through estimator).
    route_grad = tree.agg_layers[0].edge_logits.grad
    assert route_grad is not None and torch.any(route_grad != 0)

    # Operator-selection logits (in the aggregator) receive gradient.
    op_grads = [p.grad for p in agg.op_logits_per_node.parameters()]
    assert all(g is not None for g in op_grads)
    assert any(torch.any(g != 0) for g in op_grads)


def test_anneal_lowers_temperature_and_gumbel_noise():
    tree = AlternatingTree(
        num_inputs=4, temperature=3.0, final_temperature=0.1,
        use_gumbel=True, gumbel_noise_scale=1.0, device=CPU,
    )

    tree.anneal_temperature(1.0)
    tree.anneal_gumbel(1.0, initial=1.0, final=0.0)

    for layer in tree.agg_layers:
        assert abs(layer.temperature - 0.1) < 1e-6
        assert abs(layer.gumbel_noise_scale - 0.0) < 1e-6


# ---------------------------------------------------------------------------
# Functional: the tree + operator-set aggregator can learn the right operator
# ---------------------------------------------------------------------------
def test_tree_learns_to_select_multiplication_operator():
    """End-to-end proof that operator learning works.

    Coefficients and routing are frozen so the ONLY free parameters are the
    aggregator's per-node operator logits. The target is x0 * x1, so a working
    tree must drive the soft operator mixture toward "mul".
    """
    torch.manual_seed(0)
    tree = AlternatingTree(
        num_inputs=2,
        learn_coefficients=False,   # freeze coefficients (identity scaling)
        learn_first_routing=False,  # fixed 2 -> 1 routing
        use_gumbel=False,
        device=CPU,
    )
    agg = ArithmeticOperatorSet(use_gumbel=False)  # deterministic softmax mixing
    agg.attach_to_tree(tree.num_agg_nodes)

    x = 0.3 + 0.6 * torch.rand(256, 2)  # inputs in [0.3, 0.9]
    target = (x[:, 0] * x[:, 1]).unsqueeze(1)

    opt = torch.optim.Adam(agg.parameters(), lr=0.05)
    first_loss = None
    for _ in range(400):
        opt.zero_grad()
        pred = tree(x, aggregator=agg)
        loss = torch.mean((pred - target) ** 2)
        loss.backward()
        opt.step()
        if first_loss is None:
            first_loss = loss.item()

    logits = agg.op_logits_per_node[0]
    selected = agg.op_names[int(logits.argmax())]

    assert selected == "mul", f"expected 'mul' to win, got '{selected}'"
    assert loss.item() < first_loss * 0.1  # loss dropped by at least 10x


# ---------------------------------------------------------------------------
# Trace-back documentation: aggregation is called with a fixed andness of 0.5
# ---------------------------------------------------------------------------
class _RecordingAggregator(AggregatorBase):
    """Minimal aggregator that records the andness it is handed each call."""

    def __init__(self):
        self.seen_andness = []

    def aggregate_float(self, values, a, weights):  # pragma: no cover - unused
        raise NotImplementedError

    def aggregate_tensor(self, values, a, weights):
        self.seen_andness.append(float(a))
        stacked = torch.stack(list(values), dim=0)
        return stacked.mean(dim=0)


def test_aggregation_passes_a_fixed_andness_of_half():
    """Documents the traced fact: the operator is NOT chosen via andness.

    Every aggregation node receives andness == 0.5, so a graded-logic
    (andness-driven) aggregator could not learn AND/OR here; operator learning
    is delegated to an OperatorSetAggregator's op_logits instead.
    """
    tree = AlternatingTree(num_inputs=4, use_gumbel=False, device=CPU)
    tree.eval()
    rec = _RecordingAggregator()

    tree(torch.rand(3, 4), aggregator=rec)

    assert len(rec.seen_andness) == tree.num_agg_nodes
    assert all(a == 0.5 for a in rec.seen_andness)


# ---------------------------------------------------------------------------
# No external permutation layer needed: the tree learns its own input routing
# ---------------------------------------------------------------------------
def test_alternating_layout_uses_identity_input_map_when_permutation_disabled():
    """With the alternating layout the first aggregation layer learns routing,
    so the external Sinkhorn permutation is redundant and can be turned off."""
    model = binaryTreeLogicNet(
        input_size=4,
        tree_layout="alternating",
        aggregator=ArithmeticOperatorSet(),
        use_permutation_layer=False,
        device=CPU,
    )

    # No standalone permutation stage: inputs pass straight into the tree.
    assert isinstance(model.input_to_leaf, nn.Identity)
    # The tree's own first aggregation layer owns the routing instead.
    assert model.alternating_tree.agg_layers[0].learn_routing is True


def test_alternating_first_routing_learns_input_order_without_permutation_layer():
    """The internal first-routing edge logits actually train, which is what makes
    a separate permutation layer unnecessary for this layout.

    Note: at the fully symmetric init (all edge_logits == 0) the row-softmax makes
    the straight-through gradient cancel per row (a saddle); Gumbel noise breaks
    that symmetry during real training. We perturb the logits to represent any
    non-symmetric routing state and confirm the gradient then flows.
    """
    torch.manual_seed(0)
    model = binaryTreeLogicNet(
        input_size=4,
        tree_layout="alternating",
        aggregator=ArithmeticOperatorSet(use_gumbel=False),
        use_permutation_layer=False,
        alternating_learn_first_routing=True,
        device=CPU,
    )

    first_routing = model.alternating_tree.agg_layers[0]
    assert first_routing.learn_routing is True
    with torch.no_grad():
        first_routing.edge_logits.copy_(torch.randn(4, 3))

    out = model(torch.rand(8, 4))
    out.sum().backward()

    assert first_routing.edge_logits.grad is not None
    assert torch.any(first_routing.edge_logits.grad != 0)
