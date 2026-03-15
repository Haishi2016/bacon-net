import torch

from bacon.alternatingTree import AlternatingTree, CoefficientLayer
from bacon.aggregators.math import ArithmeticOperatorSet
from bacon.binaryTreeLogicNet import binaryTreeLogicNet


def test_coefficient_layer_defaults_to_ax_behavior():
    layer = CoefficientLayer(width=2, device=torch.device("cpu"), trainable=False)

    outputs = layer(torch.tensor([[2.0, 3.0]]))

    torch.testing.assert_close(outputs, torch.tensor([[2.0, 3.0]]))


def test_coefficient_layer_can_learn_squared_power_on_positive_inputs():
    layer = CoefficientLayer(
        width=2,
        device=torch.device("cpu"),
        trainable=True,
        learn_exponents=True,
        min_exponent=1.0,
        max_exponent=2.0,
    )

    with torch.no_grad():
        layer.log_coefficients.zero_()
        layer.exponent_logits.fill_(12.0)

    outputs = layer(torch.tensor([[2.0, 3.0]]))
    exponents = layer.get_exponents()

    assert torch.all(exponents <= 2.0)
    assert torch.all(exponents >= 1.0)
    torch.testing.assert_close(outputs, torch.tensor([[4.0, 9.0]]), atol=1e-4, rtol=1e-4)


def test_coefficient_layer_exponent_regularization_is_zero_without_exponents():
    layer = CoefficientLayer(width=1, device=torch.device("cpu"), trainable=False)

    loss = layer.get_exponent_regularization_loss()

    assert loss.item() == 0.0


def test_coefficient_layer_endpoint_regularizer_prefers_1_or_2_over_midpoint():
    layer = CoefficientLayer(
        width=3,
        device=torch.device("cpu"),
        trainable=True,
        learn_exponents=True,
        min_exponent=1.0,
        max_exponent=2.0,
    )

    with torch.no_grad():
        layer.exponent_logits.copy_(torch.tensor([-12.0, 0.0, 12.0]))

    penalties = ((layer.get_exponents() - 1.0) / (2.0 - 1.0)) * (1.0 - ((layer.get_exponents() - 1.0) / (2.0 - 1.0)))
    loss = layer.get_exponent_regularization_loss()

    assert penalties[1].item() > penalties[0].item()
    assert penalties[1].item() > penalties[2].item()
    assert loss.item() > 0.0


def test_single_input_alternating_tree_keeps_trainable_coefficient_layer():
    tree = AlternatingTree(num_inputs=1, learn_coefficients=True, device=torch.device("cpu"))

    assert len(tree.coeff_layers) == 1
    assert sum(parameter.numel() for parameter in tree.parameters()) > 0

    outputs = tree(torch.tensor([[3.0]]), aggregator=object())

    torch.testing.assert_close(outputs, torch.tensor([[3.0]]))


def test_binary_tree_logic_net_appends_constant_leaf_after_routing():
    model = binaryTreeLogicNet(
        input_size=2,
        tree_layout="alternating",
        aggregator=ArithmeticOperatorSet(),
        use_permutation_layer=False,
        use_constant_input=True,
        device=torch.device("cpu"),
    )

    routed = model.input_to_leaf(torch.tensor([[2.0, 3.0]]))
    assert routed.shape[1] == 2

    outputs = model(torch.tensor([[2.0, 3.0]]))

    assert model.num_leaves == 3
    assert model.alternating_tree.num_inputs == 3
    assert outputs.shape == (1, 1)