import torch

from bacon.alternatingTree import CoefficientLayer


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