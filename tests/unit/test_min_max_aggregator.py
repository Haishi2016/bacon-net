import pytest
import torch

from bacon.aggregators.bool import MinMaxAggregator

@pytest.mark.unit
@pytest.mark.parametrize(
    "values",
    [
        [0.2, 0.8],
        [0.7, 0.4, 0.9],
        [0.6, 0.1, 0.3, 0.8],
    ],
    ids=["2-vars", "3-vars", "4-vars"],
)
@pytest.mark.parametrize(
    "weights",
    [
        "equal",
        "uneven",
    ],
)
@pytest.mark.parametrize(
    "a,expected_kind",
    [
        (1.0, "min"),
        (0.0, "max"),
        (0.5, "min"),
    ],
)
def test_minmax_aggregate_float(values, weights, a, expected_kind):
    agg = MinMaxAggregator()

    if weights == "equal":
        ws = [1.0 / len(values)] * len(values)
    else:
        ws = [float(i + 1) for i in range(len(values))]

    expected = min(values) if expected_kind == "min" else max(values)
    out = agg.aggregate_float(values, a, ws)

    assert out == pytest.approx(expected, abs=1e-6)


@pytest.mark.unit
@pytest.mark.parametrize(
    "values",
    [
        [0.2, 0.8],
        [0.7, 0.4, 0.9],
        [0.6, 0.1, 0.3, 0.8],
    ],
    ids=["2-vars", "3-vars", "4-vars"],
)
@pytest.mark.parametrize("a,expected_kind", [(1.0, "min"), (0.0, "max"), (0.5, "min")])
def test_minmax_aggregate_tensor(values, a, expected_kind):
    agg = MinMaxAggregator()

    tensors = [torch.tensor(v, dtype=torch.float32) for v in values]
    expected = min(values) if expected_kind == "min" else max(values)

    out = agg.aggregate_tensor(tensors, andness=torch.tensor(a), weights=[0.9] * len(values))

    assert float(out.item()) == pytest.approx(expected, abs=1e-6)

@pytest.mark.unit
def test_weights_do_not_change_output_for_same_values_and_a():
    agg = MinMaxAggregator()
    values = [0.2, 0.9, 0.4, 0.7]

    out_equal = agg.aggregate_float(values, a=1.0, weights=[0.25, 0.25, 0.25, 0.25])
    out_uneven = agg.aggregate_float(values, a=1.0, weights=[0.7, 0.1, 0.1, 0.1])

    assert out_equal == pytest.approx(out_uneven, abs=1e-6)
