from bacon.aggregators.base import AggregatorBase
from typing import Sequence, Any


"""Min/Max boolean-style aggregator.

This module provides an aggregator that interpolates between elementwise
minimum (AND-like) and elementwise maximum (OR-like) behavior using an
``andness`` control signal.
"""

class MinMaxAggregator(AggregatorBase):
    """Aggregate a sequence of inputs with min/max interpolation.

    The aggregator computes:

    - elementwise minimum when ``andness`` is near 1
    - elementwise maximum when ``andness`` is near 0

    with a straight-through hard gating step so gradients can still flow
    through the continuous proxy during training.
    """

    _ANDNESS_SHARPNESS = 10.0

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Min/Max aggregation does not consume learned edge scales.
        self.uses_edge_scales = False
    
    def aggregate_float(self, values: Sequence[float], a: float, weights: Sequence[float]) -> float:
        """Float-friendly wrapper around :meth:`aggregate_tensor`.

        Args:
            values: Input scalar values to aggregate.
            a: Andness control value.
            weights: Optional per-input weights used as soft gates.

        Returns:
            Python float result.
        """
        import torch
        xs = torch.tensor(values, dtype=torch.float32)
        a_t = torch.tensor(a, dtype=torch.float32)
        out = self.aggregate_tensor([xi for xi in xs], a_t, weights=None)
        return float(out.item())

    def aggregate_tensor(self, values: Sequence[Any], andness, weights=None):
        """Aggregate one or more tensor-like values.

        Args:
            values: Non-empty sequence of tensors with compatible shapes.
            andness: Control signal where larger values bias toward min/AND and
                smaller values bias toward max/OR.
            weights: Optional per-input gates in ``[0, 1]``. When provided,
                each input is blended with a neutral baseline before reduction.

        Returns:
            Tensor with the same broadcast-compatible shape as each input.

        Raises:
            ValueError: If ``values`` is empty.
        """
        import torch
        if len(values) == 0:
            raise ValueError("aggregate_tensor: values must be non-empty")

        if weights is not None:
            if not isinstance(andness, torch.Tensor):
                andness = torch.tensor(andness, dtype=values[0].dtype, device=values[0].device)
            # Map AND-like behavior to neutral 1 and OR-like behavior to neutral 0.
            neutral = torch.sigmoid(andness * self._ANDNESS_SHARPNESS).to(dtype=values[0].dtype, device=values[0].device)
            gates = []
            for value, weight in zip(values, weights):
                if isinstance(weight, torch.Tensor):
                    gate = weight.to(device=value.device, dtype=value.dtype)
                else:
                    gate = torch.tensor(weight, device=value.device, dtype=value.dtype)
                gates.append(torch.clamp(torch.abs(gate), 0.0, 1.0))

            # Common fixed-weight case (0.5, 0.5, ...): do not bias inputs
            # toward the neutral baseline, otherwise the operator becomes less
            # expressive and can underfit simple boolean truth tables.
            if not all(torch.allclose(g, torch.full_like(g, 0.5), atol=1e-6, rtol=0.0) for g in gates):
                gated_values = []
                for value, gate in zip(values, gates):
                    # Gate each value toward the neutral baseline when weight is small.
                    gated_values.append(gate * value + (1 - gate) * neutral)
                values = gated_values

        X = torch.stack(values, dim=0)  # [N, ...]
        return self._MinMax(X, andness)

    def _MinMax(self, X, a):
        """Compute min/max interpolation over the first dimension.

        Args:
            X: Stacked tensor of shape ``[N, ...]``.
            a: Andness control value or tensor.

        Returns:
            Tensor reduced across the first dimension.
        """
        import torch
        try:
            epsilon = 1e-6
            X = torch.where(torch.isnan(X), torch.tensor(epsilon, device=X.device, dtype=X.dtype), X)
            X = torch.clamp(X, min=epsilon, max=1 - epsilon)
            if not isinstance(a, torch.Tensor):
                a = torch.tensor(a, dtype=X.dtype, device=X.device)
            # Straight-through estimator: hard routing in forward pass,
            # continuous surrogate for gradient flow.
            a_cont = torch.sigmoid(a * self._ANDNESS_SHARPNESS)
            a_hard = torch.round(a_cont)
            a = a_hard.detach() + (a_cont - a_cont.detach())
            mn = torch.min(X, dim=0).values
            mx = torch.max(X, dim=0).values
            return a * mn + (1 - a) * mx
        except Exception as e:
            print(f"[ERROR] Exception in MinMax_many: {e}")
            raise e