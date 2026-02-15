from bacon.aggregators.base import AggregatorBase
from typing import Sequence, Any

class MinMaxAggregator(AggregatorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)     
    
    def aggregate_float(self, values: Sequence[float], a: float, weights: Sequence[float]) -> float:
        import torch
        xs = torch.tensor(values, dtype=torch.float32)
        a_t = torch.tensor(a, dtype=torch.float32)
        out = self.aggregate_tensor([xi for xi in xs], a_t, weights=None)
        return float(out.item())

    def aggregate_tensor(self, values: Sequence[Any], andness, weights=None):
        """     
        Aggregates two tensors using the Min-Max method.

        Args:
            x1 (torch.Tensor): First input tensor.
            x2 (torch.Tensor): Second input tensor.
            andness (float): Andness.
            w0 (float): Not used in this implementation, but kept for compatibility.
            w1 (float): Not used in this implementation, but kept for compatibility.

        Returns:
            torch.Tensor: Resulting tensor after aggregation.
        """
        import torch
        if len(values) == 0:
            raise ValueError("aggregate_tensor: values must be non-empty")
        X = torch.stack(values, dim=0)  # [N, ...]
        return self._MinMax(X, andness)

    def _MinMax(self, X, a):
        import torch
        try:
            epsilon = 1e-6
            X = torch.where(torch.isnan(X), torch.tensor(epsilon, device=X.device, dtype=X.dtype), X)
            X = torch.clamp(X, min=epsilon, max=1 - epsilon)
            if not isinstance(a, torch.Tensor):
                a = torch.tensor(a, dtype=X.dtype, device=X.device)
            # STE trick reused
            a_cont = torch.sigmoid(a * 10)
            a_hard = torch.round(a_cont)
            a = a_hard.detach() + (a_cont - a_cont.detach())
            mn = torch.min(X, dim=0).values
            mx = torch.max(X, dim=0).values
            return a * mn + (1 - a) * mx
        except Exception as e:
            print(f"[ERROR] Exception in MinMax_many: {e}")
            raise e