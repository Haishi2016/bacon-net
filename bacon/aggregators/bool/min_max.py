from bacon.aggregators.base import AggregatorBase

class MinMaxAggregator(AggregatorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)     
    
    def aggregate_float(self, a: float, b: float, r: float, w0: float, w1: float) -> float:
        pass

    def aggregate_tensor(self, x1, x2, andness, w0, w1):
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
        if x1 is None or x2 is None:
            raise ValueError(f"[ERROR] One of the inputs to generalized_gcd is None! x1={x1}, x2={x2}")
        if torch.isnan(x1).any() or torch.isnan(x2).any():
            raise ValueError(f"[ERROR] NaN input to generalized_gcd: x1={x1}, x2={x2}")
        return self._MinMax(x1, x2, andness)
    
    def _MinMax(self, x, y, a):
        import torch
        try:
            epsilon = 1e-6  # To prevent division by zero

            x = torch.where(torch.isnan(x), torch.tensor(epsilon, device=x.device), x)
            y = torch.where(torch.isnan(y), torch.tensor(epsilon, device=y.device), y)

            x = torch.clamp(x, min=epsilon, max=1-epsilon)
            y = torch.clamp(y, min=epsilon, max=1-epsilon)

            if not isinstance(a, torch.Tensor):
                a = torch.tensor(a, dtype=torch.float32)
            
            # STE trick
            a_cont = torch.sigmoid(a * 10)
            a_hard = torch.round(a_cont)
            a = a_hard.detach() + (a_cont - a_cont.detach())

            result = a * torch.minimum(x, y) + (1 - a) * torch.maximum(x, y)
            return result

        except Exception as e:
            print(f"[ERROR] Exception in MinMax: {e}")
            raise e