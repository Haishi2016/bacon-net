from bacon.aggregators.base import AggregatorBase

class MinMaxAggregator(AggregatorBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)     
    
    def aggregate_float(self, a: float, b: float, r: float, w0: float, w1: float) -> float:
        pass

    def aggregate_tensor(self, a, b, r, w0, w1):
        import torch
        if a is None or b is None:
            raise ValueError(f"[ERROR] One of the inputs to generalized_gcd is None! a={a}, b={b}")
        if torch.isnan(a).any() or torch.isnan(b).any():
            raise ValueError(f"[ERROR] NaN input to generalized_gcd: a={a}, b={b}")
        return self.MinMax(a, b, r, w0, w1)
    
    def MinMax(self, x, y, a, w0, w1):
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