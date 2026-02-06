from bacon.aggregators.base import AggregatorBase
from typing import Sequence, Any

class FullWeightAggregator(AggregatorBase):   
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)        

    def aggregate_float(self, values: Sequence[float], a: float, weights: Sequence[float]) -> float:
        import torch
        xs = torch.tensor(values, dtype=torch.float32)
        ws = torch.tensor(weights, dtype=torch.float32)
        a_t = torch.tensor(a, dtype=torch.float32)
        out = self.aggregate_tensor([xi for xi in xs], a_t, ws)
        return float(out.item())

    def aggregate_tensor(self, values: Sequence[Any], andness, weights):
        """
        Aggregates two tensors using the Full Weight method.

        Args:
            x1 (torch.Tensor): First input tensor.
            x2 (torch.Tensor): Second input tensor.
            andness (float): Andness.
            w0 (float): Weight for the first tensor.
            w1 (float): Weight for the second tensor.

        Returns:
            torch.Tensor: Resulting tensor after aggregation.
        """
        import torch
        if len(values) < 1:
            raise ValueError("aggregate_tensor: values must be non-empty")
        import torch
        X = torch.stack(values, dim=0)  # [N, ...]
        # Normalize and prepare weights
        if isinstance(weights, torch.Tensor):
            w = weights
        else:
            w = torch.stack([wi if isinstance(wi, torch.Tensor) else torch.as_tensor(wi, dtype=X.dtype, device=X.device) for wi in weights])
        if w.device != X.device:
            w = w.to(X.device, dtype=X.dtype)
        eps = torch.as_tensor(1e-8, dtype=X.dtype, device=X.device)
        w_sum = w.sum() + eps
        w_norm = w / w_sum
        # Broadcast weights to X
        while w_norm.dim() < X.dim():
            w_norm = w_norm.unsqueeze(-1)
        return self._F_many(X, andness, w_norm)
    
    def _F(self, x,y,a, w0, w1):
        import torch
        try:
            epsilon = 1e-6  # To prevent division by zero

            x = torch.where(torch.isnan(x), torch.tensor(epsilon, device=x.device), x)
            y = torch.where(torch.isnan(y), torch.tensor(epsilon, device=y.device), y)

            x = torch.clamp(x, min=epsilon, max=1-epsilon)
            y = torch.clamp(y, min=epsilon, max=1-epsilon)

            if not isinstance(a, torch.Tensor):
                a = torch.tensor(a, dtype=torch.float32)

            # a = a.clamp(-1.0 + epsilon, 2.0 - epsilon)  # avoid exact ends
            # a = torch.nan_to_num(a, nan=-1.0, posinf=2.0-epsilon, neginf=-1.0+epsilon)
            # a = a.clamp(-1.0 + epsilon, 2.0 - epsilon)  # avoid exact ends
            # if a == 2, return 1 of x==y==1, otherwise 0
            if torch.any(torch.abs(a - 2) < epsilon):
                cond = torch.logical_and(torch.abs(x - 1) < epsilon, torch.abs(y - 1) < epsilon)
                result = torch.where(cond, torch.ones_like(x), torch.zeros_like(x))
                if torch.isnan(result).any():
                    print(f"[TRACE] Rule 0 result has NaN: {torch.isnan(result).any()}")
                return result

            # if 1.25 < a < 2, return (xy)^(sqrt(3/(2-a))-1)
            elif torch.logical_and(a >= 0.75, a < 2):
                result = (x ** (2*w0) * y ** (2*w1)) ** (torch.sqrt(3 / (2 - a)) - 1)
                if torch.isnan(result).any():
                    print(f"[TRACE] Rule 1 result has NaN: {torch.isnan(result).any()}")
                
                return result
            
            # 1/2 < a < 3/4 return (3-4a)(0.5x+0.5y) + (4a-2)(0.5x^R+0.5y^R)^1/R
            elif torch.logical_and(a > 0.5, a < 0.75):
                result = (3-4*a)*(w0*x+w1*y) + (4*a-2)*(x ** (2*w0) * y ** (2*w1)) ** (torch.sqrt(3 / (2 - a)) - 1)
                if torch.isnan(result).any():
                    result = torch.where(torch.isnan(result), torch.tensor(float('inf'), device=result.device), result)
                    print(f"[TRACE] Rule 6 result has NaN: {torch.isnan(result).any()} scalar_a={a} x={x} y={y}")
                return result
            
            # a == 0.5 return 0.5x+0.5y
            elif torch.any(torch.abs(a - 0.5) < epsilon):
                result = w0*x + w1*y
                if torch.isnan(result).any():
                    print(f"[TRACE] Rule 7 result has NaN: {torch.isnan(result).any()}")
                return result

            # -1 <= a < 0.5 return 1-F(1-x,1-y,1-a)
            elif torch.logical_and(a >= -1, a < 0.5):
                result = 1 - self._F(1-x, 1-y, (1-a).clamp(-1.0 + epsilon, 2.0 - epsilon), w0, w1)
                #result = 1 - self.F(1-x, 1-y, 1-a, w0, w1)
                if torch.isnan(result).any():
                    print(f"[TRACE] Rule 8 result has NaN: {torch.isnan(result).any()}")
                return result
            else:
                raise ValueError(f"Invalid value for a: {a}. Must be in [-1, 2].")
        except Exception as e:
            print(f"[ERROR] Exception in F: {e}")
            print(f"[DEBUG] x: {x}, y: {y}, a: {a}, w0: {w0}, w1: {w1}")
            raise e
    def _F_many(self, X, a, w_norm):
        import torch
        try:
            epsilon = torch.as_tensor(1e-6, dtype=X.dtype, device=X.device)
            # Clamp and sanitize
            X = torch.where(torch.isnan(X), epsilon, X)
            X = torch.clamp(X, min=epsilon.item(), max=1 - epsilon.item())
            if not torch.is_tensor(a):
                a = torch.tensor(a, dtype=X.dtype, device=X.device)
            # Note: FullWeight version kept a checks slightly different originally;
            # preserve core branches analogous to half-weight for N-ary generalization.
            x0 = X.select(dim=0, index=0)
            # a == 2
            if torch.any(torch.abs(a - 2) < epsilon):
                all_ones = torch.all(torch.abs(X - 1) < epsilon, dim=0)
                return torch.where(all_ones, torch.ones_like(x0), torch.zeros_like(x0))
            # Weighted arithmetic mean
            A = (w_norm * X).sum(dim=0)
            # Weighted geometric term
            geo_exp = 2.0 * w_norm
            G = torch.pow(X, geo_exp).prod(dim=0)
            # 0.75 <= a < 2
            if torch.logical_and(a >= 0.75, a < 2):
                return G ** (torch.sqrt(torch.as_tensor(3.0, dtype=X.dtype, device=X.device) / (2.0 - a)) - 1.0)
            # 0.5 < a < 0.75
            if torch.logical_and(a > 0.5, a < 0.75):
                return (3.0 - 4.0 * a) * A + (4.0 * a - 2.0) * (G ** (torch.sqrt(torch.as_tensor(3.0, dtype=X.dtype, device=X.device) / (2.0 - a)) - 1.0))
            # a == 0.5
            if torch.any(torch.abs(a - 0.5) < epsilon):
                return A
            # -1 <= a < 0.5
            if torch.logical_and(a >= -1, a < 0.5):
                return 1.0 - self._F_many(1.0 - X, (1.0 - a), w_norm)
            raise ValueError(f"Invalid value for a: {a}. Must be in [-1, 2].")
        except Exception as e:
            print(f"[ERROR] Exception in F_many: {e}")
            raise e