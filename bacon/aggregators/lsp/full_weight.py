from bacon.aggregators.base import AggregatorBase
from typing import Sequence, Any


def lsp_power_mean(X, a, w_norm, eps: float = 1e-6):
    r"""Element-wise GL/LSP andness-parameterised power mean (N-ary).

    This is the head-vectorizable core of the LSP ``full_weight`` / ``half_weight``
    aggregators. It computes the same function as the original scalar branching
    implementation, but selects the andness regime with ``torch.where`` instead
    of Python ``if`` statements, so ``a`` (andness) may be a **tensor** of any
    shape broadcastable against the reduced inputs. This is what lets a single
    call evaluate many independent logic nodes / heads at once (each carrying its
    own andness), exactly mirroring the scalar tree node-by-node.

    Parameters
    ----------
    X : torch.Tensor
        Inputs stacked on dim 0: ``(N, ...)`` with values in ``[0, 1]``.
    a : float or torch.Tensor
        Andness in ``[-1, 2]``, broadcastable against ``X.sum(dim=0)``.
    w_norm : torch.Tensor
        Per-input weights summing to 1 over dim 0, broadcastable against ``X``.
    eps : float
        Numerical-stability clamp.

    Notes
    -----
    Regimes (identical to the paper / scalar code):

    * ``a in [0.5, 2]``  -> blend of weighted arithmetic mean ``A`` and the
      weighted geometric power term ``P = G ** (sqrt(3/(2-a)) - 1)``:
      ``(3-4a)A + (4a-2)P`` for ``a < 0.75`` and ``P`` for ``a >= 0.75``
      (continuous at ``a = 0.75``; reduces to ``A`` at ``a = 0.5``).
    * ``a in [-1, 0.5)`` -> De Morgan dual ``1 - F(1 - X, 1 - a)`` (which maps
      ``1 - a`` into ``[0.5, 2]``, so a single non-recursive evaluation suffices).
    """
    import torch
    X = torch.where(torch.isnan(X), torch.full_like(X, eps), X)
    X = torch.clamp(X, min=eps, max=1.0 - eps)
    if not torch.is_tensor(a):
        a = torch.as_tensor(a, dtype=X.dtype, device=X.device)
    a = torch.nan_to_num(a, nan=-1.0, posinf=2.0, neginf=-1.0).clamp(-1.0, 2.0)

    # Map the disjunctive regime (a < 0.5) to the conjunctive one via De Morgan.
    use_dual = a < 0.5
    a_eff = torch.where(use_dual, 1.0 - a, a)                     # in [0.5, 2]
    X_eff = torch.where(use_dual, 1.0 - X, X)                     # broadcasts on dim 0

    A = (w_norm * X_eff).sum(dim=0)                               # weighted arithmetic mean
    G = torch.pow(X_eff, 2.0 * w_norm).prod(dim=0)                # weighted geometric term
    denom = (2.0 - a_eff).clamp(min=eps)
    r = torch.sqrt(3.0 / denom) - 1.0
    P = G ** r
    blend = (3.0 - 4.0 * a_eff) * A + (4.0 * a_eff - 2.0) * P
    upper = torch.where(a_eff < 0.75, blend, P)
    return torch.where(use_dual, 1.0 - upper, upper)


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
            return lsp_power_mean(X, a, w_norm, eps=1e-6)
        except Exception as e:
            print(f"[ERROR] Exception in F_many: {e}")
            raise e