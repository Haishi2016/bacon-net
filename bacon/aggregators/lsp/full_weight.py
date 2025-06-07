from bacon.aggregators.base import AggregatorBase

class FullWeightAggregator(AggregatorBase):   
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)        

    def aggregate_float(self, a: float, b: float, r: float, w0: float, w1: float) -> float:
        pass

    def aggregate_tensor(self, x1, x2, andness, w0, w1):
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
        if x1 is None or x2 is None:
            raise ValueError(f"[ERROR] One of the inputs to generalized_gcd is None! x1={x1}, x2={x2}")
        if torch.isnan(x1).any() or torch.isnan(x2).any():
            raise ValueError(f"[ERROR] NaN input to generalized_gcd: x1={x1}, x2={x2}")
        return self._F(x1, x2, andness, w0, w1)
    
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