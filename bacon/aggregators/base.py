from abc import ABC, abstractmethod
from typing import Sequence, Any

class AggregatorBase(ABC):
    def aggregate(self, values, a, weights):
        """
        Dispatch aggregation for either Python floats or torch.Tensors.
        Expects:
          - values: Sequence[float] or Sequence[Tensor]
          - a: float or Tensor
          - weights: Sequence[float] or Sequence[Tensor] or Tensor
        """
        if not isinstance(values, (list, tuple)):
            raise TypeError("values must be a list/tuple of scalars or tensors")
        if len(values) == 0:
            raise ValueError("values must be non-empty")
        first = values[0]
        # Float path
        if isinstance(first, (float, int)):
            return self.aggregate_float(values, a, weights)
        # Tensor path
        try:
            import torch  # type: ignore
            if isinstance(first, torch.Tensor):
                return self.aggregate_tensor(values, a, weights)
        except ImportError:
            pass
        raise TypeError("Unsupported input type for values")

    @abstractmethod
    def aggregate_float(self, values: Sequence[float], a: float, weights: Sequence[float]) -> float:
        """
        Aggregate N scalar values with scalar andness a and N weights.
        """
        pass

    @abstractmethod
    def aggregate_tensor(self, values: Sequence["torch.Tensor"], a: "torch.Tensor", weights: Sequence["torch.Tensor"] | "torch.Tensor") -> "torch.Tensor":
        """
        Aggregate N tensors with andness a and N weights.
        """
        pass
