from abc import ABC, abstractmethod

class AggregatorBase(ABC):
    def aggregate(self, x, y, a, w0, w1):
        if isinstance(x, float) and isinstance(y, float):
            return self.aggregate_float(x,y,a, w0, w1)
        else:
            try:
                import torch
                if isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
                    return self.aggregate_tensor(x,y,a, w0,w1)
            except ImportError:
                pass
        raise TypeError("Unsupported input type or missing torch")

    @abstractmethod
    def aggregate_float(self, x: float, y: float, a: float, w0: float, w1: float) -> float:
        pass

    @abstractmethod
    def aggregate_tensor(self, x: "torch.Tensor", y: "torch.Tensor", a: "torch.Tensor", w0: "torch.Tensor", w1: "torch.Tensor"):
        pass
