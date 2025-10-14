__version__ = "0.3.0"

# Public API exports
from .baconNet import baconNet
from .binaryTreeLogicNet import binaryTreeLogicNet
from .vectorLogicNet import VectorLogicNet

from .aggregators.lsp import FullWeightAggregator, HalfWeightAggregator
from .aggregators.bool import MinMaxAggregator

__all__ = [
    "__version__",
    "baconNet",
    "binaryTreeLogicNet",
    "VectorLogicNet",
    "FullWeightAggregator",
    "HalfWeightAggregator",
    "MinMaxAggregator",
]
