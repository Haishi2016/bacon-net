__version__ = "0.3.3"

# Public API exports
from .baconNet import baconNet
from .binaryTreeLogicNet import binaryTreeLogicNet

from .aggregators.lsp import FullWeightAggregator, HalfWeightAggregator
from .aggregators.bool import MinMaxAggregator

__all__ = [
    "__version__",
    "baconNet",
    "binaryTreeLogicNet",
    "FullWeightAggregator",
    "HalfWeightAggregator",
    "MinMaxAggregator",    
]
