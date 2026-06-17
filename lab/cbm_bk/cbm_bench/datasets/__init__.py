"""Dataset loaders for CBM benchmarks.

Each dataset module registers a factory via ``@register_dataset`` and returns a
:class:`DatasetBundle` describing train/val/test splits plus concept/class
metadata. Import sub-modules here so registration happens on package import.
"""

from .base import DatasetBundle
from . import cub  # noqa: F401  (import for side-effect: registration)

__all__ = ["DatasetBundle"]
