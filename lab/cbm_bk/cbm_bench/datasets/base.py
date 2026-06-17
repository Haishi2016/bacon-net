"""Common dataset abstractions shared by all CBM datasets.

A CBM dataset must yield ``(image, concepts, label)`` tuples where:
- ``image``    : float tensor ``(C, H, W)``
- ``concepts`` : float tensor ``(n_concepts,)`` with values in ``[0, 1]``
- ``label``    : long scalar class index
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from torch.utils.data import Dataset


@dataclass
class DatasetBundle:
    """Container returned by every dataset factory.

    Holds the concrete torch ``Dataset`` splits along with the metadata the
    model layer needs (number of concepts and classes). ``val`` may be ``None``
    for datasets that only define train/test.
    """

    name: str
    train: Dataset
    test: Dataset
    n_concepts: int
    n_classes: int
    val: Optional[Dataset] = None
    concept_names: Optional[list[str]] = None
    class_names: Optional[list[str]] = None
