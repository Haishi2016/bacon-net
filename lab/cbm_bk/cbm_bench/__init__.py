"""cbm_bench: a small, extensible benchmarking harness for Concept Bottleneck
Models (CBMs) across multiple datasets.

The package is organized so that *datasets* and *models* are decoupled and
registered into lookup tables. A single training / evaluation driver works for
any (dataset, model) pair that has been registered.

Layout
------
- ``cbm_bench.datasets``  : dataset loaders, each registered by name.
- ``cbm_bench.models``    : CBM model variants, each registered by name.
- ``cbm_bench.config``    : YAML config loading + experiment dataclasses.
- ``cbm_bench.metrics``   : concept / task accuracy helpers.

Add a new dataset or model by dropping a module in the relevant sub-package and
decorating its factory with ``@register_dataset`` / ``@register_model``.
"""

from .registry import (
    register_dataset,
    register_model,
    build_dataset,
    build_model,
    list_datasets,
    list_models,
)

# Import sub-packages for their registration side-effects so that datasets and
# models are available as soon as ``cbm_bench`` is imported.
from . import datasets  # noqa: F401,E402
from . import models  # noqa: F401,E402

__all__ = [
    "register_dataset",
    "register_model",
    "build_dataset",
    "build_model",
    "list_datasets",
    "list_models",
]
