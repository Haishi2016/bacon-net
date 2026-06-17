"""CBM model variants.

Import sub-modules here so their ``@register_model`` decorators run on package
import.
"""

from .base import CBMOutput
from . import baseline_cbm  # noqa: F401  (import for side-effect: registration)
from . import bacon_cbm  # noqa: F401  (import for side-effect: registration)
from . import cibm_cbm  # noqa: F401  (import for side-effect: registration)

__all__ = ["CBMOutput"]
