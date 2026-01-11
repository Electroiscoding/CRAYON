"""
Crayon Unicode Processing Module.

Implements the high-performance text normalization and multilingual support
strategies defined in Section 5 of the XERV Crayon Engineering Treatise.
"""

from .normalizer import unicode_normalize_nfc_optimized
from .multilingual import MultilingualProcessor

__all__ = ["unicode_normalize_nfc_optimized", "MultilingualProcessor"]