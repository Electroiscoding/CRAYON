"""
Crayon Adaptive Vocabulary Module.

This module implements the self-evolving vocabulary capabilities described in
Section 8 of the XERV Crayon Engineering Treatise. It handles:
1. Stable ID Assignment (Deterministic sorting & ranges)
2. Real-time Adaptation (Entropy-guided candidate selection)
3. Incremental Updates (Transactional Staging/Commit/Rollback)
"""

from .stability import StableVocabularyManager, TokenMetadata
from .manager import AdaptiveVocabularyManager
from .updater import IncrementalVocabularyUpdater

__all__ = [
    "StableVocabularyManager",
    "TokenMetadata",
    "AdaptiveVocabularyManager",
    "IncrementalVocabularyUpdater",
]