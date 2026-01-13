"""
Crayon Adaptive Module.

Implements vocabulary adaptation and stability management from Section 8
of the XERV Crayon Engineering Treatise.

Components:
- StableVocabularyManager: Deterministic ID assignment with reserved ranges
- AdaptiveVocabularyManager: Real-time vocabulary adaptation
- IncrementalVocabularyUpdater: Staged updates with rollback capability
"""

from .stability import StableVocabularyManager, TokenCategory, TokenMetadata
from .manager import AdaptiveVocabularyManager
from .updater import IncrementalVocabularyUpdater

__all__ = [
    "StableVocabularyManager",
    "TokenCategory",
    "TokenMetadata",
    "AdaptiveVocabularyManager",
    "IncrementalVocabularyUpdater",
]