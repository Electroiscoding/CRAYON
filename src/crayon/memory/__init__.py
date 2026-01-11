"""
Crayon Memory Management Module.

Implements Zero-Copy and Pooling strategies defined in Section 7.3:
1. ZeroCopyTokenizer (Memory mapped file processing)
2. MemoryPool (Buffer recycling)
3. LockFreeCache (Thread-safe lookup)
"""

from .pool import MemoryPool
from .zerocopy import ZeroCopyTokenizer
from .cache import LockFreeVocabCache

__all__ = ["MemoryPool", "ZeroCopyTokenizer", "LockFreeVocabCache"]