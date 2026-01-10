# `MemoryPool` with weakref tracking
import weakref

class MemoryPool:
    """Memory pool with weakref tracking."""
    def __init__(self):
        self._pool = weakref.WeakValueDictionary()
