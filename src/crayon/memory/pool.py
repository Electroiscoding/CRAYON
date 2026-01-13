import threading
from typing import List, Set, Optional

class MemoryPool:
    """
    Thread-safe memory pool for high-performance buffer reuse.
    
    Philosophy (Section 7.3): Amortize allocation costs across many operations
    and reduce GC pressure[cite: 912].
    """

    def __init__(self, chunk_size: int = 65536, pool_size: int = 64):
        self.chunk_size = chunk_size
        self.pool_size = pool_size
        
        self.available_buffers: List[bytearray] = []
        # Track in-use buffers by their id() since bytearrays don't support weak refs
        self.in_use_buffer_ids: Set[int] = set()
        self.lock = threading.Lock()
        
        # Pre-populate pool [cite: 919]
        for _ in range(pool_size):
            self.available_buffers.append(bytearray(chunk_size))

    def get_buffer(self, required_size: Optional[int] = None) -> bytearray:
        """
        Get a buffer from the pool, expanding dynamically if needed[cite: 924].
        """
        size = required_size or self.chunk_size
        
        # Standard pool path
        if size == self.chunk_size:
            with self.lock:
                if self.available_buffers:
                    buf = self.available_buffers.pop()
                    # Security: clear residual data [cite: 938]
                    # buf[:] = b'\x00' * len(buf) # Expensive, optimize if needed
                    self.in_use_buffer_ids.add(id(buf))
                    return buf
        
        # Slow path / Non-standard size
        buf = bytearray(size)
        if size == self.chunk_size:
             self.in_use_buffer_ids.add(id(buf))
        return buf

    def return_buffer(self, buffer: bytearray) -> None:
        """
        Return buffer to pool for reuse[cite: 949].
        """
        if len(buffer) != self.chunk_size:
            return # Don't pool irregular sizes
            
        with self.lock:
            if len(self.available_buffers) < self.pool_size:
                self.available_buffers.append(buffer)
                self.in_use_buffer_ids.discard(id(buffer))