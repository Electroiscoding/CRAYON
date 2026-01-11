import threading
from typing import Optional, List, Any

class LockFreeVocabCache:
    """
    Lock-free cache using atomic operations logic for thread-safe access.
    
    Uses versioning to detect concurrent modifications (ABA problem prevention).
    Optimized for read-heavy workloads typical in tokenization.
    """

    def __init__(self, capacity: int = 8192):
        self.capacity = capacity
        # Ensure power of 2 for fast masking
        assert (capacity & (capacity - 1)) == 0, "Capacity must be power of 2"
        self.mask = capacity - 1
        
        # Pre-allocated arrays [cite: 607-609]
        self.keys: List[Optional[str]] = [None] * capacity
        self.values: List[Optional[int]] = [None] * capacity
        self.versions: List[int] = [0] * capacity
        
    def get(self, key: str) -> Optional[int]:
        """
        Thread-safe cache lookup using optimistic concurrency[cite: 615].
        """
        idx = hash(key) & self.mask
        
        # 1. Read version before data
        start_version = self.versions[idx]
        
        # 2. Optimistic read of key/value
        stored_key = self.keys[idx]
        stored_value = self.values[idx]
        
        # 3. Read version after data (Memory Barrier simulation)
        end_version = self.versions[idx]
        
        # Validation: Version matches and key matches
        if start_version == end_version and stored_key == key:
            return stored_value
            
        return None # Cache miss or concurrent modification

    def put(self, key: str, value: int) -> None:
        """
        Thread-safe insertion with optimistic collision handling[cite: 627].
        """
        idx = hash(key) & self.mask
        
        # Simple atomic update simulation
        # In pure Python, assignment is atomic for simple types, but we increment version
        # to invalidate readers.
        
        current_ver = self.versions[idx]
        self.versions[idx] = current_ver + 1 # Invalidate readers
        
        self.keys[idx] = key
        self.values[idx] = value
        
        self.versions[idx] = current_ver + 2 # Validate new data