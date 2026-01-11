import threading
from typing import List, Optional
from ..core.vocabulary import CrayonVocab
from ..memory.cache import LockFreeVocabCache

class ThreadLocalTokenizer:
    """
    Thread-Local tokenization state to minimize cross-thread coordination.
    
    Maintains separate caches and buffers for each thread to avoid
    LOCK contention and False Sharing[cite: 639].
    """

    def __init__(self, global_vocab: CrayonVocab):
        self.global_vocab = global_vocab
        self._local = threading.local()

    @property
    def local_state(self):
        """Lazy initialization of thread-local resources[cite: 647]."""
        if not hasattr(self._local, 'initialized'):
            # L1 Cache specific to this thread (2048 entries)
            self._local.cache = LockFreeVocabCache(capacity=2048)
            # Reusable buffer to prevent allocation churn
            self._local.temp_buffer = bytearray(65536) 
            self._local.result_buffer = [] 
            self._local.initialized = True
        return self._local

    def tokenize_thread_safe(self, text: str) -> List[int]:
        """
        Thread-safe tokenization with minimal synchronization overhead.
        
        Strategy:
        1. Try thread-local L1 cache.
        2. Fallback to global vocabulary (which releases GIL in C-ext).
        """
        state = self.local_state
        cache = state.cache
        result = state.result_buffer
        result.clear()
        
        position = 0
        text_len = len(text)
        
        while position < text_len:
            # Check cache for common tokens first (Optimistic read)
            # Note: A real implementation might cache substrings at 'position'
            # Here we simplify to illustrate the pattern
            
            # Fallback to global with GIL release (simulated here via method call)
            # In C-extension, this call releases the GIL [cite: 590]
            token_id, match_len = self.global_vocab.longest_match(text, position)
            
            if match_len > 0:
                result.append(token_id)
                # Update local cache for next time
                # cache.put(substring, token_id) 
                position += match_len
            else:
                result.append(self.global_vocab.unk_token_id)
                position += 1
                
        # Return a copy, keeping the buffer for next run
        return list(result)