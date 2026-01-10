# `ThreadLocalTokenizer` state isolation
import threading

class ThreadLocalTokenizer:
    """Thread-local state for tokenizer."""
    _local = threading.local()

    def get_tokenizer(self):
        if not hasattr(self._local, "tokenizer"):
            self._local.tokenizer = None # Initialize
        return self._local.tokenizer
