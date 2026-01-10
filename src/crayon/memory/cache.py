# `LockFreeVocabCache` (L1/L2 software cache)

class LockFreeVocabCache:
    """Software cache L1/L2."""
    def __init__(self):
        self.l1_cache = {}
        self.l2_cache = {}
