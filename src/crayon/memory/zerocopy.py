# `ZeroCopyTokenizer` using `mmap`
import mmap

class ZeroCopyTokenizer:
    """Tokenizer implementation using zero-copy mmap."""
    def __init__(self, filename):
        with open(filename, "r+b") as f:
            self.mm = mmap.mmap(f.fileno(), 0)
