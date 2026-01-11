from typing import List
from .vocabulary import CrayonVocab

# Try importing C-extension
try:
    from ..c_ext import _core
    _C_EXT_AVAILABLE = True
except ImportError:
    _C_EXT_AVAILABLE = False

def crayon_tokenize(text: str, vocab: CrayonVocab) -> List[int]:
    """
    Core tokenization algorithm optimized for throughput and accuracy.
    
    Time Complexity: O(n) due to O(1) average lookup and constant max_lookahead.
    Space Complexity: O(n) for output tokens.
    
    Section 6.1: Specialized for Python 3.12+ [cite: 358-375].
    """
    # 1. Fast Path: Use C-Extension if available [cite: 479]
    if _C_EXT_AVAILABLE and hasattr(vocab, '_c_trie'):
        return _core.crayon_tokenize_fast(text, len(text), vocab._c_trie, vocab.unk_token_id)

    # 2. Slow Path: Pure Python Implementation (Fallback)
    # Optimized using local variables for loop speed
    tokens: List[int] = []
    position: int = 0
    text_length: int = len(text)
    
    # Pre-fetch methods to avoid attribute lookup in loop
    vocab_match = vocab.longest_match
    tokens_append = tokens.append
    unk_id = vocab.unk_token_id
    
    while position < text_length:
        # Longest matching token using optimized trie traversal
        token_id, match_length = vocab_match(text, position)
        
        if match_length > 0:
            tokens_append(token_id)
            position += match_length
        else:
            # Handle out-of-vocabulary characters
            tokens_append(unk_id)
            position += 1
            
    return tokens