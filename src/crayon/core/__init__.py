"""
Crayon Core Module.

Contains the fundamental algorithms and data structures for tokenization:
1. Tokenizer (The algorithmic driver)
2. Vocabulary (The data structure)
3. Primitives (Metadata structures)
"""

from .tokenizer import crayon_tokenize
from .vocabulary import CrayonVocab
from .primitives import TokenMetadata

__all__ = ["crayon_tokenize", "CrayonVocab", "TokenMetadata"]