"""
Crayon Core Module.

Contains the fundamental algorithms and data structures for tokenization:
1. Tokenizer (The algorithmic driver)
2. Vocabulary (The data structure)
3. Primitives (Metadata structures)
4. Vocab Builder (Entropy-guided construction)
"""

from .tokenizer import crayon_tokenize
from .vocabulary import CrayonVocab
from .primitives import TokenMetadata
from .vocab_builder import (
    EntropyVocabBuilder,
    construct_optimal_vocabulary,
    deterministic_sort_key,
    assign_stable_ids
)

__all__ = [
    "crayon_tokenize",
    "CrayonVocab",
    "TokenMetadata",
    "EntropyVocabBuilder",
    "construct_optimal_vocabulary",
    "deterministic_sort_key",
    "assign_stable_ids"
]