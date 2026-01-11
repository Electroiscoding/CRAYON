"""
XERV Crayon: Production-Grade Tokenizer.

Top-level package exposing the primary public API.
"""

from .core.tokenizer import crayon_tokenize
from .core.vocabulary import CrayonVocab
from .concurrency.pipeline import PipelineTokenizer
from .memory.zerocopy import ZeroCopyTokenizer

__version__ = "1.0.0"
__author__ = "Xerv Research Engineering Division"

__all__ = [
    "crayon_tokenize",
    "CrayonVocab",
    "PipelineTokenizer",
    "ZeroCopyTokenizer"
]