"""
XERV Crayon: Production-Grade Tokenizer.

A high-performance tokenizer achieving >2M tokens/s via AVX2 SIMD optimizations,
entropy-guided vocabulary construction, and cache-aligned data structures.

Top-level package exposing the primary public API.

Quick Start:
    >>> from crayon import CrayonVocab
    >>> 
    >>> # Option 1: Load from existing vocabulary
    >>> vocab = CrayonVocab(["hello", "world", "!"])
    >>> tokens = vocab.tokenize("hello world!")
    >>> 
    >>> # Option 2: Build from your corpus
    >>> vocab = CrayonVocab.from_corpus("Your training text here...")
    >>> 
    >>> # Option 3: Use batteries-included default sources
    >>> vocab = CrayonVocab.from_default_sources(vocab_size=50000)

Features:
    - SIMD-accelerated tokenization (AVX2)
    - Entropy-guided vocabulary construction
    - Zero-copy file processing
    - Pipeline parallelization
    - Adaptive vocabulary updates
"""

from .core.tokenizer import crayon_tokenize
from .core.vocabulary import CrayonVocab
from .concurrency.pipeline import PipelineTokenizer
from .memory.zerocopy import ZeroCopyTokenizer
from .training import train_vocabulary, build_default_vocabulary

__version__ = "1.1.0"
__author__ = "Xerv Research Engineering Division"

__all__ = [
    # Core
    "crayon_tokenize",
    "CrayonVocab",
    # Concurrency
    "PipelineTokenizer",
    # Memory
    "ZeroCopyTokenizer",
    # Training
    "train_vocabulary",
    "build_default_vocabulary",
]


def get_version() -> str:
    """Return the package version."""
    return __version__


def check_c_extension() -> bool:
    """
    Check if the C extension is available and working.
    
    Returns:
        True if C extension is loaded, False otherwise
    """
    try:
        from .c_ext import _core
        return hasattr(_core, 'build_trie') and hasattr(_core, 'crayon_tokenize_fast')
    except ImportError:
        return False


def check_resources() -> dict:
    """
    Check availability of optional resources for vocabulary building.
    
    Returns:
        Dict with availability status for each resource type
    """
    try:
        from .resources import check_resource_availability
        return check_resource_availability()
    except ImportError:
        return {
            "requests_available": False,
            "huggingface_available": False,
            "builtin_available": True
        }