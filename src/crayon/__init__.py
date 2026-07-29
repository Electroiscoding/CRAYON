"""
XERV Crayon: Production-Grade Omni-Backend Tokenizer
=====================================================

A high-performance tokenizer achieving >2M tokens/s via:
- AVX2/AVX-512 SIMD optimizations (CPU)
- NVIDIA CUDA kernels (GPU)
- AMD ROCm/HIP kernels (GPU)
- Entropy-guided vocabulary construction
- Cache-aligned Double-Array Trie data structures

Quick Start:
    >>> from crayon import CrayonVocab
    >>> 
    >>> # Auto-detect best device (GPU if available, else CPU)
    >>> vocab = CrayonVocab(device="auto")
    >>> vocab.load_profile("lite")
    >>> tokens = vocab.tokenize("Hello, world!")
    >>> 
    >>> # Batch processing
    >>> batch_tokens = vocab.tokenize(["text 1", "text 2", "text 3"])
    >>> 
    >>> # Decode back to text
    >>> text = vocab.decode(tokens)

Device Selection:
    >>> vocab = CrayonVocab(device="cpu")   # Force CPU (lowest latency)
    >>> vocab = CrayonVocab(device="cuda")  # Force NVIDIA GPU
    >>> vocab = CrayonVocab(device="rocm")  # Force AMD GPU
    >>> vocab = CrayonVocab(device="auto")  # Auto-detect best

Profile Management:
    >>> vocab.load_profile("lite")      # General purpose
    >>> vocab.load_profile("standard")  # Larger general purpose
    >>> 
    >>> # Context manager for temporary switch
    >>> with vocab.using_profile("standard"):
    ...     tokens = vocab.tokenize(source_code)

Environment Variables:
    CRAYON_DEVICE: Override device selection (cpu|cuda|rocm)
    CRAYON_PROFILE_DIR: Custom profile search directory
"""

from __future__ import annotations

__version__ = "5.5.9"
__author__ = "Xerv Research Engineering Division"

# ============================================================================
# CORE IMPORTS
# ============================================================================

from .core.tokenizer import crayon_tokenize
from .core.vocabulary import (
    CrayonVocab,
    DeviceType,
    DeviceState,
    HardwareInfo,
    quick_tokenize,
    enable_verbose_logging,
    disable_verbose_logging,
)

# ============================================================================
# OPTIONAL IMPORTS (May not be available in minimal installs)
# ============================================================================

try:
    from .concurrency.pipeline import PipelineTokenizer
except ImportError:
    PipelineTokenizer = None  # type: ignore

try:
    from .memory.zerocopy import ZeroCopyTokenizer
except ImportError:
    ZeroCopyTokenizer = None  # type: ignore

try:
    from .training import train_vocabulary, build_default_vocabulary
except ImportError:
    train_vocabulary = None  # type: ignore
    build_default_vocabulary = None  # type: ignore


# ============================================================================
# BACKEND UTILITIES
# ============================================================================

def get_version() -> str:
    """Return the package version string."""
    return __version__


def check_c_extension() -> bool:
    """
    Check if the core C extension is available.
    
    Returns:
        True if crayon_cpu extension is loaded and functional.
    """
    try:
        from .c_ext import crayon_cpu
        return hasattr(crayon_cpu, 'tokenize') and hasattr(crayon_cpu, 'load_dat')
    except ImportError:
        return False


def check_backends() -> dict:
    """
    Check availability of all backends.
    
    Returns:
        Dictionary with status for cpu, cuda, and rocm backends.
        
    Example:
        >>> from crayon import check_backends
        >>> backends = check_backends()
        >>> print(backends)
        {'cpu': True, 'cuda': True, 'rocm': False}
    """
    try:
        from .c_ext import is_cuda_available, is_rocm_available
        return {
            "cpu": check_c_extension(),
            "cuda": is_cuda_available(),
            "rocm": is_rocm_available(),
        }
    except ImportError:
        return {
            "cpu": check_c_extension(),
            "cuda": False,
            "rocm": False,
        }


def get_backend_info() -> dict:
    """
    Get detailed information about all backends.
    
    Returns:
        Dictionary with availability, hardware info, and errors for each backend.
    """
    try:
        from .c_ext import get_backend_info as _get_backend_info
        return _get_backend_info()
    except ImportError:
        return {"cpu": {"available": check_c_extension()}}


def check_resources() -> dict:
    """
    Check availability of optional resources for vocabulary building.
    
    Returns:
        Dictionary with availability status for each resource type.
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


# ============================================================================
# PUBLIC API
# ============================================================================

__all__ = [
    # Version
    "__version__",
    "__author__",
    "get_version",
    
    # Core
    "CrayonVocab",
    "crayon_tokenize",
    "quick_tokenize",
    "DeviceType",
    "DeviceState",
    "HardwareInfo",
    
    # Logging
    "enable_verbose_logging",
    "disable_verbose_logging",
    
    # Backend checks
    "check_c_extension",
    "check_backends",
    "get_backend_info",
    "check_resources",
    
    # Optional modules (may be None)
    "PipelineTokenizer",
    "ZeroCopyTokenizer",
    "train_vocabulary",
    "build_default_vocabulary",
]