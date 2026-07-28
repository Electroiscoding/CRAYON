"""
XERV CRAYON C-Extensions Package
================================

This package contains the native C/C++/CUDA extensions:

- crayon_cpu: AVX2/AVX-512 accelerated CPU tokenizer (always available)
- crayon_cuda: NVIDIA CUDA GPU tokenizer (optional, requires nvcc)
- crayon_rocm: AMD ROCm GPU tokenizer (optional, requires hipcc)

Import Behavior:
    - crayon_cpu is imported eagerly and will raise ImportError if missing
    - crayon_cuda and crayon_rocm are lazy-loaded to avoid import errors
    - Use check_* functions to safely probe availability

Example:
    >>> from crayon.c_ext import crayon_cpu
    >>> from crayon.c_ext import is_cuda_available, is_rocm_available
    >>> 
    >>> if is_cuda_available():
    ...     from crayon.c_ext import crayon_cuda
"""

import sys
from typing import Optional, Tuple

# ============================================================================
# CPU BACKEND (Required - Lazy Import to avoid circular dependencies)
# ============================================================================

_cpu_module: Optional[object] = None
_cpu_checked: bool = False
_cpu_error: Optional[str] = None


def _load_cpu_backend() -> Optional[object]:
    """Internal function to load the CPU backend."""
    global _cpu_checked, _cpu_module, _cpu_error
    
    if _cpu_checked:
        return _cpu_module
    
    _cpu_checked = True
    try:
        # Use absolute import to avoid circular dependency issues
        import crayon.c_ext.crayon_cpu as _cpu
        # Verify it's functional
        if hasattr(_cpu, 'tokenize') and hasattr(_cpu, 'load_dat'):
            _cpu_module = _cpu
            return _cpu_module
        else:
            _cpu_error = "crayon_cpu module missing required functions (tokenize, load_dat)"
            return None
    except ImportError as e:
        _cpu_error = (
            f"Failed to import crayon_cpu extension. {e}\n"
            "Possible causes:\n"
            "  1. The package was not installed correctly (try: pip install --force-reinstall xerv-crayon)\n"
            "  2. The C++ extension failed to compile (check for compiler errors during install)\n"
            "  3. Python version mismatch (Crayon requires Python 3.10+)"
        )
        return None
    except Exception as e:
        _cpu_error = f"Unexpected error loading crayon_cpu: {e}"
        return None


def get_cpu_backend() -> Optional[object]:
    """Get the CPU backend module, loading it if necessary."""
    return _load_cpu_backend()


def is_cpu_available() -> bool:
    """Check if the CPU backend is available."""
    return _load_cpu_backend() is not None


def get_cpu_error() -> Optional[str]:
    """Get the error message if CPU backend is unavailable."""
    _load_cpu_backend()  # Ensure check has run
    return _cpu_error


# Create a proxy object for backward compatibility
class _CPUProxy:
    """Proxy object that lazily loads crayon_cpu when accessed."""
    
    def __getattr__(self, name):
        cpu_module = _load_cpu_backend()
        if cpu_module is None:
            raise ImportError(f"CPU backend not available: {get_cpu_error()}")
        return getattr(cpu_module, name)
    
    def __dir__(self):
        cpu_module = _load_cpu_backend()
        if cpu_module is None:
            return []
        return dir(cpu_module)


# Create the proxy instance
crayon_cpu = _CPUProxy()


# ============================================================================
# TURBO BACKEND (Preferred over CPU when available)
# ============================================================================

_turbo_module: Optional[object] = None
_turbo_checked: bool = False
_turbo_error: Optional[str] = None


def _load_turbo_backend() -> Optional[object]:
    """Load the Turbo backend (>=105M tok/s engine)."""
    global _turbo_checked, _turbo_module, _turbo_error
    
    if _turbo_checked:
        return _turbo_module
    
    _turbo_checked = True
    try:
        import crayon.c_ext.crayon_turbo as _turbo
        if hasattr(_turbo, 'tokenize') and hasattr(_turbo, 'load_dat'):
            _turbo_module = _turbo
            return _turbo_module
        else:
            _turbo_error = "crayon_turbo module missing required functions"
            return None
    except ImportError as e:
        _turbo_error = f"Turbo engine not compiled: {e}"
        return None
    except Exception as e:
        _turbo_error = f"Turbo engine error: {e}"
        return None


def is_turbo_available() -> bool:
    """Check if the Turbo backend is available."""
    return _load_turbo_backend() is not None


def get_turbo_backend() -> Optional[object]:
    """Get the Turbo backend module."""
    return _load_turbo_backend()


def get_turbo_error() -> Optional[str]:
    """Get error message if Turbo is unavailable."""
    _load_turbo_backend()
    return _turbo_error



# ============================================================================
# GPU BACKENDS (Optional - Lazy Import)
# ============================================================================

_cuda_module: Optional[object] = None
_rocm_module: Optional[object] = None
_cuda_checked: bool = False
_rocm_checked: bool = False
_cuda_error: Optional[str] = None
_rocm_error: Optional[str] = None


def is_cuda_available() -> bool:
    """
    Check if the CUDA backend is available.
    
    Returns:
        True if crayon_cuda can be imported and CUDA is functional.
    """
    global _cuda_checked, _cuda_module, _cuda_error
    
    if _cuda_checked:
        return _cuda_module is not None
    
    _cuda_checked = True
    try:
        from . import crayon_cuda as _cuda
        # Verify it's functional
        _ = _cuda.get_hardware_info()
        _cuda_module = _cuda
        return True
    except ImportError as e:
        _cuda_error = f"ImportError: {e}"
        return False
    except Exception as e:
        _cuda_error = f"RuntimeError: {e}"
        return False


def is_rocm_available() -> bool:
    """
    Check if the ROCm backend is available.
    
    Returns:
        True if crayon_rocm can be imported and ROCm is functional.
    """
    global _rocm_checked, _rocm_module, _rocm_error
    
    if _rocm_checked:
        return _rocm_module is not None
    
    _rocm_checked = True
    try:
        from . import crayon_rocm as _rocm
        # Verify it's functional
        info = _rocm.get_hardware_info()
        if isinstance(info, str) and "Device Not Found" in info:
            _rocm_error = info
            return False
        _rocm_module = _rocm
        return True
    except ImportError as e:
        _rocm_error = f"ImportError: {e}"
        return False
    except Exception as e:
        _rocm_error = f"RuntimeError: {e}"
        return False


def get_cuda_error() -> Optional[str]:
    """Get the error message if CUDA is unavailable."""
    is_cuda_available()  # Ensure check has run
    return _cuda_error


def get_rocm_error() -> Optional[str]:
    """Get the error message if ROCm is unavailable."""
    is_rocm_available()  # Ensure check has run
    return _rocm_error


def get_available_backends() -> Tuple[str, ...]:
    """
    Get list of available backends.
    
    Returns:
        Tuple of available backend names ("cpu", "cuda", "rocm").
    """
    backends = ["cpu"]
    if is_cuda_available():
        backends.append("cuda")
    if is_rocm_available():
        backends.append("rocm")
    return tuple(backends)


def get_backend_info() -> dict:
    """
    Get detailed information about all backends.
    
    Returns:
        Dictionary with backend status and hardware info.
    """
    info = {
        "cpu": {
            "available": True,
            "hardware": crayon_cpu.get_hardware_info() if hasattr(crayon_cpu, 'get_hardware_info') else "Unknown"
        }
    }
    
    if is_cuda_available():
        try:
            from . import crayon_cuda
            hw = crayon_cuda.get_hardware_info()
            info["cuda"] = {"available": True, "hardware": hw}
        except Exception as e:
            info["cuda"] = {"available": False, "error": str(e)}
    else:
        info["cuda"] = {"available": False, "error": _cuda_error}
    
    if is_rocm_available():
        try:
            from . import crayon_rocm
            hw = crayon_rocm.get_hardware_info()
            info["rocm"] = {"available": True, "hardware": hw}
        except Exception as e:
            info["rocm"] = {"available": False, "error": str(e)}
    else:
        info["rocm"] = {"available": False, "error": _rocm_error}
    
    return info


# ============================================================================
# CONDITIONAL IMPORTS FOR TYPE CHECKING
# ============================================================================

# These will fail at runtime if not available, which is intentional
# Use is_cuda_available() / is_rocm_available() before importing

__all__ = [
    "crayon_cpu",
    "is_cpu_available",
    "get_cpu_backend",
    "get_cpu_error",
    "is_cuda_available",
    "is_rocm_available",
    "get_cuda_error",
    "get_rocm_error",
    "get_available_backends",
    "get_backend_info",
]
