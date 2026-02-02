"""
XERV CRAYON V4.2.0 - OMNI-BACKEND FRONTEND
==========================================
The unified interface for CPU (AVX2/512), CUDA (NVIDIA), and ROCm (AMD) tokenization.
Handles automatic hardware detection, zero-copy memory mapping, and dynamic profile switching.

Architecture:
    - Default (device="auto"): Scans system for NVIDIA/AMD GPUs, falls back to CPU
    - Manual Override: Force device="cpu", "cuda", or "rocm"
    - Unified API: Same .tokenize() method works on all platforms

Production Features:
    - Thread-safe operations with RLock
    - Zero-copy memory mapping for DAT profiles
    - Graceful fallback on hardware failures
    - Context manager for temporary profile switching
    - Full decode support with companion JSON files
"""

from __future__ import annotations

import contextlib
import json
import logging
import mmap
import os
import platform
import sys
import tempfile
import threading
from dataclasses import dataclass, field
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Final,
    List,
    Literal,
    Optional,
    Protocol,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    cast,
    runtime_checkable,
)

if TYPE_CHECKING:
    from types import ModuleType

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

_logger = logging.getLogger("crayon.vocab")
_logger.addHandler(logging.NullHandler())

# Production log handler (user can override)
_console_handler = logging.StreamHandler()
_console_handler.setFormatter(
    logging.Formatter("[CRAYON] %(levelname)s: %(message)s")
)


def enable_verbose_logging(level: int = logging.INFO) -> None:
    """Enable console logging for Crayon operations."""
    _logger.addHandler(_console_handler)
    _logger.setLevel(level)


def disable_verbose_logging() -> None:
    """Disable console logging."""
    _logger.removeHandler(_console_handler)


# ============================================================================
# TYPE DEFINITIONS
# ============================================================================

DeviceType = Literal["auto", "cpu", "cuda", "rocm"]
TokenIds = List[int]
BatchTokenIds = List[List[int]]

# Device priority order for auto-detection
_DEVICE_PRIORITY: Final[Tuple[DeviceType, ...]] = ("cuda", "rocm", "cpu")


class DeviceState(Enum):
    """Backend initialization states."""
    UNINITIALIZED = "uninitialized"
    READY = "ready"
    FAILED = "failed"
    FALLBACK = "fallback"


@runtime_checkable
class CPUBackendProtocol(Protocol):
    """Protocol for CPU backend module."""
    def load_dat(self, buffer: Any) -> int: ...
    def tokenize(self, text: str) -> List[int]: ...
    def get_hardware_info(self) -> str: ...


@runtime_checkable
class GPUBackendProtocol(Protocol):
    """Protocol for GPU backend modules (CUDA/ROCm)."""
    def get_hardware_info(self) -> Any: ...


@runtime_checkable
class CUDABackendProtocol(Protocol):
    """Protocol for CUDA backend module."""
    def get_hardware_info(self) -> Any: ...
    def load_gpu(self, data: bytes) -> Any: ...
    def tokenize_batch_gpu(self, batch: List[str]) -> Any: ...


@runtime_checkable
class ROCmBackendProtocol(Protocol):
    """Protocol for ROCm backend module."""
    def get_hardware_info(self) -> Any: ...
    def load_rocm(self, data: bytes) -> int: ...
    def tokenize_batch_rocm(self, batch: List[str]) -> List[List[int]]: ...


# ============================================================================
# HARDWARE DETECTION UTILITIES
# ============================================================================

@dataclass(frozen=True)
class HardwareInfo:
    """Immutable hardware detection result."""
    device: DeviceType
    name: str
    features: str
    vram_mb: Optional[int] = None
    compute_capability: Optional[str] = None
    is_available: bool = True
    error: Optional[str] = None


def _detect_cuda_availability() -> Tuple[bool, Optional[str]]:
    """
    Multi-layer CUDA detection.
    
    Checks in order:
    1. Direct extension import + runtime test
    2. PyTorch CUDA availability (if installed)
    3. Environment markers (CUDA_VISIBLE_DEVICES, etc.)
    
    Returns:
        Tuple of (is_available, error_message)
    """
    # Layer 1: Direct extension
    try:
        from ..c_ext import crayon_cuda
        info = crayon_cuda.get_hardware_info()
        if isinstance(info, dict) and info.get("name"):
            return True, None
        return True, None
    except ImportError:
        pass
    except Exception as e:
        return False, f"CUDA extension failed: {e}"
    
    # Layer 2: PyTorch check
    try:
        import torch
        if torch.cuda.is_available():
            return True, None
    except ImportError:
        pass
    except Exception:
        pass
    
    # Layer 3: Environment check
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible and cuda_visible != "-1":
        # CUDA devices are set, but we can't use them without the extension
        return False, "CUDA_VISIBLE_DEVICES set but extension not available"
    
    return False, "No CUDA installation detected"


def _detect_rocm_availability() -> Tuple[bool, Optional[str]]:
    """
    Multi-layer ROCm detection.
    
    Checks in order:
    1. Direct extension import + runtime test
    2. HIP environment markers
    3. AMD GPU sysfs check (Linux only)
    
    Returns:
        Tuple of (is_available, error_message)
    """
    # Layer 1: Direct extension
    try:
        from ..c_ext import crayon_rocm
        info = crayon_rocm.get_hardware_info()
        if isinstance(info, str):
            if "Device Not Found" in info:
                return False, info
            return True, None
        if isinstance(info, dict):
            return True, None
        return True, None
    except ImportError:
        pass
    except Exception as e:
        return False, f"ROCm extension failed: {e}"
    
    # Layer 2: HIP environment check
    hip_visible = os.environ.get("HIP_VISIBLE_DEVICES", "")
    if hip_visible and hip_visible != "-1":
        return False, "HIP_VISIBLE_DEVICES set but extension not available"
    
    # Layer 3: Linux sysfs check
    if sys.platform == "linux":
        amd_gpu_paths = ["/sys/class/drm/card0/device/vendor"]
        for path in amd_gpu_paths:
            try:
                with open(path, "r") as f:
                    vendor = f.read().strip()
                    if vendor == "0x1002":  # AMD vendor ID
                        return False, "AMD GPU detected but extension not available"
            except (IOError, OSError):
                pass
    
    return False, "No ROCm installation detected"


def _get_cpu_info() -> HardwareInfo:
    """Detect CPU capabilities."""
    try:
        from ..c_ext import crayon_cpu
        info_str = crayon_cpu.get_hardware_info()
        return HardwareInfo(
            device="cpu",
            name=info_str.split("[")[0].strip() if "[" in info_str else info_str,
            features=info_str.split("[")[1].rstrip("]") if "[" in info_str else "Standard",
            is_available=True,
        )
    except Exception as e:
        # Fallback to platform info
        return HardwareInfo(
            device="cpu",
            name=platform.processor() or "Unknown CPU",
            features="Standard",
            is_available=True,
            error=str(e),
        )


# ============================================================================
# PROFILE RESOLUTION
# ============================================================================

def _get_profile_search_paths(profile_name: str) -> List[str]:
    """
    Generate ordered list of paths to search for a profile.
    
    Search order:
    1. Exact path (if file exists)
    2. Package resources (editable install)
    3. pkg_resources (wheel install)
    4. importlib.resources (modern Python)
    5. CRAYON_PROFILE_DIR environment variable
    6. User cache (~/.cache/xerv/crayon/profiles/)
    7. System cache (/var/cache/crayon/ on Linux)
    """
    paths: List[str] = []
    expected_dat = f"vocab_{profile_name}.dat"
    
    # Package resources (editable install)
    rel_path = os.path.join(
        os.path.dirname(__file__), "..", "resources", "dat", expected_dat
    )
    paths.append(os.path.abspath(rel_path))
    
    # importlib.resources (Python 3.9+ - preferred modern approach)
    try:
        from importlib import resources
        try:
            # Python 3.11+ API with files()
            ref = resources.files("crayon").joinpath("resources", "dat", expected_dat)
            with resources.as_file(ref) as p:
                paths.append(str(p))
        except (TypeError, AttributeError, FileNotFoundError):
            pass
    except Exception:
        pass
    
    # CRAYON_PROFILE_DIR environment variable
    profile_dir = os.environ.get("CRAYON_PROFILE_DIR")
    if profile_dir:
        paths.append(os.path.join(os.path.expanduser(profile_dir), expected_dat))
    
    # User cache
    home = os.path.expanduser("~")
    paths.append(os.path.join(home, ".cache", "xerv", "crayon", "profiles", expected_dat))
    
    # System cache (Linux)
    if sys.platform == "linux":
        paths.append(f"/var/cache/crayon/{expected_dat}")
    
    return paths


# ============================================================================
# MAIN CLASS: CrayonVocab
# ============================================================================

class CrayonVocab:
    """
    The High-Performance Tokenizer Interface.
    
    Automatically dispatches to the fastest available hardware backend.
    Supports hot-swapping vocabulary profiles and batch processing.
    
    Thread Safety:
        All public methods are thread-safe via an internal RLock.
        
    Memory Model:
        - CPU: Zero-copy mmap access to DAT file
        - CUDA: Full copy to GPU VRAM (async transfer)
        - ROCm: Full copy to GPU HBM (async transfer)
    
    Examples:
        >>> # Auto-detect best device
        >>> vocab = CrayonVocab(device="auto")
        >>> vocab.load_profile("lite")
        >>> tokens = vocab.tokenize("Hello, world!")
        
        >>> # Force CPU for latency-sensitive workloads
        >>> vocab = CrayonVocab(device="cpu")
        >>> vocab.load_profile("code")
        >>> tokens = vocab.tokenize("def forward(self, x):")
        
        >>> # Batch processing on GPU
        >>> vocab = CrayonVocab(device="cuda")
        >>> vocab.load_profile("lite")
        >>> batch_tokens = vocab.tokenize(["doc1", "doc2", "doc3"])
        
        >>> # Context manager for temporary profile switch
        >>> with vocab.using_profile("science"):
        ...     tokens = vocab.tokenize("E=mc²")
    """
    
    __slots__ = (
        "_lock",
        "_cpu_backend",
        "_gpu_backend",
        "_dat_file_ref",
        "_dat_mem_ref",
        "_idx_to_str",
        "current_profile_path",
        "_profile_loaded",
        "_temp_dat_path",
        "unk_token",
        "unk_token_id",
        "device",
        "_requested_device",
        "_device_state",
        "_hardware_info",
    )
    
    def __init__(
        self, 
        vocab_list: Optional[List[str]] = None, 
        device: DeviceType = "auto",
        unk_token: str = "<UNK>"
    ) -> None:
        """
        Initialize the tokenizer engine.

        Args:
            vocab_list: Optional list of strings to build an ad-hoc vocabulary.
            device: Device selection mode.
                - "auto": Detects GPU. If available, uses it. Else CPU.
                - "cpu": Forces AVX2/AVX-512 CPU backend (best for latency).
                - "cuda": Forces NVIDIA GPU backend (best for batch throughput).
                - "rocm": Forces AMD GPU backend (best for batch throughput).
            unk_token: String to use as the unknown token placeholder.
                
        Raises:
            ImportError: If the CPU backend extension is not available.
            ValueError: If an invalid device string is provided.
            
        Environment Variables:
            CRAYON_DEVICE: Override device selection (cpu|cuda|rocm)
            CRAYON_PROFILE_DIR: Custom profile search directory
        """
        self._lock = threading.RLock()
        
        # Backend references
        self._cpu_backend: Optional[CPUBackendProtocol] = None
        self._gpu_backend: Optional[Union[CUDABackendProtocol, ROCmBackendProtocol]] = None
        
        # Profile state
        self._dat_file_ref: Optional[Any] = None
        self._dat_mem_ref: Optional[mmap.mmap] = None
        self._idx_to_str: List[str] = []
        self.current_profile_path: Optional[str] = None
        self._profile_loaded: bool = False
        self._temp_dat_path: Optional[str] = None
        
        # Public properties for test compatibility
        self.unk_token = unk_token
        self.unk_token_id = 1 # Hardware convention in Crayon v2
        
        # Device state
        self._requested_device: DeviceType = device
        self._device_state: DeviceState = DeviceState.UNINITIALIZED
        self._hardware_info: Optional[HardwareInfo] = None
        
        # Validate device parameter
        if device not in ("auto", "cpu", "cuda", "rocm"):
            raise ValueError(
                f"Invalid device: {device!r}. Must be 'auto', 'cpu', 'cuda', or 'rocm'."
            )
        
        # --- Critical: Load CPU Backend ---
        self._load_cpu_backend()
        
        # --- Resolve and Initialize Device ---
        self.device = self._resolve_device(device)
        self._init_selected_backend()
        
        # --- Load ad-hoc vocab if provided ---
        if vocab_list:
            self.load_from_list(vocab_list)
    
    def _load_cpu_backend(self) -> None:
        """Load the CPU extension (required as fallback for all modes)."""
        try:
            from ..c_ext import crayon_cpu
            self._cpu_backend = crayon_cpu
            _logger.debug("CPU backend loaded successfully")
        except ImportError as e:
            _logger.critical("Failed to load crayon_cpu extension")
            raise ImportError(
                "Critical Crayon Error: 'crayon_cpu' extension not found. "
                "The package may not be installed correctly. Try:\n"
                "  pip install --force-reinstall xerv-crayon\n"
                "Or for development:\n"
                "  pip install -e .\n"
            ) from e
    
    def _resolve_device(self, requested: DeviceType) -> DeviceType:
        """
        Resolve the actual device to use based on request and availability.
        
        Auto mode priority: CUDA > ROCm > CPU
        """
        # Check environment override
        env_override = os.environ.get("CRAYON_DEVICE", "").strip().lower()
        if requested == "auto" and env_override in ("cpu", "cuda", "rocm"):
            requested = cast(DeviceType, env_override)
            _logger.info("Device override from CRAYON_DEVICE=%s", env_override)
        
        # Direct request (non-auto)
        if requested != "auto":
            return requested
        
        # Auto-detection priority
        cuda_ok, cuda_err = _detect_cuda_availability()
        if cuda_ok:
            _logger.debug("CUDA detected and available")
            return "cuda"
        elif cuda_err:
            _logger.debug("CUDA check: %s", cuda_err)
        
        rocm_ok, rocm_err = _detect_rocm_availability()
        if rocm_ok:
            _logger.debug("ROCm detected and available")
            return "rocm"
        elif rocm_err:
            _logger.debug("ROCm check: %s", rocm_err)
        
        _logger.debug("Defaulting to CPU backend")
        return "cpu"
    
    def _init_selected_backend(self) -> None:
        """Initialize the selected backend with fallback handling."""
        if self.device == "cpu":
            self._gpu_backend = None
            self._device_state = DeviceState.READY
            try:
                info = self._cpu_backend.get_hardware_info()
                self._hardware_info = HardwareInfo(
                    device="cpu",
                    name=info.split("[")[0].strip() if "[" in info else info,
                    features=info.split("[")[1].rstrip("]") if "[" in info else "Standard",
                )
                _logger.info("🔵 CPU Engine Active: %s", info)
            except Exception:
                self._hardware_info = _get_cpu_info()
                _logger.info("🔵 CPU Engine Active")
            return
        
        if self.device == "cuda":
            try:
                from ..c_ext import crayon_cuda
                info = crayon_cuda.get_hardware_info()
                self._gpu_backend = crayon_cuda
                self._device_state = DeviceState.READY
                
                if isinstance(info, dict):
                    self._hardware_info = HardwareInfo(
                        device="cuda",
                        name=info.get("name", "NVIDIA GPU"),
                        features="CUDA",
                        vram_mb=info.get("vram_mb"),
                        compute_capability=info.get("compute_capability"),
                    )
                    _logger.info("🟢 NVIDIA CUDA Engine Active: %s", info.get("full_info", info.get("name")))
                else:
                    self._hardware_info = HardwareInfo(
                        device="cuda",
                        name=str(info),
                        features="CUDA",
                    )
                    _logger.info("🟢 NVIDIA CUDA Engine Active: %s", info)
                return
            except ImportError:
                _logger.warning("CUDA extension not compiled. Falling back to CPU.")
            except Exception as e:
                _logger.warning("CUDA initialization failed (%s). Falling back to CPU.", e)
            
            self._device_state = DeviceState.FALLBACK
            self.device = "cpu"
            self._init_selected_backend()
            return
        
        if self.device == "rocm":
            try:
                from ..c_ext import crayon_rocm
                info = crayon_rocm.get_hardware_info()
                
                if isinstance(info, str) and "Device Not Found" in info:
                    raise RuntimeError(info)
                
                self._gpu_backend = crayon_rocm
                self._device_state = DeviceState.READY
                
                if isinstance(info, str):
                    self._hardware_info = HardwareInfo(
                        device="rocm",
                        name=info.split("[")[0].strip() if "[" in info else info,
                        features="ROCm/HIP",
                    )
                else:
                    self._hardware_info = HardwareInfo(
                        device="rocm",
                        name=str(info),
                        features="ROCm/HIP",
                    )
                _logger.info("🔴 AMD ROCm Engine Active: %s", info)
                return
            except ImportError:
                _logger.warning("ROCm extension not compiled. Falling back to CPU.")
            except Exception as e:
                _logger.warning("ROCm initialization failed (%s). Falling back to CPU.", e)
            
            self._device_state = DeviceState.FALLBACK
            self.device = "cpu"
            self._init_selected_backend()
            return
    
    def set_device(
        self,
        device: DeviceType,
        *,
        reload_profile: bool = True,
    ) -> None:
        """
        Switch the active backend at runtime.

        Args:
            device: New device to use ("auto", "cpu", "cuda", "rocm").
            reload_profile: If True and a profile was loaded, reload it on new backend.
            
        Note:
            If the requested backend is unavailable, this falls back to CPU.
        """
        with self._lock:
            previous_profile = self.current_profile_path
            had_profile = self._profile_loaded and previous_profile is not None
            
            self._requested_device = device
            self.device = self._resolve_device(device)
            self._init_selected_backend()
            
            if reload_profile and had_profile:
                self.load_profile(previous_profile)
    
    def _resolve_profile_path(self, name_or_path: str) -> str:
        """
        Resolve a profile name or path to an absolute file path.
        
        Args:
            name_or_path: Either a profile name ("lite", "code") or full path.
            
        Returns:
            Absolute path to the .dat file.
            
        Raises:
            FileNotFoundError: If the profile cannot be found.
        """
        # Check if it's already a valid path
        candidate = os.path.expanduser(name_or_path)
        if os.path.exists(candidate):
            return os.path.abspath(candidate)
        
        # Search in known locations
        search_paths = _get_profile_search_paths(name_or_path)
        for path in search_paths:
            if os.path.exists(path):
                return path
        
        # Generate helpful error message
        checked_locations = "\n".join(f"  - {p}" for p in search_paths[:4])
        raise FileNotFoundError(
            f"Profile '{name_or_path}' not found.\n"
            f"Searched locations:\n{checked_locations}\n"
            f"You can specify the full path or set CRAYON_PROFILE_DIR environment variable."
        )
    
    @property
    def id_to_token(self) -> List[str]:
        """Get the ID-to-token mapping list (for compatibility)."""
        return self._idx_to_str

    def __len__(self) -> int:
        """Return the total number of tokens in the active vocabulary."""
        return len(self._idx_to_str)

    def __contains__(self, token: str) -> bool:
        """Check if a token exists in the active vocabulary (O(N) fallback)."""
        return token in self._idx_to_str

    def load_from_list(self, vocab: List[str]) -> None:
        """Build and load a temporary DAT profile from a list of strings."""
        try:
            from ..c_ext import crayon_compiler
        except ImportError:
            raise ImportError("crayon_compiler extension required for load_from_list()")

        with self._lock:
            # Create a secure temporary file
            fd, path = tempfile.mkstemp(suffix=".dat")
            os.close(fd)
            
            try:
                # Compile to the temp file
                crayon_compiler.compile_dat(vocab, path)
                
                # IMPORTANT: Since load_profile() expects a .json file to load _idx_to_str,
                # we create a dummy JSON or just bypass the load_profile JSON loading
                # by manually setting _idx_to_str after load_profile.
                self.load_profile(path)
                
                # Override the idx_to_str which failed to load during load_profile (since no .json exists)
                self._idx_to_str = list(vocab)
                self._temp_dat_path = path
                
            except Exception as e:
                if os.path.exists(path):
                    os.unlink(path)
                raise RuntimeError(f"Failed to build ad-hoc vocabulary: {e}")

    def _close_profile_handles(self) -> None:
        """Safely close any open file handles."""
        if self._dat_mem_ref is not None:
            try:
                self._dat_mem_ref.close()
            except Exception:
                pass
            self._dat_mem_ref = None
        
        if self._dat_file_ref is not None:
            try:
                self._dat_file_ref.close()
            except Exception:
                pass
            self._dat_file_ref = None
            
        # Clean up temporary DAT if exists
        if hasattr(self, '_temp_dat_path') and self._temp_dat_path and os.path.exists(self._temp_dat_path):
            try:
                os.unlink(self._temp_dat_path)
            except Exception:
                pass
            self._temp_dat_path = None
    
    def close(self) -> None:
        """Release all resources and close file handles."""
        with self._lock:
            self._close_profile_handles()
            self.current_profile_path = None
            self._idx_to_str = []
            self._profile_loaded = False
    
    def __del__(self) -> None:
        """Destructor to ensure resources are released."""
        try:
            self.close()
        except Exception:
            pass
    
    def __enter__(self) -> "CrayonVocab":
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit (closes resources)."""
        self.close()
    
    def load_profile(self, name_or_path: str) -> None:
        """
        Hot-swap the active vocabulary profile.

        Args:
            name_or_path: Either a profile name (e.g., "lite", "code", "science")
                         or a full path to a .dat file.
                         
        Raises:
            FileNotFoundError: If the profile cannot be found.
            OSError: If the file cannot be memory-mapped.
            RuntimeError: If profile loading fails on the current device.
            
        Note:
            This method automatically loads the companion .json file for decode().
            The .json file should have the same base name as the .dat file.
        """
        with self._lock:
            self._profile_loaded = False
            path = self._resolve_profile_path(name_or_path)
            self.current_profile_path = path
            
            # Load decoder mapping (companion JSON)
            # Load decoder mapping (companion JSON)
            json_path = os.path.splitext(path)[0] + ".json"
            if os.path.exists(json_path):
                try:
                    with open(json_path, "r", encoding="utf-8") as jf:
                        loaded = json.load(jf)
                        
                        if isinstance(loaded, list):
                            # V1 Legacy Format (List of strings)
                            self._idx_to_str = loaded
                        elif isinstance(loaded, dict) and "vocab" in loaded:
                            # V2 Format (Dict with 'vocab' key: string -> int)
                            vocab_map = loaded["vocab"]
                            if not vocab_map:
                                self._idx_to_str = []
                            else:
                                max_id = max(vocab_map.values())
                                temp_list = [""] * (max_id + 1)
                                for token, tid in vocab_map.items():
                                    if 0 <= tid <= max_id:
                                        temp_list[tid] = token
                                self._idx_to_str = temp_list
                        else:
                            raise ValueError("JSON must be a list or dict with 'vocab' key")
                            
                except Exception as e:
                    _logger.warning("Failed to load decoder JSON: %s", e)
                    self._idx_to_str = []
            else:
                self._idx_to_str = []
            
            # Close previous handles
            self._close_profile_handles()
            
            # Memory-map the DAT file
            try:
                self._dat_file_ref = open(path, "rb")
                self._dat_mem_ref = mmap.mmap(
                    self._dat_file_ref.fileno(), 0, access=mmap.ACCESS_READ
                )
            except OSError as e:
                self._close_profile_handles()
                raise OSError(
                    f"Failed to memory-map profile: {path}. "
                    f"Ensure the file exists and is readable. Error: {e}"
                ) from e
            
            # Dispatch to appropriate backend
            if self.device == "cpu":
                self._cpu_backend.load_dat(self._dat_mem_ref)
                self._profile_loaded = True
                _logger.debug("Profile loaded on CPU: %s", os.path.basename(path))
                return
            
            if self.device == "cuda":
                try:
                    raw_bytes = self._dat_mem_ref[:]
                    result = self._gpu_backend.load_gpu(raw_bytes)
                    self._profile_loaded = True
                    # ALSO LOAD CPU FOR FALLBACK
                    self._cpu_backend.load_dat(self._dat_mem_ref)
                    _logger.debug("Profile loaded on CUDA: %s (result: %s)", os.path.basename(path), result)
                    return
                except Exception as e:
                    _logger.warning("CUDA profile load failed (%s). Falling back to CPU.", e)
                    self.device = "cpu"
                    self._device_state = DeviceState.FALLBACK
                    self._init_selected_backend()
                    self._cpu_backend.load_dat(self._dat_mem_ref)
                    self._profile_loaded = True
                    return
            
            if self.device == "rocm":
                try:
                    raw_bytes = self._dat_mem_ref[:]
                    self._gpu_backend.load_rocm(raw_bytes)
                    self._profile_loaded = True
                    # ALSO LOAD CPU FOR FALLBACK
                    self._cpu_backend.load_dat(self._dat_mem_ref)
                    _logger.debug("Profile loaded on ROCm: %s", os.path.basename(path))
                    return
                except Exception as e:
                    _logger.warning("ROCm profile load failed (%s). Falling back to CPU.", e)
                    self.device = "cpu"
                    self._device_state = DeviceState.FALLBACK
                    self._init_selected_backend()
                    self._cpu_backend.load_dat(self._dat_mem_ref)
                    self._profile_loaded = True
                    return
            
            raise RuntimeError(f"Unhandled device state: {self.device!r}")
    
    @contextlib.contextmanager
    def using_profile(self, name_or_path: str):
        """
        Context manager for temporarily switching profiles.
        
        Args:
            name_or_path: Profile name or path to use within the context.
            
        Yields:
            self: The CrayonVocab instance with the new profile loaded.
            
        Note:
            The previous profile is automatically restored on exit.
            If no profile was loaded before, the new profile remains active.
            
        Example:
            >>> vocab.load_profile("lite")
            >>> with vocab.using_profile("code"):
            ...     tokens = vocab.tokenize(source_code)
            >>> # Back to "lite" profile automatically
        """
        previous_path = self.current_profile_path
        try:
            self.load_profile(name_or_path)
            yield self
        finally:
            if previous_path:
                self.load_profile(previous_path)
    
    def tokenize(
        self,
        text_input: Union[str, Sequence[str]],
    ) -> Union[List[int], List[List[int]]]:
        """
        Tokenize text using the active vocabulary profile.

        Args:
            text_input: Input to tokenize.
                - str: Returns List[int] (single sequence)
                - Sequence[str]: Returns List[List[int]] (batch)
                
        Returns:
            Token IDs as a list or list of lists.
            
        Raises:
            RuntimeError: If no profile is loaded.
            TypeError: If input is not str or sequence of str.
            
        Performance Notes:
            - CPU: Optimized for single-string latency (~1µs overhead)
            - GPU: Optimized for batch throughput (launch overhead amortized)
            - For <100 strings, CPU may be faster even with GPU available
        """
        with self._lock:
            if not self._profile_loaded:
                raise RuntimeError(
                    "No vocabulary profile loaded. Call load_profile() first."
                )
            
            # Determine input type
            if isinstance(text_input, str):
                is_batch = False
                batch: List[str] = [text_input]
            else:
                is_batch = True
                batch = list(text_input)
            
            # Handle empty batch
            if not batch:
                return [] if is_batch else []
            
            # Validate all items are strings
            for i, item in enumerate(batch):
                if not isinstance(item, str):
                    raise TypeError(
                        f"tokenize() expects str or Sequence[str], "
                        f"got {type(item).__name__} at index {i}"
                    )
            
            # --- GPU PATH ---
            if self.device in ("cuda", "rocm") and self._gpu_backend is not None:
                try:
                    if self.device == "cuda":
                        ret = self._gpu_backend.tokenize_batch_gpu(batch)
                        # CUDA returns (results, metadata) tuple
                        results = ret[0] if isinstance(ret, tuple) else ret
                    else:
                        results = self._gpu_backend.tokenize_batch_rocm(batch)
                    
                    return results if is_batch else results[0]
                except Exception as e:
                    _logger.warning("GPU tokenization failed (%s). Using CPU fallback.", e)
                    # Fall through to CPU path
            
            # --- CPU PATH ---
            if is_batch:
                return [self._cpu_backend.tokenize(s) for s in batch]
            return self._cpu_backend.tokenize(batch[0])
    
    def decode(self, tokens: Sequence[int]) -> str:
        """
        Decode token IDs back to text.

        Args:
            tokens: Sequence of token IDs to decode.
            
        Returns:
            Reconstructed text string.
            
        Raises:
            RuntimeError: If no profile is loaded or decoder JSON is missing.
            TypeError: If tokens is not a sequence of integers.
            ValueError: If any token ID is out of range.
            
        Note:
            Requires a companion .json file with the same base name as the .dat profile.
        """
        if not self._profile_loaded:
            raise RuntimeError(
                "No vocabulary profile loaded. Call load_profile() first."
            )
        
        if not self._idx_to_str:
            raise RuntimeError(
                "Decoder mapping not loaded. Ensure the profile has a companion .json file "
                "with the same base name as the .dat file."
            )
        
        out: List[str] = []
        for i, t in enumerate(tokens):
            if not isinstance(t, int):
                raise TypeError(
                    f"decode() expects sequence of ints, got {type(t).__name__} at index {i}"
                )
            if t < 0 or t >= len(self._idx_to_str):
                raise ValueError(
                    f"Token ID {t} out of range [0, {len(self._idx_to_str) - 1}]"
                )
            out.append(self._idx_to_str[t])
        
        return "".join(out)
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get metadata about the current engine state.
        
        Returns:
            Dictionary with device info, backend type, and active profile.
        """
        profile_name = (
            os.path.basename(self.current_profile_path)
            if self.current_profile_path
            else None
        )
        backend = (
            "cpu_extension" if self.device == "cpu" else f"{self.device}_extension"
        )
        
        info: Dict[str, Any] = {
            "device": self.device,
            "backend": backend,
            "active_profile": profile_name,
            "profile_loaded": self._profile_loaded,
            "vocab_size": len(self._idx_to_str) if self._idx_to_str else None,
            "device_state": self._device_state.value,
        }
        
        if self._hardware_info:
            info["hardware"] = {
                "name": self._hardware_info.name,
                "features": self._hardware_info.features,
            }
            if self._hardware_info.vram_mb:
                info["hardware"]["vram_mb"] = self._hardware_info.vram_mb
            if self._hardware_info.compute_capability:
                info["hardware"]["compute_capability"] = self._hardware_info.compute_capability
        
        return info
    
    def __repr__(self) -> str:
        """Return a developer-friendly representation."""
        profile = os.path.basename(self.current_profile_path) if self.current_profile_path else "None"
        return f"<CrayonVocab device={self.device!r} profile={profile!r} loaded={self._profile_loaded}>"
    
    @property
    def vocab_size(self) -> int:
        """Get the vocabulary size (number of tokens)."""
        return len(self._idx_to_str) if self._idx_to_str else 0
    
    @property
    def is_gpu(self) -> bool:
        """Check if running on GPU backend."""
        return self.device in ("cuda", "rocm") and self._gpu_backend is not None
    
    @property
    def is_profile_loaded(self) -> bool:
        """Check if a profile is currently loaded."""
        return self._profile_loaded

    @property
    def fast_mode(self) -> bool:
        """Check if running in high-performance mode (C++ backend)."""
        return self.device in ("cpu", "cuda", "rocm") and (self._cpu_backend is not None or self._gpu_backend is not None)

    def longest_match(self, text: str, pos: int = 0) -> Tuple[int, int]:
        """
        Find the longest matching token at the given position (Compatibility Mode).
        
        Note: This is slower than tokenize() as it creates a substring.
        """
        if pos >= len(text):
            return self.unk_token_id, 0
            
        # Optimization: We only need to check a reasonable window
        # The longest token is rarely more than 100 characters.
        window = text[pos : pos + 128]
        tokens = self.tokenize(window)
        
        if not tokens:
            return self.unk_token_id, 1
            
        # Get the first token ID
        first_id = tokens[0]
        
        # Get its length from id_to_token
        if 0 <= first_id < len(self._idx_to_str):
            token_str = self._idx_to_str[first_id]
            return first_id, len(token_str)
        else:
            return self.unk_token_id, 1


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def quick_tokenize(
    text: Union[str, Sequence[str]],
    profile: str = "lite",
    device: DeviceType = "auto",
) -> Union[List[int], List[List[int]]]:
    """
    One-shot tokenization without explicitly managing CrayonVocab.
    
    Args:
        text: Text or list of texts to tokenize.
        profile: Profile name to use (default: "lite").
        device: Device selection (default: "auto").
        
    Returns:
        Token IDs.
        
    Note:
        For repeated tokenization, create a CrayonVocab instance instead.
        This function has initialization overhead on each call.
    """
    vocab = CrayonVocab(device=device)
    vocab.load_profile(profile)
    return vocab.tokenize(text)


# ============================================================================
# MODULE EXPORTS
# ============================================================================

__all__ = [
    "CrayonVocab",
    "DeviceType",
    "HardwareInfo",
    "DeviceState",
    "quick_tokenize",
    "enable_verbose_logging",
    "disable_verbose_logging",
]