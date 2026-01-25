import mmap
import os
import contextlib
from typing import List, Union, Literal

# Type Alias for Device Selection
DeviceType = Literal["cpu", "cuda", "rocm"]

class CrayonVocab:
    def __init__(self, device: DeviceType = "cpu"):
        """
        Initializes the Crayon Tokenizer on the specified hardware.
        
        Args:
            device: 
                'cpu'  - Auto-selects AVX2 or AVX-512 based on hardware.
                'cuda' - Uses NVIDIA GPU (requires crayon_cuda extension).
                'rocm' - Uses AMD GPU (requires crayon_rocm extension).
        """
        self.device = device
        self._cpu_backend = None
        self._gpu_backend = None
        self._dat_mem_ref = None # Keep reference to mmap to prevent garbage collection
        self.current_profile_path = None
        
        # 1. Load CPU Backend (Always Required for Fallback/IO)
        try:
            from ..c_ext import crayon_cpu
            self._cpu_backend = crayon_cpu
            if device == "cpu":
                info = self._cpu_backend.get_hardware_info()
                print(f"[CRAYON] [+] CPU Engine Active: {info}")
        except ImportError:
            raise ImportError("Critical: 'crayon_cpu' extension missing. Build failed?")

        # 2. Conditional GPU Backend Loading
        if device == "cuda":
            try:
                from ..c_ext import crayon_cuda
                self._gpu_backend = crayon_cuda
                info = self._gpu_backend.get_hardware_info()
                print(f"[CRAYON] [+] NVIDIA Engine Active: {info}")
            except ImportError:
                print("[CRAYON] [!] CUDA requested but backend missing. Falling back to CPU.")
                self.device = "cpu"

        elif device == "rocm":
            try:
                from ..c_ext import crayon_rocm
                self._gpu_backend = crayon_rocm
                info = self._gpu_backend.get_hardware_info()
                print(f"[CRAYON] [+] AMD ROCm Engine Active: {info}")
            except ImportError:
                print("[CRAYON] [!] ROCm requested but backend missing. Falling back to CPU.")
                self.device = "cpu"

    def load_profile(self, name_or_path: str):
        """
        Hot-Swaps the active vocabulary cartridge into memory.
        Supports instant switching on CPU and fast switching on GPU.
        """
        # Resolve path (assuming standard cache location if name is given)
        if os.path.exists(name_or_path):
            path = name_or_path
        else:
             # Default lookup path
             path = os.path.expanduser(f"~/.cache/xerv/crayon/profiles/vocab_{name_or_path}.dat")
        
        if not os.path.exists(path):
            raise FileNotFoundError(f"Profile not found: {path}")

        self.current_profile_path = path

        # Always memory map on host first (Zero-Copy)
        f = open(path, "rb")
        self._dat_mem_ref = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        # Dispatch Load Command based on active device
        if self.device == "cpu":
            # CPU Engine reads directly from the memory map via pointer
            self._cpu_backend.load_dat(self._dat_mem_ref)
        
        elif self.device == "cuda":
            # NVIDIA Engine: Seek to start and read bytes to copy to VRAM
            f.seek(0)
            raw_bytes = f.read()
            self._gpu_backend.load_gpu(raw_bytes)
            
        elif self.device == "rocm":
            # AMD Engine: Seek to start and read bytes to copy to HBM
            f.seek(0)
            raw_bytes = f.read()
            self._gpu_backend.load_rocm(raw_bytes)

    @contextlib.contextmanager
    def using_profile(self, name: str):
        """
        Context Manager for temporary profile switching.
        Allows 'within a line' or block-scoped profile changes.
        """
        # 1. Save current state
        previous_path = self.current_profile_path
        
        # 2. Switch to new profile
        try:
            self.load_profile(name)
            yield self
        finally:
            # 3. Restore previous profile
            if previous_path:
                self.load_profile(previous_path)

    def tokenize(self, text_input: Union[str, List[str]]) -> Union[List[int], List[List[int]]]:
        """
        The 'Smash Through' Tokenizer Function.
        Handles single strings or massive batches automatically using the active engine.
        """
        is_batch = isinstance(text_input, list)

        # --- GPU PATHS (Batch Optimized) ---
        if self.device in ["cuda", "rocm"] and self._gpu_backend:
            # GPU demands a list. Wrap single string if needed.
            batch = text_input if is_batch else [text_input]
            
            # Execute Kernel
            if self.device == "cuda":
                # CUDA V2 returns (tokens, metadata)
                ret = self._gpu_backend.tokenize_batch_gpu(batch)
                if isinstance(ret, tuple):
                    results, meta = ret
                    # Optional: Expose metadata if context requires, or log on high latency
                    # if meta['processing_time_ms'] > 1000:
                    #     print(f"[CRAYON] Slow Batch: {meta['processing_time_ms']:.2f}ms")
                else:
                    results = ret
            
            else:
                results = self._gpu_backend.tokenize_batch_rocm(batch)
            
            # Unwrap if input was single string
            return results if is_batch else results[0]

        # --- CPU PATH (Latency Optimized) ---
        else:
            # CPU engine handles single strings natively for lowest latency
            if is_batch:
                return [self._cpu_backend.tokenize(s) for s in text_input]
            else:
                return self._cpu_backend.tokenize(text_input)