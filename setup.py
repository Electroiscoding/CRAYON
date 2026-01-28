"""
XERV CRAYON SETUP - Production Omni-Backend Build System
=========================================================

Uses PyTorch's CUDAExtension for proper nvcc compilation on GPU systems.

Backends:
1. CPU (crayon_cpu) - Always built with AVX2/AVX-512
2. CUDA (crayon_cuda) - Built if torch+CUDA available
3. ROCm (crayon_rocm) - Built if hipcc available
"""

import os
import sys
import subprocess
import shutil
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

# ============================================================================
# VERSION
# ============================================================================

VERSION = "4.2.4"

# ============================================================================
# LOGGING
# ============================================================================

def log(msg: str, level: str = "INFO") -> None:
    emoji = {"INFO": "[*]", "WARN": "[!]", "ERROR": "[X]", "OK": "[+]"}.get(level, "")
    print(f"[CRAYON-BUILD] {emoji} {msg}", flush=True)


# ============================================================================
# DETECT PYTORCH CUDA AVAILABILITY
# ============================================================================

TORCH_AVAILABLE = False
TORCH_CUDA_AVAILABLE = False

try:
    import torch
    TORCH_AVAILABLE = True
    TORCH_CUDA_AVAILABLE = torch.cuda.is_available()
    log(f"PyTorch {torch.__version__} detected", "OK")
    log(f"PyTorch CUDA: {TORCH_CUDA_AVAILABLE}", "OK" if TORCH_CUDA_AVAILABLE else "WARN")
except ImportError:
    log("PyTorch not found - will check for standalone nvcc", "WARN")


# ============================================================================
# DETECT STANDALONE NVCC (for non-PyTorch CUDA builds)
# ============================================================================

def find_nvcc() -> str | None:
    """Find nvcc compiler."""
    # Check common locations
    cuda_home = os.environ.get("CUDA_HOME", os.environ.get("CUDA_PATH", "/usr/local/cuda"))
    
    candidates = [
        shutil.which("nvcc"),
        os.path.join(cuda_home, "bin", "nvcc"),
        "/usr/local/cuda/bin/nvcc",
        "/usr/local/cuda-12/bin/nvcc",
        "/usr/local/cuda-11/bin/nvcc",
    ]
    
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


def find_hipcc() -> str | None:
    """Find AMD hipcc compiler."""
    rocm_home = os.environ.get("ROCM_HOME", "/opt/rocm")
    candidates = [
        shutil.which("hipcc"),
        os.path.join(rocm_home, "bin", "hipcc"),
    ]
    for path in candidates:
        if path and os.path.exists(path):
            return path
    return None


NVCC_PATH = find_nvcc()
HIPCC_PATH = find_hipcc()
FORCE_CPU = os.environ.get("CRAYON_FORCE_CPU", "0") == "1"

# Determine what to build
BUILD_CUDA = (TORCH_CUDA_AVAILABLE or NVCC_PATH) and not FORCE_CPU
BUILD_ROCM = HIPCC_PATH is not None and not FORCE_CPU

log(f"NVCC: {NVCC_PATH}" if NVCC_PATH else "NVCC not found", "OK" if NVCC_PATH else "WARN")
log(f"Build CUDA: {BUILD_CUDA}")
log(f"Build ROCm: {BUILD_ROCM}")


# ============================================================================
# EXTENSION MODULES
# ============================================================================

ext_modules = []

# CPU compile args
if sys.platform == "win32":
    cpu_args = ["/O2", "/arch:AVX2", "/std:c++17"]
elif sys.platform == "darwin":
    cpu_args = ["-O3", "-std=c++17", "-march=native"]
else:
    cpu_args = ["-O3", "-std=c++17", "-fPIC", "-march=native", "-mavx2"]

# --- CPU Extension (Always built) ---
ext_modules.append(Extension(
    "crayon.c_ext.crayon_cpu",
    sources=["src/crayon/c_ext/cpu_engine.cpp"],
    extra_compile_args=cpu_args,
    language="c++",
))
log("CPU extension configured")

# --- CUDA Extension (via PyTorch CUDAExtension) ---
if BUILD_CUDA and TORCH_AVAILABLE:
    try:
        from torch.utils.cpp_extension import CUDAExtension
        
        cuda_ext = CUDAExtension(
            name="crayon.c_ext.crayon_cuda",
            sources=["src/crayon/c_ext/gpu_engine_cuda.cu"],
            extra_compile_args={
                "cxx": ["-O3", "-std=c++17"],
                "nvcc": [
                    "-O3",
                    "-std=c++17",
                    "--use_fast_math",
                    "-gencode=arch=compute_70,code=sm_70",  # V100
                    "-gencode=arch=compute_75,code=sm_75",  # T4
                    "-gencode=arch=compute_80,code=sm_80",  # A100
                    "-gencode=arch=compute_86,code=sm_86",  # RTX 3090
                    "--expt-relaxed-constexpr",
                    "-allow-unsupported-compiler",
                ],
            },
        )
        ext_modules.append(cuda_ext)
        log("CUDA extension configured (via PyTorch CUDAExtension)", "OK")
    except Exception as e:
        log(f"Failed to configure CUDA extension: {e}", "ERROR")

# --- ROCm Extension ---
if BUILD_ROCM:
    rocm_home = os.environ.get("ROCM_HOME", "/opt/rocm")
    ext_modules.append(Extension(
        "crayon.c_ext.crayon_rocm",
        sources=["src/crayon/c_ext/rocm_engine.cpp"],
        libraries=["amdhip64"],
        library_dirs=[os.path.join(rocm_home, "lib")],
        runtime_library_dirs=[os.path.join(rocm_home, "lib")],
        include_dirs=[os.path.join(rocm_home, "include")],
        extra_compile_args=["-O3", "-std=c++17", "-fPIC", "-D__HIP_PLATFORM_AMD__"],
        language="c++",
    ))
    log("ROCm extension configured", "OK")


# ============================================================================
# CUSTOM BUILD EXTENSION
# ============================================================================

class CrayonBuildExt(build_ext):
    """Custom build_ext that uses PyTorch's BuildExtension for CUDA."""
    
    def build_extensions(self):
        # Check if we have CUDA extensions
        has_cuda_ext = any("crayon_cuda" in ext.name for ext in self.extensions)
        
        if has_cuda_ext and TORCH_AVAILABLE:
            try:
                from torch.utils.cpp_extension import BuildExtension
                # Use PyTorch's BuildExtension for CUDA
                torch_builder = BuildExtension.with_options(use_ninja=False)
                torch_builder.build_extensions(self)
                return
            except Exception as e:
                log(f"PyTorch BuildExtension failed: {e}", "ERROR")
                # Fall back to standard build for non-CUDA extensions
                self.extensions = [e for e in self.extensions if "crayon_cuda" not in e.name]
        
        # Standard build for CPU/ROCm
        super().build_extensions()


# ============================================================================
# SETUP
# ============================================================================

# Choose cmdclass
if BUILD_CUDA and TORCH_AVAILABLE:
    try:
        from torch.utils.cpp_extension import BuildExtension
        cmdclass = {"build_ext": BuildExtension.with_options(use_ninja=False)}
        log("Using PyTorch BuildExtension")
    except:
        cmdclass = {"build_ext": CrayonBuildExt}
else:
    cmdclass = {"build_ext": CrayonBuildExt}


setup(
    name="xerv-crayon",
    version=VERSION,
    description="Omni-Backend Tokenizer - CPU (AVX2/512), CUDA (NVIDIA), ROCm (AMD)",
    author="Xerv Research Engineering Division",
    author_email="engineering@xerv.ai",
    packages=find_packages("src"),
    package_dir={"": "src"},
    include_package_data=True,
    package_data={
        "crayon.resources.dat": ["*.dat", "*.json"],
        "crayon.c_ext": ["*.h", "*.c", "*.cpp", "*.cu"],
    },
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    python_requires=">=3.10",
    install_requires=[],
    extras_require={
        "cuda": ["torch>=2.0.0"],
    },
    zip_safe=False,
)