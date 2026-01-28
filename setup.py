"""
XERV CRAYON SETUP - Production Omni-Backend Build System
=========================================================

Uses PyTorch's CUDAExtension for reliable CUDA compilation.
Falls back to CPU-only if PyTorch/CUDA not available.
"""

import os
import sys
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
# DETECT BUILD ENVIRONMENT
# ============================================================================

FORCE_CPU = os.environ.get("CRAYON_FORCE_CPU", "0") == "1"

# Check for PyTorch with CUDA
TORCH_CUDA_AVAILABLE = False
try:
    import torch
    from torch.utils.cpp_extension import CUDAExtension, BuildExtension, CUDA_HOME
    TORCH_CUDA_AVAILABLE = torch.cuda.is_available() and CUDA_HOME is not None
    if TORCH_CUDA_AVAILABLE:
        log(f"PyTorch CUDA detected: {torch.version.cuda}", "OK")
        log(f"CUDA_HOME: {CUDA_HOME}", "OK")
    else:
        log("PyTorch available but CUDA not detected", "WARN")
except ImportError:
    log("PyTorch not available, using standard build", "WARN")
    CUDAExtension = None
    BuildExtension = None
    CUDA_HOME = None

# Check for HIP/ROCm
HAS_ROCM = False
ROCM_HOME = os.environ.get("ROCM_HOME", "/opt/rocm")
if os.path.exists(os.path.join(ROCM_HOME, "bin", "hipcc")):
    HAS_ROCM = True
    log(f"ROCm detected: {ROCM_HOME}", "OK")


# ============================================================================
# EXTENSION DEFINITIONS
# ============================================================================

ext_modules = []

# CPU compile args
if sys.platform == "win32":
    cpu_args = ["/O2", "/arch:AVX2", "/std:c++17"]
    cpu_link = []
elif sys.platform == "darwin":
    cpu_args = ["-O3", "-std=c++17", "-march=native"]
    cpu_link = []
else:
    cpu_args = ["-O3", "-std=c++17", "-fPIC", "-march=native", "-mavx2"]
    cpu_link = []


# --- CPU Extension (Always built) ---
ext_modules.append(Extension(
    "crayon.c_ext.crayon_cpu",
    sources=["src/crayon/c_ext/cpu_engine.cpp"],
    extra_compile_args=cpu_args,
    extra_link_args=cpu_link,
    language="c++",
))
log("CPU extension configured")


# --- CUDA Extension (via PyTorch if available) ---
if TORCH_CUDA_AVAILABLE and not FORCE_CPU and CUDAExtension is not None:
    cuda_ext = CUDAExtension(
        name="crayon.c_ext.crayon_cuda",
        sources=["src/crayon/c_ext/gpu_engine_cuda.cu"],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": [
                "-O3",
                "-std=c++17",
                "--expt-relaxed-constexpr",
                "-gencode=arch=compute_70,code=sm_70",   # V100
                "-gencode=arch=compute_75,code=sm_75",   # T4, RTX 2080
                "-gencode=arch=compute_80,code=sm_80",   # A100
                "-gencode=arch=compute_86,code=sm_86",   # RTX 3090
            ],
        },
    )
    ext_modules.append(cuda_ext)
    log("CUDA extension configured (PyTorch CUDAExtension)", "OK")


# --- ROCm Extension (if available) ---
if HAS_ROCM and not FORCE_CPU:
    ext_modules.append(Extension(
        "crayon.c_ext.crayon_rocm",
        sources=["src/crayon/c_ext/rocm_engine.cpp"],
        libraries=["amdhip64"],
        library_dirs=[os.path.join(ROCM_HOME, "lib")],
        runtime_library_dirs=[os.path.join(ROCM_HOME, "lib")],
        include_dirs=[os.path.join(ROCM_HOME, "include")],
        extra_compile_args=["-O3", "-std=c++17", "-fPIC", "-D__HIP_PLATFORM_AMD__"],
        language="c++",
    ))
    log("ROCm extension configured")


# ============================================================================
# CUSTOM BUILD COMMAND
# ============================================================================

# Use PyTorch's BuildExtension if available, otherwise standard
if BuildExtension is not None and TORCH_CUDA_AVAILABLE:
    cmdclass = {"build_ext": BuildExtension}
    log("Using PyTorch BuildExtension for CUDA compilation")
else:
    cmdclass = {}
    log("Using standard setuptools build")


# ============================================================================
# SETUP
# ============================================================================

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
    zip_safe=False,
)
