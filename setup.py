"""
XERV CRAYON SETUP - Production Omni-Backend Build System
=========================================================

Features:
- PyTorch CUDAExtension for reliable NVCC compilation
- Automatic fallback to CPU if CUDA/ROCm unavailable
- Smart Architecture Detection: Compiles only for the active GPU to save RAM/Time
- MAX_JOBS control to prevent OOM
"""

import os
import sys
import shutil
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

# ============================================================================
# VERSION
# ============================================================================

VERSION = "4.2.6"

# ============================================================================
# PRE-FLIGHT CHECKS
# ============================================================================

# Default to serial build to prevent OOM on Colab/Free tiers
os.environ["MAX_JOBS"] = os.environ.get("MAX_JOBS", "1")

def log(msg: str, level: str = "INFO") -> None:
    print(f"[CRAYON-BUILD] {msg}", flush=True)

# Detect Force CPU
FORCE_CPU = os.environ.get("CRAYON_FORCE_CPU", "0") == "1"

# Detect PyTorch & CUDA
try:
    import torch
    from torch.utils.cpp_extension import CUDAExtension, BuildExtension, CUDA_HOME
    TORCH_CUDA_AVAILABLE = torch.cuda.is_available() and (CUDA_HOME is not None)
except ImportError:
    TORCH_CUDA_AVAILABLE = False
    CUDAExtension = None
    BuildExtension = None
    CUDA_HOME = None

# Detect ROCm
ROCM_HOME = os.environ.get("ROCM_HOME", "/opt/rocm")
HAS_ROCM = os.path.exists(os.path.join(ROCM_HOME, "bin", "hipcc"))


# ============================================================================
# ARCHITECTURE SELECTION
# ============================================================================

def get_cuda_arch_flags():
    """
    Determine the best CUDA architecture flags.
    If CRAYON_GENERIC_BUILD=1, build for all common architectures (for PyPI wheels).
    Otherwise, build ONLY for the detected GPU (faster, less RAM).
    """
    base_flags = ["-O3", "-std=c++17", "--expt-relaxed-constexpr"]
    
    # Generic build for distribution (Wheel)
    if os.environ.get("CRAYON_GENERIC_BUILD", "0") == "1":
        log("Building for ALL common CUDA architectures (Generic Wheel)")
        return base_flags + [
            "-gencode=arch=compute_70,code=sm_70", # V100
            "-gencode=arch=compute_75,code=sm_75", # T4
            "-gencode=arch=compute_80,code=sm_80", # A100
            "-gencode=arch=compute_86,code=sm_86", # RTX 3090
            "-gencode=arch=compute_90,code=sm_90", # H100
        ]
    
    # Local build (Colab/User Machine)
    if TORCH_CUDA_AVAILABLE:
        try:
            major, minor = torch.cuda.get_device_capability()
            arch = f"{major}{minor}"
            log(f"Detected GPU: SM {major}.{minor} -> Compiling for sm_{arch} ONLY")
            return base_flags + [f"-gencode=arch=compute_{arch},code=sm_{arch}"]
        except Exception as e:
            log(f"Error detecting GPU capability: {e}. Falling back to common archs.")
    
    # Fallback if detection fails or no GPU present (but CUDA_HOME exists)
    return base_flags + [
        "-gencode=arch=compute_75,code=sm_75", # T4 (Safe default for Colab)
    ]


# ============================================================================
# EXTENSION CONFIGURATION
# ============================================================================

ext_modules = []

# --- 1. CPU Extension (Always) ---
cpu_args = ["/O2", "/arch:AVX2"] if sys.platform == "win32" else ["-O3", "-march=native", "-mavx2"]
if sys.platform != "win32":
    cpu_args.append("-fPIC")
    cpu_args.append("-std=c++17")
else:
    cpu_args.append("/std:c++17")

ext_modules.append(Extension(
    "crayon.c_ext.crayon_cpu",
    sources=["src/crayon/c_ext/cpu_engine.cpp"],
    extra_compile_args=cpu_args,
    language="c++",
))


# --- 2. CUDA Extension (via PyTorch) ---
if TORCH_CUDA_AVAILABLE and not FORCE_CPU and CUDAExtension:
    nvcc_flags = get_cuda_arch_flags()
    log(f"Configuring CUDA extension (max_jobs={os.environ['MAX_JOBS']})")
    
    ext_modules.append(CUDAExtension(
        name="crayon.c_ext.crayon_cuda",
        sources=["src/crayon/c_ext/gpu_engine_cuda.cu"],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": nvcc_flags,
        },
    ))

elif not FORCE_CPU and CUDAExtension:
    log("Skipping CUDA extension (PyTorch CUDA not found or CUDA_HOME missing)")


# --- 3. ROCm Extension ---
if HAS_ROCM and not FORCE_CPU:
    log(f"Configuring ROCm extension (HOME={ROCM_HOME})")
    ext_modules.append(Extension(
        "crayon.c_ext.crayon_rocm",
        sources=["src/crayon/c_ext/rocm_engine.cpp"],
        libraries=["amdhip64"],
        library_dirs=[os.path.join(ROCM_HOME, "lib")],
        include_dirs=[os.path.join(ROCM_HOME, "include")],
        extra_compile_args=["-O3", "-std=c++17", "-fPIC", "-D__HIP_PLATFORM_AMD__"],
        language="c++",
    ))


# ============================================================================
# BUILD STRATEGY
# ============================================================================

cmdclass = {}
if BuildExtension and (TORCH_CUDA_AVAILABLE or HAS_ROCM):
    cmdclass["build_ext"] = BuildExtension.with_options(no_python_abi_suffix=True)

setup(
    name="xerv-crayon",
    version=VERSION,
    packages=find_packages("src"),
    package_dir={"": "src"},
    include_package_data=True,
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    python_requires=">=3.10",
    zip_safe=False,
)
