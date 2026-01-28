"""
XERV CRAYON SETUP - Production Omni-Backend Build System
=========================================================

Features:
- PyTorch CUDAExtension for reliable NVCC compilation
- Automatic fallback to CPU if CUDA/ROCm unavailable
- MAX_JOBS control to prevent OOM on smaller instances
"""

import os
import sys
import shutil
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

# ============================================================================
# VERSION
# ============================================================================

VERSION = "4.2.5"

# ============================================================================
# PRE-FLIGHT CHECKS
# ============================================================================

# Control parallelism to prevent OOM
os.environ["MAX_JOBS"] = os.environ.get("MAX_JOBS", "4")

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
    log(f"Configuring CUDA extension (PyTorch {torch.__version__}, CUDA {torch.version.cuda})")
    ext_modules.append(CUDAExtension(
        name="crayon.c_ext.crayon_cuda",
        sources=["src/crayon/c_ext/gpu_engine_cuda.cu"],
        extra_compile_args={
            "cxx": ["-O3", "-std=c++17"],
            "nvcc": [
                "-O3", "-std=c++17",
                "--expt-relaxed-constexpr",
                # Broad architecture support
                "-gencode=arch=compute_70,code=sm_70",
                "-gencode=arch=compute_75,code=sm_75",
                "-gencode=arch=compute_80,code=sm_80",
                "-gencode=arch=compute_86,code=sm_86",
                "-gencode=arch=compute_90,code=sm_90",
            ],
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
