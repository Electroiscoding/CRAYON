"""
XERV CRAYON SETUP v4.3.0 - Production Omni-Backend Build System
================================================================

CRITICAL FIX for ROCm/HIP Compilation:
--------------------------------------
The ROCm engine uses HIP kernel syntax (__global__, blockIdx, hipLaunchKernelGGL)
which REQUIRES the hipcc compiler. Standard g++ CANNOT compile these.

This setup.py implements:
1. Custom build_ext that explicitly invokes hipcc for .hip files
2. PyTorch CUDAExtension for reliable NVCC compilation
3. Automatic fallback to CPU if CUDA/ROCm unavailable
4. Smart Architecture Detection: Compiles only for the active GPU to save RAM/Time
5. MAX_JOBS control to prevent OOM

Supported Backends:
- CPU: AVX2/AVX-512 (always built)
- CUDA: NVIDIA via PyTorch CUDAExtension
- ROCm: AMD via hipcc direct invocation
"""

import os
import sys
import subprocess
import shutil
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from distutils.sysconfig import get_python_inc

# ============================================================================
# VERSION
# ============================================================================

VERSION = "4.3.0"

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
    
    # FOR PRODUCTION/CI: Build if CUDA_HOME exists and CRAYON_FORCE_CUDA is set,
    # even if no GPU is found on the build machine.
    FORCE_CUDA = os.environ.get("CRAYON_FORCE_CUDA", "0") == "1"
    TORCH_CUDA_AVAILABLE = (torch.cuda.is_available() or FORCE_CUDA) and (CUDA_HOME is not None)
    
except ImportError:
    TORCH_CUDA_AVAILABLE = False
    CUDAExtension = None
    BuildExtension = None
    CUDA_HOME = None

# Detect ROCm
FORCE_ROCM = os.environ.get("CRAYON_FORCE_ROCM", "0") == "1"
ROCM_HOME = os.environ.get("ROCM_HOME", "/opt/rocm" if sys.platform != "win32" else "")
HIPCC_PATH = os.path.join(ROCM_HOME, "bin", "hipcc") if ROCM_HOME else ""
HAS_ROCM = (os.path.exists(HIPCC_PATH) if HIPCC_PATH else False) or FORCE_ROCM

if HAS_ROCM:
    log(f"ROCm detected at {ROCM_HOME} (Forced={FORCE_ROCM})")
    log(f"hipcc found at {HIPCC_PATH}")
else:
    log("ROCm not detected - skipping AMD backend")


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
# CUSTOM BUILD CLASS FOR HIP COMPILATION
# ============================================================================

class CrayonBuildExt(build_ext):
    """
    Custom build_ext that:
    1. Compiles .hip files using hipcc directly
    2. Falls back to standard behavior for other extensions
    """
    
    def build_extension(self, ext):
        # Check if this is the ROCm extension that needs hipcc
        if hasattr(ext, '_needs_hipcc') and ext._needs_hipcc:
            self._build_hip_extension(ext)
        else:
            # Use standard build for CPU and CUDA extensions
            super().build_extension(ext)
    
    def _build_hip_extension(self, ext):
        """Build HIP extension using hipcc directly"""
        log(f"Building {ext.name} with hipcc...")
        
        # Get output path
        fullname = self.get_ext_fullname(ext.name)
        filename = self.get_ext_filename(ext.name)
        modpath = fullname.split('.')
        
        # Create output directory
        ext_filepath = os.path.join(self.build_lib, *modpath[:-1], modpath[-1] + '.cpython-' + 
                                    str(sys.version_info.major) + str(sys.version_info.minor) + 
                                    '-x86_64-linux-gnu.so')
        
        # Use the proper extension filename
        ext_filepath = os.path.join(self.build_lib, filename)
        
        os.makedirs(os.path.dirname(ext_filepath), exist_ok=True)
        
        # Get Python include directories
        python_include = get_python_inc()
        
        # Build hipcc command
        hip_source = ext.sources[0]  # Should be the .hip file
        
        # hipcc compilation command
        cmd = [
            HIPCC_PATH,
            "-O3",
            "-std=c++17",
            "-fPIC",
            "-shared",
            "-D__HIP_PLATFORM_AMD__",
            f"-I{python_include}",
            f"-I{ROCM_HOME}/include",
            f"-L{ROCM_HOME}/lib",
            "-lamdhip64",
        ]
        
        # Add any additional include dirs
        for inc_dir in ext.include_dirs:
            cmd.append(f"-I{inc_dir}")
        
        # Add output and source
        cmd.extend(["-o", ext_filepath, hip_source])
        
        log(f"Executing: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            if result.stdout:
                print(result.stdout)
            log(f"Successfully built {ext.name}")
        except subprocess.CalledProcessError as e:
            print(f"HIPCC STDOUT:\n{e.stdout}")
            print(f"HIPCC STDERR:\n{e.stderr}")
            raise RuntimeError(f"hipcc compilation failed for {ext.name}") from e


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

# --- Trainer Extension (Always) ---
# Hyper-fast BPE trainer with Linked-List + Inverted Index + Lazy Heap
trainer_args = ["/O2"] if sys.platform == "win32" else ["-O3", "-fPIC"]
if sys.platform != "win32":
    trainer_args.append("-std=c++17")
else:
    trainer_args.append("/std:c++17")

ext_modules.append(Extension(
    "crayon.c_ext.crayon_trainer",
    sources=["src/crayon/c_ext/trainer.cpp"],
    extra_compile_args=trainer_args,
    language="c++",
))

# --- DAT Compiler Extension (Always) ---
# Hyper-fast DAT builder with optimized parking spot search (~500x faster than Python)
compiler_args = ["/O2"] if sys.platform == "win32" else ["-O3", "-fPIC"]
if sys.platform != "win32":
    compiler_args.append("-std=c++17")
else:
    compiler_args.append("/std:c++17")

ext_modules.append(Extension(
    "crayon.c_ext.crayon_compiler",
    sources=["src/crayon/c_ext/compiler.cpp"],
    extra_compile_args=compiler_args,
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


# --- 3. ROCm Extension (AMD - using hipcc directly) ---
if HAS_ROCM and not FORCE_CPU:
    log(f"Configuring ROCm extension (HOME={ROCM_HOME})")
    
    # Create a custom extension marker for HIP files
    hip_ext = Extension(
        "crayon.c_ext.crayon_rocm",
        sources=["src/crayon/c_ext/rocm_engine.hip"],  # .hip file!
        include_dirs=[os.path.join(ROCM_HOME, "include")],
        library_dirs=[os.path.join(ROCM_HOME, "lib")],
        libraries=["amdhip64"],
        language="c++",
    )
    # Mark this extension as needing hipcc
    hip_ext._needs_hipcc = True
    ext_modules.append(hip_ext)


# ============================================================================
# BUILD STRATEGY
# ============================================================================

# Choose the right build command class
if HAS_ROCM and not FORCE_CPU:
    # Use our custom build class that handles hipcc
    log("Using CrayonBuildExt for HIP compilation")
    cmdclass = {"build_ext": CrayonBuildExt}
elif BuildExtension and TORCH_CUDA_AVAILABLE:
    # Use PyTorch's BuildExtension for CUDA
    log("Using PyTorch BuildExtension for CUDA compilation")
    cmdclass = {"build_ext": BuildExtension.with_options(no_python_abi_suffix=True)}
else:
    # Use default
    cmdclass = {}


# ============================================================================
# SETUP ENTRY POINT
# ============================================================================

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
