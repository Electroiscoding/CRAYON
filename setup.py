"""
XERV CRAYON SETUP - Omni-Backend Build System
==============================================

This setup.py handles:
1. CPU Backend (crayon_cpu) - Always built with AVX2/AVX-512
2. CUDA Backend (crayon_cuda) - Built if nvcc is available
3. ROCm Backend (crayon_rocm) - Built if hipcc is available

Build Process:
    pip install .                    # Auto-detect and build available backends
    pip install -e .                 # Editable install for development
    pip install --no-build-isolation .  # Force local build

Environment Variables:
    CRAYON_FORCE_CPU=1       # Skip GPU backend compilation
    CRAYON_CUDA_ARCH=sm_75   # Override CUDA architecture (e.g., sm_75 for T4)
    CUDA_HOME=/usr/local/cuda # Custom CUDA path
    ROCM_HOME=/opt/rocm      # Custom ROCm path
"""

import os
import sys
import subprocess
import shutil
import sysconfig
from pathlib import Path
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext

# ============================================================================
# VERSION
# ============================================================================

VERSION = "4.2.2"

# ============================================================================
# LOGGING
# ============================================================================

def log(msg: str, level: str = "INFO") -> None:
    """Print build status messages."""
    emoji = {"INFO": "[*]", "WARN": "[!]", "ERROR": "[X]", "OK": "[+]"}.get(level, "")
    sys.stderr.write(f"[CRAYON] {emoji} {msg}\n")
    sys.stderr.flush()


# ============================================================================
# TOOLCHAIN DETECTION
# ============================================================================

def find_executable(names: list, search_paths: list = None) -> str | None:
    """Find an executable in PATH or specified paths."""
    for name in names:
        found = shutil.which(name)
        if found:
            return found
        if search_paths:
            for base in search_paths:
                candidate = os.path.join(base, "bin", name)
                if os.path.exists(candidate) and os.access(candidate, os.X_OK):
                    return candidate
    return None


def get_nvcc_path() -> str | None:
    """Find NVIDIA CUDA compiler."""
    cuda_home = os.environ.get("CUDA_HOME", os.environ.get("CUDA_PATH", ""))
    search_paths = [
        cuda_home,
        "/usr/local/cuda",
        "/usr/local/cuda-12",
        "/usr/local/cuda-11",
        "/opt/cuda",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.0",
        "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8",
    ]
    return find_executable(["nvcc", "nvcc.exe"], search_paths)


def get_hipcc_path() -> str | None:
    """Find AMD ROCm HIP compiler."""
    rocm_home = os.environ.get("ROCM_HOME", os.environ.get("HIP_PATH", ""))
    search_paths = [
        rocm_home,
        "/opt/rocm",
        "/opt/rocm-5.7.0",
        "/opt/rocm-6.0.0",
    ]
    return find_executable(["hipcc"], search_paths)


# Detect compilers
NVCC_BIN = get_nvcc_path()
HIPCC_BIN = get_hipcc_path()
HAS_NVCC = NVCC_BIN is not None and not os.environ.get("CRAYON_FORCE_CPU")
HAS_HIPCC = HIPCC_BIN is not None and not os.environ.get("CRAYON_FORCE_CPU")

if HAS_NVCC:
    log(f"CUDA compiler found: {NVCC_BIN}", "OK")
else:
    log("CUDA compiler not found - CUDA backend will not be built", "WARN")

if HAS_HIPCC:
    log(f"ROCm HIP compiler found: {HIPCC_BIN}", "OK")
else:
    log("ROCm HIP compiler not found - AMD backend will not be built", "WARN")


# ============================================================================
# CUDA UTILITIES
# ============================================================================

def get_cuda_home() -> str:
    """Get CUDA installation directory."""
    cuda_home = os.environ.get("CUDA_HOME", os.environ.get("CUDA_PATH", ""))
    if cuda_home and os.path.exists(cuda_home):
        return cuda_home
    
    # Common locations
    for path in ["/usr/local/cuda", "/opt/cuda", "/usr"]:
        if os.path.exists(os.path.join(path, "include", "cuda.h")):
            return path
    
    return "/usr/local/cuda"


def get_cuda_include_dirs() -> list:
    """Get CUDA include directories."""
    cuda_home = get_cuda_home()
    dirs = [os.path.join(cuda_home, "include")]
    return [d for d in dirs if os.path.exists(d)]


def get_cuda_library_dirs() -> list:
    """Get CUDA library directories."""
    cuda_home = get_cuda_home()
    dirs = []
    for libdir in ["lib64", "lib", "lib/x64"]:
        path = os.path.join(cuda_home, libdir)
        if os.path.exists(path):
            dirs.append(path)
    return dirs


# ============================================================================
# CUSTOM BUILD EXTENSION FOR CUDA
# ============================================================================

class CUDABuildExt(build_ext):
    """Custom build_ext that handles CUDA (.cu) compilation via nvcc."""
    
    def build_extensions(self):
        # Save original compiler
        original_compile = self.compiler.compile
        original_link = self.compiler.link_shared_object
        
        # Check if we need CUDA compilation
        cuda_extensions = [e for e in self.extensions if any(
            s.endswith('.cu') for s in e.sources
        )]
        
        if cuda_extensions and HAS_NVCC:
            log("Configuring CUDA compilation...")
            
            # Wrap compiler to handle .cu files
            def custom_compile(sources, output_dir=None, macros=None, include_dirs=None,
                             debug=0, extra_preargs=None, extra_postargs=None, depends=None):
                cu_sources = [s for s in sources if s.endswith('.cu')]
                cc_sources = [s for s in sources if not s.endswith('.cu')]
                
                objects = []
                
                # Compile CUDA files with nvcc
                if cu_sources:
                    for cu_src in cu_sources:
                        obj = self._compile_cuda(cu_src, output_dir, include_dirs)
                        if obj:
                            objects.append(obj)
                
                # Compile regular C/C++ files normally
                if cc_sources:
                    cc_objects = original_compile(cc_sources, output_dir, macros, include_dirs,
                                                 debug, extra_preargs, extra_postargs, depends)
                    objects.extend(cc_objects)
                
                return objects
            
            self.compiler.compile = custom_compile
        
        # Build all extensions
        super().build_extensions()
    
    def _compile_cuda(self, source, output_dir, include_dirs):
        """Compile a .cu file using nvcc."""
        obj_ext = ".obj" if sys.platform == "win32" else ".o"
        obj_name = os.path.splitext(os.path.basename(source))[0] + obj_ext
        
        if output_dir is None:
            output_dir = self.build_temp
        
        os.makedirs(output_dir, exist_ok=True)
        obj_path = os.path.join(output_dir, obj_name)
        
        # Build nvcc command
        includes = [sysconfig.get_path("include")]
        includes.extend(get_cuda_include_dirs())
        if include_dirs:
            includes.extend(include_dirs)
        include_flags = [f"-I{d}" for d in includes if d and os.path.exists(d)]
        
        # Compiler flags
        flags = ["-O3", "-c", "-std=c++17", "-Xcompiler", "-fPIC"]
        
        # GPU architectures - Tesla T4 (7.5), V100 (7.0), A100 (8.0), H100 (9.0)
        arch_env = os.environ.get("CRAYON_CUDA_ARCH")
        if arch_env:
            flags.append(f"-arch={arch_env}")
        else:
            # Support all common Colab/Cloud GPUs
            flags.extend([
                "-gencode=arch=compute_70,code=sm_70",   # V100
                "-gencode=arch=compute_75,code=sm_75",   # T4, RTX 2080
                "-gencode=arch=compute_80,code=sm_80",   # A100
                "-gencode=arch=compute_86,code=sm_86",   # RTX 3090
            ])
        
        if sys.platform != "win32":
            flags.append("-allow-unsupported-compiler")
        
        cmd = [NVCC_BIN, source, "-o", obj_path] + flags + include_flags
        log(f"Compiling CUDA: {os.path.basename(source)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            if result.returncode != 0:
                log(f"CUDA compilation failed:\n{result.stderr}", "ERROR")
                return None
            log(f"CUDA compilation successful: {obj_name}", "OK")
            return obj_path
        except subprocess.TimeoutExpired:
            log("CUDA compilation timed out", "ERROR")
            return None
        except Exception as e:
            log(f"CUDA compilation error: {e}", "ERROR")
            return None


# ============================================================================
# EXTENSION DEFINITIONS
# ============================================================================

ext_modules = []

# Get Python include for extensions
python_include = sysconfig.get_path("include")

# --- CPU Extension (Always built) ---
cpu_compile_args = []
cpu_link_args = []

if sys.platform == "win32":
    cpu_compile_args = ["/O2", "/arch:AVX2", "/std:c++17"]
elif sys.platform == "darwin":
    cpu_compile_args = ["-O3", "-std=c++17", "-march=native"]
else:
    cpu_compile_args = ["-O3", "-std=c++17", "-fPIC", "-march=native", "-mavx2"]

ext_modules.append(Extension(
    "crayon.c_ext.crayon_cpu",
    sources=["src/crayon/c_ext/cpu_engine.cpp"],
    extra_compile_args=cpu_compile_args,
    extra_link_args=cpu_link_args,
    language="c++",
))
log("CPU extension configured")

# --- CUDA Extension (Optional) ---
if HAS_NVCC:
    cuda_ext = Extension(
        "crayon.c_ext.crayon_cuda",
        sources=["src/crayon/c_ext/gpu_engine_cuda.cu"],  # nvcc will compile this
        libraries=["cudart"],
        library_dirs=get_cuda_library_dirs(),
        runtime_library_dirs=get_cuda_library_dirs() if sys.platform != "win32" else [],
        include_dirs=get_cuda_include_dirs() + [python_include],
        language="c++",
    )
    ext_modules.append(cuda_ext)
    log("CUDA extension configured (will compile with nvcc)")

# --- ROCm Extension (Optional) ---
if HAS_HIPCC:
    rocm_home = os.environ.get("ROCM_HOME", "/opt/rocm")
    rocm_ext = Extension(
        "crayon.c_ext.crayon_rocm",
        sources=["src/crayon/c_ext/rocm_engine.cpp"],
        libraries=["amdhip64"],
        library_dirs=[os.path.join(rocm_home, "lib")],
        runtime_library_dirs=[os.path.join(rocm_home, "lib")],
        include_dirs=[os.path.join(rocm_home, "include"), python_include],
        extra_compile_args=["-O3", "-std=c++17", "-fPIC", "-D__HIP_PLATFORM_AMD__"],
        language="c++",
    )
    ext_modules.append(rocm_ext)
    log("ROCm extension configured")


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
    cmdclass={"build_ext": CUDABuildExt},
    python_requires=">=3.10",
    zip_safe=False,
)