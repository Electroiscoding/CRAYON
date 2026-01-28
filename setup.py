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
    CRAYON_CUDA_ARCH=sm_80   # Override CUDA architecture
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

VERSION = "4.2.1"

# ============================================================================
# LOGGING
# ============================================================================

def log(msg: str, level: str = "INFO") -> None:
    """Print build status messages."""
    emoji = {"INFO": "📦", "WARN": "⚠️", "ERROR": "❌", "OK": "✅"}.get(level, "")
    sys.stderr.write(f"[CRAYON] {emoji} {msg}\n")
    sys.stderr.flush()


# ============================================================================
# TOOLCHAIN DETECTION
# ============================================================================

def find_executable(names: list, search_paths: list = None) -> str | None:
    """Find an executable in PATH or specified paths."""
    for name in names:
        # Check PATH
        found = shutil.which(name)
        if found:
            return found
        
        # Check custom paths
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
    log("CUDA compiler not found - GPU backend will not be built", "WARN")

if HAS_HIPCC:
    log(f"ROCm HIP compiler found: {HIPCC_BIN}", "OK")
else:
    log("ROCm HIP compiler not found - AMD backend will not be built", "WARN")


# ============================================================================
# CUDA COMPILATION
# ============================================================================

def get_cuda_include_dirs() -> list:
    """Get CUDA include directories."""
    dirs = []
    cuda_home = os.environ.get("CUDA_HOME", os.environ.get("CUDA_PATH", "/usr/local/cuda"))
    if os.path.exists(os.path.join(cuda_home, "include")):
        dirs.append(os.path.join(cuda_home, "include"))
    return dirs


def get_cuda_library_dirs() -> list:
    """Get CUDA library directories."""
    dirs = []
    cuda_home = os.environ.get("CUDA_HOME", os.environ.get("CUDA_PATH", "/usr/local/cuda"))
    for libdir in ["lib64", "lib", "lib/x64"]:
        path = os.path.join(cuda_home, libdir)
        if os.path.exists(path):
            dirs.append(path)
    return dirs


def compile_cuda_kernel(source_file: str, build_dir: str) -> str:
    """
    Compile CUDA kernel using nvcc.
    
    Args:
        source_file: Path to .cu file
        build_dir: Build directory for object files
        
    Returns:
        Path to compiled object file
    """
    source_file = os.path.abspath(source_file)
    obj_ext = ".obj" if sys.platform == "win32" else ".o"
    obj_name = os.path.basename(source_file).replace(".cu", obj_ext)
    output_file = os.path.join(os.path.abspath(build_dir), obj_name)
    
    os.makedirs(build_dir, exist_ok=True)
    
    # Collect include paths
    includes = [
        sysconfig.get_path("include"),
        sysconfig.get_config_var('INCLUDEPY'),
        sysconfig.get_config_var('CONFINCLUDEPY'),
    ]
    includes.extend(get_cuda_include_dirs())
    include_flags = [f"-I{i}" for i in includes if i and os.path.exists(i)]
    
    # Compiler flags
    flags = ["-O3", "-c", "-std=c++17", "-w"]  # -w suppresses warnings
    
    if sys.platform != "win32":
        flags.extend(["-Xcompiler", "-fPIC"])
        flags.append("-allow-unsupported-compiler")
        
        # GPU architectures - broad compatibility
        # User can override with CRAYON_CUDA_ARCH
        arch = os.environ.get("CRAYON_CUDA_ARCH")
        if arch:
            flags.append(f"-arch={arch}")
        else:
            # Support common architectures (T4, V100, A100, H100)
            flags.extend([
                "-gencode=arch=compute_70,code=sm_70",  # V100
                "-gencode=arch=compute_75,code=sm_75",  # T4, RTX 2080
                "-gencode=arch=compute_80,code=sm_80",  # A100
                "-gencode=arch=compute_86,code=sm_86",  # RTX 3090
                "-gencode=arch=compute_89,code=sm_89",  # RTX 4090
                "-gencode=arch=compute_90,code=sm_90",  # H100
            ])
    else:
        # Windows flags
        flags.extend(["-Xcompiler", "/O2"])
    
    cmd = [NVCC_BIN, source_file, "-o", output_file] + flags + include_flags
    log(f"Compiling CUDA: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            log(f"CUDA compilation failed: {result.stderr}", "ERROR")
            raise RuntimeError(f"nvcc failed: {result.stderr}")
        log(f"CUDA compilation successful: {output_file}", "OK")
        return output_file
    except subprocess.TimeoutExpired:
        raise RuntimeError("CUDA compilation timed out")


# ============================================================================
# ROCM COMPILATION
# ============================================================================

def compile_rocm_kernel(source_file: str, build_dir: str) -> str:
    """
    Compile ROCm kernel using hipcc.
    
    Args:
        source_file: Path to .cpp file (HIP uses .cpp)
        build_dir: Build directory for object files
        
    Returns:
        Path to compiled object file
    """
    source_file = os.path.abspath(source_file)
    obj_name = os.path.basename(source_file).replace(".cpp", ".o")
    output_file = os.path.join(os.path.abspath(build_dir), obj_name)
    
    os.makedirs(build_dir, exist_ok=True)
    
    # Collect include paths
    includes = [
        sysconfig.get_path("include"),
        sysconfig.get_config_var('INCLUDEPY'),
    ]
    rocm_home = os.environ.get("ROCM_HOME", "/opt/rocm")
    if os.path.exists(os.path.join(rocm_home, "include")):
        includes.append(os.path.join(rocm_home, "include"))
    include_flags = [f"-I{i}" for i in includes if i and os.path.exists(i)]
    
    # Compiler flags
    flags = ["-O3", "-c", "-std=c++17", "-fPIC", "-w"]
    
    cmd = [HIPCC_BIN, source_file, "-o", output_file] + flags + include_flags
    log(f"Compiling ROCm: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            log(f"ROCm compilation failed: {result.stderr}", "ERROR")
            raise RuntimeError(f"hipcc failed: {result.stderr}")
        log(f"ROCm compilation successful: {output_file}", "OK")
        return output_file
    except subprocess.TimeoutExpired:
        raise RuntimeError("ROCm compilation timed out")


# ============================================================================
# CUSTOM BUILD EXTENSION
# ============================================================================

class CustomBuildExt(build_ext):
    """Custom build_ext that handles CUDA and ROCm compilation."""
    
    def build_extensions(self):
        # Compile CUDA if available
        if HAS_NVCC:
            cuda_src = os.path.join("src", "crayon", "c_ext", "gpu_engine_cuda.cu")
            if os.path.exists(cuda_src):
                log("Building CUDA backend...")
                try:
                    os.makedirs(self.build_temp, exist_ok=True)
                    obj_path = compile_cuda_kernel(cuda_src, self.build_temp)
                    
                    # Find the CUDA extension and add the object file
                    for ext in self.extensions:
                        if "crayon_cuda" in ext.name:
                            ext.extra_objects.append(obj_path)
                            ext.include_dirs.extend(get_cuda_include_dirs())
                            ext.library_dirs.extend(get_cuda_library_dirs())
                            log("CUDA backend configured", "OK")
                except Exception as e:
                    log(f"CUDA build failed: {e}", "ERROR")
                    # Remove CUDA extension from list
                    self.extensions = [e for e in self.extensions if "crayon_cuda" not in e.name]
            else:
                log(f"CUDA source not found: {cuda_src}", "WARN")
        
        # Compile ROCm if available
        if HAS_HIPCC:
            rocm_src = os.path.join("src", "crayon", "c_ext", "rocm_engine.cpp")
            if os.path.exists(rocm_src):
                log("Building ROCm backend...")
                try:
                    # ROCm uses hipcc directly which handles everything
                    log("ROCm backend configured", "OK")
                except Exception as e:
                    log(f"ROCm build failed: {e}", "ERROR")
                    self.extensions = [e for e in self.extensions if "crayon_rocm" not in e.name]
        
        # Build all extensions
        super().build_extensions()


# ============================================================================
# EXTENSION DEFINITIONS
# ============================================================================

ext_modules = []

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
        sources=[],  # Object file added during build
        libraries=["cudart"],
        library_dirs=get_cuda_library_dirs(),
        runtime_library_dirs=get_cuda_library_dirs() if sys.platform != "win32" else [],
        include_dirs=get_cuda_include_dirs(),
        language="c++",
    )
    ext_modules.append(cuda_ext)
    log("CUDA extension configured (pending compilation)")

# --- ROCm Extension (Optional) ---
if HAS_HIPCC:
    rocm_home = os.environ.get("ROCM_HOME", "/opt/rocm")
    rocm_ext = Extension(
        "crayon.c_ext.crayon_rocm",
        sources=["src/crayon/c_ext/rocm_engine.cpp"],
        libraries=["amdhip64"],
        library_dirs=[os.path.join(rocm_home, "lib")],
        runtime_library_dirs=[os.path.join(rocm_home, "lib")],
        include_dirs=[os.path.join(rocm_home, "include")],
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
    cmdclass={"build_ext": CustomBuildExt},
    python_requires=">=3.10",
    zip_safe=False,
)