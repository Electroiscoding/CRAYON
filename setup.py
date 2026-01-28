"""
XERV CRAYON SETUP - Production Omni-Backend Build System
=========================================================

Handles:
1. CPU Backend (crayon_cpu) - Always built with AVX2/AVX-512
2. CUDA Backend (crayon_cuda) - Built if nvcc is available  
3. ROCm Backend (crayon_rocm) - Built if hipcc is available

The CUDA compilation works by:
1. Detecting nvcc at setup time
2. Pre-compiling .cu files to .o using nvcc subprocess
3. Linking the .o into a Python extension
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

VERSION = "4.2.3"

# ============================================================================
# LOGGING
# ============================================================================

def log(msg: str, level: str = "INFO") -> None:
    emoji = {"INFO": "[*]", "WARN": "[!]", "ERROR": "[X]", "OK": "[+]"}.get(level, "")
    print(f"[CRAYON-BUILD] {emoji} {msg}", flush=True)


# ============================================================================
# TOOLCHAIN DETECTION
# ============================================================================

def find_executable(names: list, search_paths: list = None) -> str | None:
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


def get_cuda_home() -> str:
    cuda_home = os.environ.get("CUDA_HOME", os.environ.get("CUDA_PATH", ""))
    if cuda_home and os.path.exists(cuda_home):
        return cuda_home
    for path in ["/usr/local/cuda", "/usr/local/cuda-12", "/usr/local/cuda-11", "/opt/cuda"]:
        if os.path.exists(os.path.join(path, "bin", "nvcc")):
            return path
    return "/usr/local/cuda"


def get_nvcc_path() -> str | None:
    cuda_home = get_cuda_home()
    search_paths = [cuda_home, "/usr/local/cuda", "/usr/local/cuda-12", "/usr/local/cuda-11"]
    return find_executable(["nvcc", "nvcc.exe"], search_paths)


def get_hipcc_path() -> str | None:
    rocm_home = os.environ.get("ROCM_HOME", "/opt/rocm")
    search_paths = [rocm_home, "/opt/rocm", "/opt/rocm-6.0.0", "/opt/rocm-5.7.0"]
    return find_executable(["hipcc"], search_paths)


NVCC_PATH = get_nvcc_path()
HIPCC_PATH = get_hipcc_path()
CUDA_HOME = get_cuda_home()
FORCE_CPU = os.environ.get("CRAYON_FORCE_CPU", "0") == "1"

HAS_NVCC = NVCC_PATH is not None and not FORCE_CPU
HAS_HIPCC = HIPCC_PATH is not None and not FORCE_CPU

log(f"CUDA Home: {CUDA_HOME}")
log(f"NVCC Path: {NVCC_PATH}" if NVCC_PATH else "NVCC not found", "OK" if NVCC_PATH else "WARN")
log(f"HIPCC Path: {HIPCC_PATH}" if HIPCC_PATH else "HIPCC not found", "OK" if HIPCC_PATH else "WARN")


# ============================================================================
# CUDA COMPILATION FUNCTION
# ============================================================================

def compile_cuda_to_object(cu_file: str, output_dir: str) -> str | None:
    """Compile a CUDA .cu file to a .o object file using nvcc."""
    if not NVCC_PATH:
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    
    obj_file = os.path.join(output_dir, os.path.basename(cu_file).replace(".cu", ".o"))
    
    # Get Python include directory
    python_include = sysconfig.get_path("include")
    cuda_include = os.path.join(CUDA_HOME, "include")
    
    # Build nvcc command
    cmd = [
        NVCC_PATH,
        "-c", cu_file,
        "-o", obj_file,
        "-O3",
        "-std=c++17",
        "-Xcompiler", "-fPIC",
        "-I", python_include,
        "-I", cuda_include,
        # Multi-arch for T4, V100, A100, RTX 3090
        "-gencode=arch=compute_70,code=sm_70",
        "-gencode=arch=compute_75,code=sm_75",
        "-gencode=arch=compute_80,code=sm_80",
        "-gencode=arch=compute_86,code=sm_86",
        "-allow-unsupported-compiler",
        "--compiler-options", "-w",  # Suppress warnings
    ]
    
    log(f"Compiling CUDA: {cu_file}")
    log(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            log(f"NVCC stderr: {result.stderr}", "ERROR")
            return None
        log(f"CUDA compiled: {obj_file}", "OK")
        return obj_file
    except subprocess.TimeoutExpired:
        log("CUDA compilation timed out", "ERROR")
        return None
    except Exception as e:
        log(f"CUDA compilation failed: {e}", "ERROR")
        return None


# ============================================================================
# CUSTOM BUILD EXTENSION
# ============================================================================

class CrayonBuildExt(build_ext):
    """Custom build_ext that pre-compiles CUDA files with nvcc."""
    
    def run(self):
        # Pre-compile CUDA if available
        if HAS_NVCC:
            cuda_src = os.path.join("src", "crayon", "c_ext", "gpu_engine_cuda.cu")
            if os.path.exists(cuda_src):
                log("=" * 60)
                log("STARTING CUDA COMPILATION")
                log("=" * 60)
                
                obj_file = compile_cuda_to_object(cuda_src, self.build_temp)
                
                if obj_file and os.path.exists(obj_file):
                    # Find the CUDA extension and configure it
                    for ext in self.extensions:
                        if "crayon_cuda" in ext.name:
                            ext.extra_objects = [obj_file]
                            ext.library_dirs = [os.path.join(CUDA_HOME, "lib64")]
                            ext.runtime_library_dirs = [os.path.join(CUDA_HOME, "lib64")]
                            ext.libraries = ["cudart"]
                            ext.include_dirs = [
                                os.path.join(CUDA_HOME, "include"),
                                sysconfig.get_path("include"),
                            ]
                            log("CUDA extension configured with compiled object", "OK")
                else:
                    log("CUDA compilation failed, removing CUDA extension", "ERROR")
                    self.extensions = [e for e in self.extensions if "crayon_cuda" not in e.name]
            else:
                log(f"CUDA source not found: {cuda_src}", "WARN")
        
        # Now run the standard build
        super().run()
    
    def build_extension(self, ext):
        # For CUDA extension, we need a dummy source since the real code is in the .o file
        if "crayon_cuda" in ext.name and ext.extra_objects:
            # Create a minimal C++ stub that just includes Python.h
            stub_dir = os.path.join(self.build_temp, "cuda_stub")
            os.makedirs(stub_dir, exist_ok=True)
            stub_file = os.path.join(stub_dir, "cuda_stub.cpp")
            
            with open(stub_file, "w") as f:
                f.write('// Stub - real implementation in linked .o file\n')
                f.write('#include <Python.h>\n')
                f.write('// PyInit_crayon_cuda is defined in gpu_engine_cuda.o\n')
            
            ext.sources = [stub_file]
        
        super().build_extension(ext)


# ============================================================================
# EXTENSION DEFINITIONS
# ============================================================================

ext_modules = []

# CPU compile args
if sys.platform == "win32":
    cpu_args = ["/O2", "/arch:AVX2", "/std:c++17"]
elif sys.platform == "darwin":
    cpu_args = ["-O3", "-std=c++17", "-march=native"]
else:
    cpu_args = ["-O3", "-std=c++17", "-fPIC", "-march=native", "-mavx2"]

# --- CPU Extension (Always) ---
ext_modules.append(Extension(
    "crayon.c_ext.crayon_cpu",
    sources=["src/crayon/c_ext/cpu_engine.cpp"],
    extra_compile_args=cpu_args,
    language="c++",
))
log("CPU extension configured")

# --- CUDA Extension (if nvcc available) ---
if HAS_NVCC:
    # Note: sources will be replaced with stub in build_extension
    cuda_ext = Extension(
        "crayon.c_ext.crayon_cuda",
        sources=[],  # Will be set during build
        libraries=["cudart"],
        library_dirs=[os.path.join(CUDA_HOME, "lib64")],
        runtime_library_dirs=[os.path.join(CUDA_HOME, "lib64")],
        include_dirs=[os.path.join(CUDA_HOME, "include")],
        language="c++",
    )
    ext_modules.append(cuda_ext)
    log("CUDA extension configured (will compile with nvcc)")

# --- ROCm Extension (if hipcc available) ---
if HAS_HIPCC:
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
    cmdclass={"build_ext": CrayonBuildExt},
    python_requires=">=3.10",
    zip_safe=False,
)