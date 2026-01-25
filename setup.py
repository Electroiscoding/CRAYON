import os
import sys
import subprocess
from setuptools import setup, Extension, find_packages

# --- DETECTOR UTILITIES ---
def check_binary(binary_name):
    """
    Checks if a compiler binary exists in the system PATH.
    Returns True if found, False otherwise.
    """
    try:
        # Run with --version to verify it's executable
        subprocess.check_output([binary_name, "--version"], stderr=subprocess.STDOUT)
        return True
    except (OSError, subprocess.CalledProcessError):
        return False

# Detect Hardware Compilers via shell
HAS_NVCC = check_binary("nvcc")
HAS_HIPCC = check_binary("hipcc")
# Allow forced overrides via environment variables for CI/CD pipelines
FORCE_ROCM = os.environ.get("CRAYON_FORCE_ROCM") == "1"
FORCE_CUDA = os.environ.get("CRAYON_FORCE_CUDA") == "1"

ext_modules = []

# --- 1. CPU BACKEND (The Foundation) ---
# Supports AVX2 (Standard) and AVX-512 (Nitro)
cpu_flags = ["-O3", "-fPIC"]
if sys.platform == "linux" or sys.platform == "darwin":
    cpu_flags.append("-march=native") # Optimize for the build machine's CPU
    # Explicitly enable AVX512 support in compiler so intrinsics code compiles
    cpu_flags.append("-mavx512f") 
    cpu_flags.append("-mavx512bw")
elif sys.platform == "win32":
    # MSVC specific flags
    cpu_flags = ["/O2", "/arch:AVX2"] 

ext_modules.append(Extension(
    "crayon.c_ext.crayon_cpu",
    sources=["src/crayon/c_ext/cpu_engine.cpp"],
    extra_compile_args=cpu_flags,
    language="c++"
))

# --- 2. NVIDIA CUDA BACKEND (Green Team) ---
if HAS_NVCC or FORCE_CUDA:
    print("[+] [CRAYON BUILD] NVIDIA NVCC detected. Building 'crayon_cuda'...")
    # In a production PyPI package, you would typically use a custom 'build_ext' class
    # to handle .cu files correctly. Here we define the extension intent.
    # Assuming the environment knows how to link cudart.
    ext_modules.append(Extension(
        "crayon.c_ext.crayon_cuda",
        sources=["src/crayon/c_ext/gpu_engine_cuda.cu"],
        libraries=["cudart"],
        # runtime_library_dirs=["/usr/local/cuda/lib64"] 
    ))
else:
    print("[!] [CRAYON BUILD] NVCC not found. Skipping NVIDIA Backend.")

# --- 3. AMD ROCm BACKEND (Red Team) ---
if HAS_HIPCC or FORCE_ROCM:
    print("[+] [CRAYON BUILD] AMD HIPCC detected. Building 'crayon_rocm'...")
    
    # HIPCC acts like GCC/Clang but links ROCm runtime automatically when invoked.
    # We define the macro so the preprocessor knows we are on AMD.
    ext_modules.append(Extension(
        "crayon.c_ext.crayon_rocm",
        sources=["src/crayon/c_ext/rocm_engine.cpp"],
        extra_compile_args=["-D__HIP_PLATFORM_AMD__"],
        # Link against the HIP runtime library
        libraries=["amdhip64"], 
    ))
else:
    print("[!] [CRAYON BUILD] HIPCC not found. Skipping AMD Backend.")

# Read README for PyPI long description
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="xerv-crayon",
    version="4.0.1",
    description="The Omni-Backend Tokenizer (CPU/CUDA/ROCm)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Xerv-AI/crayon",
    author="Xerv Research Engineering Division",
    author_email="botmaker583@gmail.com",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: POSIX :: Linux",
        "Operating System :: MacOS",
    ],
    packages=find_packages("src"),
    package_dir={"": "src"},
    ext_modules=ext_modules,
    python_requires=">=3.10",
)