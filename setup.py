import os
import sys
import platform
from setuptools import setup, Extension, find_packages

# -----------------------------------------------------------------------------
# C-Extension Configuration
# -----------------------------------------------------------------------------

def get_compile_args():
    """
    Determine compiler flags based on the platform.
    """
    system = platform.system()
    args = []
    
    if system == 'Windows':
        # MSVC flags
        args = ['/Ox', '/arch:AVX2', '/D_CRT_SECURE_NO_WARNINGS']
    else:
        # GCC/Clang flags (Linux/macOS)
        args = [
            '-O3',                      # Max optimization
            '-mavx2',                   # Enable AVX2 instructions [cite: 512]
            '-mfma',                    # Enable Fused Multiply-Add
            '-falign-functions=64',     # Align functions for cache lines
            '-std=c99',                 # C99 standard
            '-Wall',                    # All warnings
            '-Wno-unused-function'      # Suppress unused static inline warnings
        ]
        
    return args

# Define the Extension
crayon_core = Extension(
    name="crayon.c_ext._core",
    sources=[
        "src/crayon/c_ext/crayon_module.c",
        "src/crayon/c_ext/simd_ops.c"
    ],
    include_dirs=["src/crayon/c_ext"],
    extra_compile_args=get_compile_args(),
    optional=False  # Fail installation if C extension cannot be built
)

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------

setup(
    name="xerv-crayon",
    version="1.0.0",
    description="Production-grade tokenizer with AVX2 optimizations",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    author="Xerv Research Engineering Division",
    author_email="research@xerv.com",
    url="https://github.com/xerv/crayon",
    license="MIT",
    
    # Source layout
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    
    # Compile C extension
    ext_modules=[crayon_core],
    
    # Dependencies (Zero dependencies as per paper specs)
    install_requires=[],
    
    python_requires=">=3.12",
    zip_safe=False,  # C-extensions cannot be zipped safely
)