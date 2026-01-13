"""
XERV Crayon Setup Script.

Handles C-extension compilation with platform-specific AVX2 flags.
"""

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
    
    Returns platform-appropriate optimization and SIMD flags.
    """
    system = platform.system()
    args = []
    
    if system == 'Windows':
        # MSVC flags
        args = [
            '/O2',                          # Optimize for speed
            '/arch:AVX2',                   # Enable AVX2 instructions
            '/D_CRT_SECURE_NO_WARNINGS',    # Suppress security warnings
            '/W3',                          # Warning level 3
        ]
    else:
        # GCC/Clang flags (Linux/macOS)
        args = [
            '-O3',                          # Max optimization
            '-mavx2',                       # Enable AVX2 instructions [cite: 512]
            '-mfma',                        # Enable Fused Multiply-Add
            '-falign-functions=64',         # Align functions for cache lines
            '-std=c99',                     # C99 standard
            '-Wall',                        # All warnings
            '-Wno-unused-function',         # Suppress unused static inline warnings
            '-fPIC',                        # Position independent code
        ]
        
    return args


def get_link_args():
    """Get platform-specific linker arguments."""
    system = platform.system()
    if system == 'Windows':
        return []
    else:
        return ['-lm']  # Link math library on Unix


# Define the Extension
crayon_core = Extension(
    name="crayon.c_ext._core",
    sources=[
        "src/crayon/c_ext/crayon_module.c",
        "src/crayon/c_ext/simd_ops.c"
    ],
    include_dirs=["src/crayon/c_ext"],
    extra_compile_args=get_compile_args(),
    extra_link_args=get_link_args(),
    optional=False  # Fail installation if C extension cannot be built
)

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------

# Read README for long description
try:
    with open("README.md", encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "XERV Crayon: High-Performance Tokenizer"

setup(
    name="xerv-crayon",
    version="1.1.0",
    description="Production-grade tokenizer achieving >2M tokens/s via AVX2 optimizations",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Soham Pal",
    author_email="botmaker583@gmail.com",
    url="https://github.com/xerv/crayon",
    license="MIT",
    
    # Source layout
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    
    # Compile C extension
    ext_modules=[crayon_core],
    
    # Dependencies
    install_requires=[],
    extras_require={
        "full": [
            "requests>=2.31.0",
            "datasets>=2.18.0",
            "huggingface-hub>=0.21.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-benchmark>=4.0.0",
        ],
    },
    
    python_requires=">=3.12",
    zip_safe=False,  # C-extensions cannot be zipped safely
    
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: C",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Text Processing :: Linguistic",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS",
    ],
)