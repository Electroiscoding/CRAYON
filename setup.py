"""
XERV CRAYON SETUP v5.2.3 - PRODUCTION BUILD
============================================
With CPU extensions for guaranteed performance
"""

import os
import sys
from setuptools import setup, Extension, find_packages

VERSION = "5.2.4"

def log(msg: str) -> None:
    print(f"[CRAYON-BUILD] {msg}", flush=True)

# Compiler flags
if sys.platform == "win32":
    cpu_cflags = ["/O2", "/std:c++17"]
else:
    cpu_cflags = ["-O3", "-fPIC", "-std=c++17"]

# CPU Extensions (always built)
ext_modules = []

log("Adding CPU extensions...")

ext_modules.append(Extension(
    "crayon.c_ext.crayon_cpu",
    sources=["src/crayon/c_ext/cpu_engine.cpp"],
    extra_compile_args=cpu_cflags,
    language="c++",
))

ext_modules.append(Extension(
    "crayon.c_ext.crayon_trainer",
    sources=["src/crayon/c_ext/trainer.cpp"],
    extra_compile_args=cpu_cflags,
    language="c++",
))

ext_modules.append(Extension(
    "crayon.c_ext.crayon_compiler",
    sources=["src/crayon/c_ext/compiler.cpp"],
    extra_compile_args=cpu_cflags,
    language="c++",
))

setup(
    name="xerv-crayon",
    version=VERSION,
    packages=find_packages("src"),
    package_dir={"": "src"},
    include_package_data=True,
    ext_modules=ext_modules,
    python_requires=">=3.8,<3.14",
    install_requires=[
        "numpy>=1.21.0",
    ],
)
