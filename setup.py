
import sys
import platform
from setuptools import setup, Extension, find_packages

# Determine Compiler Flags for AVX2 Support
extra_compile_args = []
if platform.system() == "Windows":
    # MSVC Flags
    extra_compile_args = ["/O2", "/arch:AVX2"]
else:
    # GCC / Clang Flags
    extra_compile_args = ["-O3", "-march=native", "-fPIC"]

# Define the Extension
crayon_fast = Extension(
    'crayon.c_ext.crayon_fast',
    sources=['src/crayon/c_ext/engine.cpp'],
    extra_compile_args=extra_compile_args,
    language='c++'
)

setup(
    name='xerv-crayon',
    version='2.0.0',
    description='Hyper-Production Tokenizer with AVX2 Backend',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    ext_modules=[crayon_fast],
)