"""
XERV CRAYON SETUP v5.3.5 - WITH C++ EXTENSIONS
==============================================
Builds native extensions for maximum performance on CPU (AVX2), CUDA, and ROCm
"""

import os
import sys
import platform
import shutil
import sysconfig
import subprocess
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

VERSION = "5.4.2"

class CustomBuildExt(build_ext):
    """Custom build extension with CUDA support and fallback for missing compilers"""
    
    def build_extension(self, ext):
        try:
            # Special handling for CUDA extensions
            if ext.name.endswith('_cuda'):
                self._build_cuda_extension(ext)
            else:
                super().build_extension(ext)
            print(f"Successfully built: {ext.name}")
        except Exception as e:
            print(f"Warning: Failed to build {ext.name}: {e}")
    
    def _build_cuda_extension(self, ext):
        """Build CUDA extension using nvcc"""
        cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
        nvcc = shutil.which('nvcc') or (os.path.join(cuda_home, 'bin', 'nvcc') if cuda_home else None)
        
        if not nvcc or not os.path.exists(nvcc):
            raise RuntimeError("NVCC not found")
        
        # Build directory
        build_temp = os.path.join(self.build_temp, ext.name)
        os.makedirs(build_temp, exist_ok=True)
        
        # Output directory  
        build_lib = os.path.join(self.build_lib, 'crayon', 'c_ext')
        os.makedirs(build_lib, exist_ok=True)
        
        # Source file
        cuda_src = ext.sources[0]
        
        # Object file
        obj_file = os.path.join(build_temp, 'cuda_engine.o')
        
        # Library file
        lib_name = f"{ext.name}{sysconfig.get_config_var('EXT_SUFFIX')}"
        lib_file = os.path.join(build_lib, lib_name)
        
        # Include directories
        include_dirs = [
            sysconfig.get_paths()['include'],  # Python headers
            os.path.join(os.path.dirname(nvcc), '..', 'include'),  # CUDA headers
        ]
        include_flags = ' '.join(f'-I"{d}"' for d in include_dirs if os.path.exists(d))
        
        # CUDA architecture flags (compile for common GPUs)
        gpu_arch_flags = '-gencode=arch=compute_70,code=sm_70 ' \
                       '-gencode=arch=compute_75,code=sm_75 ' \
                       '-gencode=arch=compute_80,code=sm_80 ' \
                       '-gencode=arch=compute_86,code=sm_86 ' \
                       '-gencode=arch=compute_89,code=sm_89 ' \
                       '-gencode=arch=compute_90,code=sm_90'
        
        # Compile CUDA to object
        compile_cmd = f'"{nvcc}" -c "{cuda_src}" -o "{obj_file}" {include_flags} ' \
                      f'-O3 --compiler-options "-fPIC" -std=c++17 {gpu_arch_flags}'
        
        print(f"Compiling CUDA extension: {compile_cmd}")
        subprocess.check_call(compile_cmd, shell=True)
        
        # Link into shared library
        link_cmd = f'"{nvcc}" -shared "{obj_file}" -o "{lib_file}" ' \
                   f'-L"{os.path.join(os.path.dirname(nvcc), "..", "lib64")}" -lcudart'
        
        print(f"Linking CUDA extension: {link_cmd}")
        subprocess.check_call(link_cmd, shell=True)
        
        # Copy to final destination
        dest_file = os.path.join(self.get_ext_fullpath(ext.name))
        os.makedirs(os.path.dirname(dest_file), exist_ok=True)
        shutil.copy2(lib_file, dest_file)

def get_extensions():
    """Get list of C/C++ extensions to build"""
    extensions = []
    
    # Use relative paths from setup.py location
    c_ext_dir = os.path.join("src", "crayon", "c_ext")
    
    # CPU EXTENSION
    cpu_sources = []
    cpu_engine_path = os.path.join(c_ext_dir, "cpu_engine.cpp")
    crayon_module_path = os.path.join(c_ext_dir, "crayon_module.c")
    simd_ops_path = os.path.join(c_ext_dir, "simd_ops.c")
    
    if os.path.exists(cpu_engine_path):
        cpu_sources.append(cpu_engine_path)
    elif os.path.exists(crayon_module_path):
        cpu_sources.extend([crayon_module_path, simd_ops_path])
    
    if cpu_sources:
        if platform.system() == 'Windows':
            extra_args = ['/O2', '/std:c++17', '/W3', '/wd4244', '/wd4267']
        else:
            extra_args = ['-O3', '-std=c++17', '-fPIC', '-Wall']
            if platform.machine() in ('x86_64', 'AMD64'):
                extra_args.extend(['-mavx2', '-mfma'])
        
        cpu_ext = Extension(
            'crayon.c_ext.crayon_cpu',
            sources=cpu_sources,
            include_dirs=[c_ext_dir],
            extra_compile_args=extra_args,
            language='c++'
        )
        extensions.append(cpu_ext)
    
    # TURBO ENGINE (Ultra-fast C engine with OpenMP + word cache)
    turbo_src = os.path.join(c_ext_dir, "crayon_turbo.c")
    if os.path.exists(turbo_src):
        if platform.system() == 'Windows':
            turbo_args = ['/O2', '/W3', '/wd4244', '/wd4267', '/openmp']
            turbo_link_args = []
        else:
            turbo_args = ['-O3', '-fPIC', '-Wall', '-Wno-unused-function',
                          '-ffast-math', '-funroll-loops']
            turbo_link_args = []
            if platform.machine() in ('x86_64', 'AMD64'):
                turbo_args.extend(['-mavx2', '-mfma'])
            if platform.system() != 'Darwin':
                turbo_args.append('-fopenmp')
                turbo_link_args.append('-fopenmp')
        
        # Get numpy include path for the C API
        try:
            import numpy as np
            numpy_inc = np.get_include()
            turbo_inc = [c_ext_dir, numpy_inc]
            print(f"Turbo engine: numpy include = {numpy_inc}")
        except ImportError:
            turbo_inc = [c_ext_dir]
            print("Turbo engine: numpy not found — numpy output disabled")
        
        turbo_ext = Extension(
            'crayon.c_ext.crayon_turbo',
            sources=[turbo_src],
            include_dirs=turbo_inc,
            extra_compile_args=turbo_args,
            extra_link_args=turbo_link_args,
            language='c'
        )
        extensions.append(turbo_ext)
        print(f"Turbo engine configured: {turbo_src}")
    
    # CUDA EXTENSION (Linux only - requires nvcc)
    if platform.system() != 'Windows':
        cuda_home = os.environ.get('CUDA_HOME') or os.environ.get('CUDA_PATH')
        nvcc = shutil.which('nvcc') or (os.path.join(cuda_home, 'bin', 'nvcc') if cuda_home else None)
        cuda_src = os.path.join(c_ext_dir, "gpu_engine_cuda.cu")
        
        if nvcc and os.path.exists(nvcc) and os.path.exists(cuda_src) and not os.environ.get('CRAYON_SKIP_CUDA'):
            cuda_ext = Extension(
                'crayon.c_ext.crayon_cuda',
                sources=[cuda_src],
                include_dirs=[c_ext_dir],
                language='c++'
            )
            extensions.append(cuda_ext)
            print(f"CUDA extension configured (NVCC: {nvcc})")
    
    return extensions

build_extensions = '--no-extensions' not in sys.argv

if build_extensions:
    try:
        extensions = get_extensions()
    except Exception as e:
        print(f"Extension setup failed: {e}")
        extensions = []
else:
    extensions = []
    sys.argv.remove('--no-extensions')

setup(
    name="xerv-crayon",
    version="5.5.4",
    author="Xerv Research Engineering Division",
    description="Omni-Backend Tokenizer - CPU (AVX2/512), CUDA (NVIDIA), ROCm (AMD)",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_packages("src"),
    package_dir={"": "src"},
    python_requires=">=3.8",
    setup_requires=["numpy>=1.21.0"],
    install_requires=["numpy>=1.21.0"],
    ext_modules=extensions,
    cmdclass={'build_ext': CustomBuildExt},
    package_data={
        "crayon": [
            "resources/dat/*.dat",
            "resources/dat/*.json",
            "resources/*.txt",
            "c_ext/*.h",
            "c_ext/*.c",
            "c_ext/*.cpp",
            "c_ext/*.cu",
            "c_ext/*.hip",
        ]
    },
    include_package_data=True,
)
