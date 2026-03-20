"""
XERV CRAYON SETUP v5.2.6 - WITH C++ EXTENSIONS
==============================================
Builds native extensions for maximum performance
"""

import os
import sys
import platform
from pathlib import Path
from setuptools import setup, find_packages, Extension
from setuptools.command.build_ext import build_ext

VERSION = "5.2.6"

class CustomBuildExt(build_ext):
    """Custom build extension to handle platform-specific compilation"""
    
    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except Exception as e:
            print(f"Warning: Failed to build {ext.name}: {e}")
            print("Falling back to pure Python implementation...")
            # Continue without this extension

def get_extensions():
    """Get list of extensions to build"""
    extensions = []
    
    # Get source directory
    script_dir = Path(__file__).parent
    c_ext_dir = script_dir / "src" / "crayon" / "c_ext"
    
    # CPU Extension (always try to build)
    cpu_sources = [
        str(c_ext_dir / "crayon_module.c"),
        str(c_ext_dir / "simd_ops.c"),
        str(c_ext_dir / "cpu_engine.cpp"),
    ]
    
    cpu_ext = Extension(
        'crayon.c_ext.crayon_cpu',
        sources=cpu_sources,
        include_dirs=[str(c_ext_dir)],
        extra_compile_args=['-O3', '-march=native'] if platform.system() != 'Windows' else ['/O2'],
        language='c++'
    )
    extensions.append(cpu_ext)
    
    # CUDA Extension (optional)
    if os.environ.get('CUDA_HOME') or shutil.which('nvcc'):
        cuda_sources = [
            str(c_ext_dir / "gpu_engine_cuda.cu"),
        ]
        
        cuda_ext = Extension(
            'crayon.c_ext.crayon_cuda',
            sources=cuda_sources,
            include_dirs=[str(c_ext_dir)],
            extra_compile_args=['-O3'],
            language='c++'
        )
        extensions.append(cuda_ext)
    
    return extensions

# Check if we should build extensions
build_extensions = '--no-extensions' not in sys.argv

if build_extensions:
    try:
        extensions = get_extensions()
        print(f"Building {len(extensions)} extension(s)...")
    except Exception as e:
        print(f"Warning: Extension setup failed: {e}")
        print("Falling back to pure Python...")
        extensions = []
else:
    extensions = []
    print("Skipping extension build (using pure Python)")

setup(
    name="xerv-crayon",
    version=VERSION,
    packages=find_packages("src"),
    package_dir={"": "src"},
    python_requires=">=3.8,<3.14",
    install_requires=[
        "numpy>=1.21.0",
    ],
    ext_modules=extensions if build_extensions else [],
    cmdclass={'build_ext': CustomBuildExt} if build_extensions else {},
    package_data={
        "crayon": [
            "resources/dat/vocab_lite.dat",
            "resources/dat/vocab_lite.json", 
            "resources/dat/vocab_standard.dat",
            "resources/dat/vocab_standard.json",
            "resources/*.txt",
            "resources/*.csv",
            "c_ext/*.h",
            "c_ext/*.c",
            "c_ext/*.cpp",
            "c_ext/*.cu",
            "c_ext/*.hip",
            "c_ext/*.pyd",
            "c_ext/*.so"
        ]
    },
)
