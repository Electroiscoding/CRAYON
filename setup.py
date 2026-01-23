"""
XERV Crayon - Production Build System
======================================
Handles C++ extension compilation with AVX2/SIMD optimizations across all platforms.
"""
import sys
import os
import platform
from pathlib import Path
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext


class AVX2BuildExt(build_ext):
    """Custom build extension that properly handles AVX2 compilation."""
    
    def build_extensions(self):
        # Detect compiler
        compiler_type = self.compiler.compiler_type
        
        for ext in self.extensions:
            if compiler_type == 'msvc':
                # Microsoft Visual C++ (Windows)
                ext.extra_compile_args = [
                    '/O2',           # Maximum optimization
                    '/arch:AVX2',    # Enable AVX2 SIMD
                    '/EHsc',         # Exception handling
                    '/std:c++17',    # C++17 standard
                    '/DNDEBUG',      # Disable debug assertions
                ]
            elif compiler_type in ('unix', 'mingw32'):
                # GCC / Clang (Linux, macOS, MinGW)
                ext.extra_compile_args = [
                    '-O3',                    # Maximum optimization
                    '-march=native',          # Use native CPU features (includes AVX2 if available)
                    '-fPIC',                  # Position independent code
                    '-std=c++17',             # C++17 standard
                    '-DNDEBUG',               # Disable debug assertions
                    '-ffast-math',            # Fast floating point
                    '-funroll-loops',         # Unroll loops
                ]
                
                # macOS specific
                if platform.system() == 'Darwin':
                    # Check if we're on Apple Silicon or Intel
                    if platform.machine() == 'arm64':
                        # Apple Silicon - use ARM NEON instead of AVX2
                        ext.extra_compile_args = [
                            '-O3',
                            '-std=c++17',
                            '-DNDEBUG',
                            '-ffast-math',
                            '-funroll-loops',
                        ]
                    else:
                        # Intel Mac
                        ext.extra_compile_args.append('-mavx2')
                else:
                    # Linux - explicitly enable AVX2
                    ext.extra_compile_args.append('-mavx2')
                    
        build_ext.build_extensions(self)


def get_extension_modules():
    """Define C++ extension modules."""
    
    extensions = []
    
    # Main AVX2 engine (Double-Array Trie)
    # MUST USE RELATIVE PATHS for setuptools
    engine_cpp = 'src/crayon/c_ext/engine.cpp'
    
    if os.path.exists(engine_cpp):
        extensions.append(
            Extension(
                'crayon.c_ext.crayon_fast',
                sources=[engine_cpp],
                language='c++',
            )
        )
    
    # Legacy C Trie engine (fallback)
    crayon_module_c = 'src/crayon/c_ext/crayon_module.c'
    simd_ops_c = 'src/crayon/c_ext/simd_ops.c'
    
    if os.path.exists(crayon_module_c) and os.path.exists(simd_ops_c):
        extensions.append(
            Extension(
                'crayon.c_ext._core',
                sources=[crayon_module_c, simd_ops_c],
                language='c',
            )
        )
    
    return extensions


# Read long description from README
def get_long_description():
    readme_path = Path(__file__).parent / 'README.md'
    if readme_path.exists():
        return readme_path.read_text(encoding='utf-8')
    return ''


if __name__ == '__main__':
    setup(
        name='xerv-crayon',
        version='2.0.0',
        description='Production-grade tokenizer achieving >16M tokens/s via AVX2/SIMD optimizations.',
        long_description=get_long_description(),
        long_description_content_type='text/markdown',
        author='Soham Pal',
        author_email='botmaker583@gmail.com',
        url='https://github.com/xerv/crayon',
        license='MIT',
        packages=find_packages('src'),
        package_dir={'': 'src'},
        ext_modules=get_extension_modules(),
        cmdclass={'build_ext': AVX2BuildExt},
        include_package_data=True,
        package_data={
            'crayon': [
                'resources/dat/*.dat',
                'resources/*.txt',
                'resources/*.csv',
                'c_ext/*.h',
                'c_ext/*.c',
                'c_ext/*.cpp',
            ],
        },
        python_requires='>=3.10',
        classifiers=[
            'Development Status :: 5 - Production/Stable',
            'Intended Audience :: Science/Research',
            'Intended Audience :: Developers',
            'License :: OSI Approved :: MIT License',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.10',
            'Programming Language :: Python :: 3.11',
            'Programming Language :: Python :: 3.12',
            'Programming Language :: Python :: 3.13',
            'Programming Language :: C',
            'Programming Language :: C++',
            'Topic :: Scientific/Engineering :: Artificial Intelligence',
            'Topic :: Text Processing :: Linguistic',
            'Operating System :: POSIX :: Linux',
            'Operating System :: Microsoft :: Windows',
            'Operating System :: MacOS',
        ],
        zip_safe=False,  # Required for C extensions
    )