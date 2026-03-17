"""
XERV CRAYON SETUP v5.1.1 - Production Omni-Backend Build System
================================================================
Fixed for PyTorch 2.10+ and seamless CUDA compilation
"""

import os
import sys
import subprocess
import shutil
from setuptools import setup, Extension, find_packages
from setuptools.command.build_ext import build_ext
from distutils.sysconfig import get_python_inc

VERSION = "5.1.3"

def log(msg: str, level: str = "INFO") -> None:
    print(f"[CRAYON-BUILD] {msg}", flush=True)

# ============================================================================
# CUDA DETECTION AND COMPILATION
# ============================================================================

FORCE_CPU = os.environ.get("CRAYON_FORCE_CPU", "0") == "1"
os.environ["MAX_JOBS"] = os.environ.get("MAX_JOBS", "1")

# Detect PyTorch & CUDA - FIXED FOR PYTORCH 2.10+
try:
    import torch
    
    # Try multiple import paths for PyTorch 2.10+
    CUDAExtension = None
    BuildExtension = None
    CUDA_HOME = None
    
    # Method 1: Old path (PyTorch < 2.10)
    try:
        from torch.utils.cpp_extension import CUDAExtension, BuildExtension, CUDA_HOME
        log("Using old PyTorch cpp_extension import")
    except ImportError:
        pass
    
    # Method 2: New path (PyTorch 2.10+)
    if CUDAExtension is None:
        try:
            from torch.cuda.cpp_extension import CUDAExtension, BuildExtension, CUDA_HOME
            log("Using new PyTorch cpp_extension import")
        except ImportError:
            pass
    
    # Method 3: Direct import
    if CUDAExtension is None:
        try:
            import torch.cuda.cpp_extension as cpp_ext
            CUDAExtension = cpp_ext.CUDAExtension
            BuildExtension = cpp_ext.BuildExtension
            CUDA_HOME = cpp_ext.CUDA_HOME
            log("Using direct torch.cuda.cpp_extension import")
        except ImportError:
            pass
    
    # Method 4: Manual detection
    if CUDAExtension is None:
        try:
            CUDA_HOME = torch.cuda.cuda_config().get('cuda_include_path', None) or '/usr/local/cuda'
            # Try to import from torch.cuda
            if hasattr(torch.cuda, 'cpp_extension'):
                cpp_ext = torch.cuda.cpp_extension
                CUDAExtension = getattr(cpp_ext, 'CUDAExtension', None)
                BuildExtension = getattr(cpp_ext, 'BuildExtension', None)
            log("Using manual PyTorch cpp_extension detection")
        except:
            pass
    
    # Check if we got the extensions
    if CUDAExtension is not None and BuildExtension is not None:
        FORCE_CUDA = os.environ.get("CRAYON_FORCE_CUDA", "0") == "1"
        TORCH_CUDA_AVAILABLE = (torch.cuda.is_available() or FORCE_CUDA) and (CUDA_HOME is not None)
        
        if TORCH_CUDA_AVAILABLE:
            log(f"PyTorch v{torch.__version__} with CUDA detected")
            if torch.cuda.is_available():
                log(f"GPU: {torch.cuda.get_device_name(0)}")
            elif FORCE_CUDA:
                log("Forced CUDA build (CRAYON_FORCE_CUDA=1)")
        elif CUDA_HOME:
            log(f"CUDA_HOME found at {CUDA_HOME} but PyTorch CUDA not available")
        else:
            log("CUDA_HOME not found - CUDA extensions will not be built")
    else:
        TORCH_CUDA_AVAILABLE = False
        log("PyTorch CUDA extension not available")
        
except ImportError:
    TORCH_CUDA_AVAILABLE = False
    CUDAExtension = None
    BuildExtension = None
    CUDA_HOME = None
    log("PyTorch not installed - CUDA extensions will not be built")

# ============================================================================
# CUSTOM CUDA BUILD CLASS
# ============================================================================

class CrayonBuildExt(build_ext):
    """Custom build class that handles CUDA compilation with proper flags"""
    
    def build_extension(self, ext):
        if ext.name == "crayon.c_ext.crayon_cuda":
            self._build_cuda_extension(ext)
        else:
            super().build_extension(ext)
    
    def _build_cuda_extension(self, ext):
        """Build CUDA extension with proper flags"""
        log(f"Building CUDA extension: {ext.name}")
        
        # Get paths
        python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
        python_include = f"/usr/include/python{python_version}"
        
        # Try to find torch include
        torch_include_paths = []
        try:
            import torch
            torch_path = os.path.dirname(torch.__file__)
            torch_include_paths.append(f"{torch_path}/include")
            torch_include_paths.append(f"{torch_path}/include/torch/csrc/api/include")
        except ImportError:
            pass
        
        # CUDA include
        cuda_include = "/usr/local/cuda/include"
        
        # Build command
        cmd = [
            "nvcc",
            "-O3", "-std=c++17",
            "--compiler-options", "-fPIC",
            "-shared",
            "-o", self.get_ext_fullname(ext.name).replace('.', '/') + ".so",
            ext.sources[0],
            f"-I{python_include}",
            f"-I{cuda_include}",
            "-D_GLIBCXX_USE_CXX11_ABI=0"
        ]
        
        # Add torch includes
        for inc_path in torch_include_paths:
            if os.path.exists(inc_path):
                cmd.append(f"-I{inc_path}")
        
        # Add extra compile args if specified
        if hasattr(ext, 'extra_compile_args') and 'nvcc' in ext.extra_compile_args:
            cmd.extend(ext.extra_compile_args['nvcc'])
        
        log(f"CUDA build command: {' '.join(cmd)}")
        
        try:
            # Create output directory
            output_dir = os.path.dirname(self.get_ext_fullname(ext.name).replace('.', '/'))
            os.makedirs(os.path.join(self.build_lib, output_dir), exist_ok=True)
            
            # Run compilation
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.build_lib)
            if result.returncode == 0:
                log(f"✓ CUDA extension {ext.name} built successfully")
            else:
                log(f"✗ CUDA build failed: {result.stderr}")
                raise RuntimeError(f"CUDA compilation failed: {result.stderr}")
        except Exception as e:
            log(f"CUDA build error: {e}")
            raise

# ============================================================================
# EXTENSION CONFIGURATION
# ============================================================================

ext_modules = []

# CPU Extensions (always built)
if sys.platform == "win32":
    cpu_args = ["/O2", "/std:c++17"]
else:
    cpu_args = ["-O3", "-fPIC", "-std=c++17"]

ext_modules.append(Extension(
    "crayon.c_ext.crayon_cpu",
    sources=["src/crayon/c_ext/cpu_engine.cpp"],
    extra_compile_args=cpu_args,
    language="c++",
))

ext_modules.append(Extension(
    "crayon.c_ext.crayon_trainer", 
    sources=["src/crayon/c_ext/trainer.cpp"],
    extra_compile_args=cpu_args,
    language="c++",
))

ext_modules.append(Extension(
    "crayon.c_ext.crayon_compiler",
    sources=["src/crayon/c_ext/compiler.cpp"],
    extra_compile_args=cpu_args,
    language="c++",
))

# CUDA Extension (if available)
if TORCH_CUDA_AVAILABLE and not FORCE_CPU and CUDAExtension:
    log("Adding CUDA extension to build")
    
    # Get GPU architecture
    try:
        major, minor = torch.cuda.get_device_capability()
        arch = f"{major}{minor}"
        cuda_flags = ["-O3", "-std=c++17", "--expt-relaxed-constexpr"]
        cuda_flags.append(f"-gencode=arch=compute_{arch},code=sm_{arch}")
        log(f"Compiling for GPU architecture: sm_{arch}")
    except:
        cuda_flags = ["-O3", "-std=c++17", "--expt-relaxed-constexpr", "-gencode=arch=compute_75,code=sm_75"]
        log("Using default GPU architecture: sm_75")
    
    # Use our custom build class
    ext_modules.append(Extension(
        "crayon.c_ext.crayon_cuda",
        sources=["src/crayon/c_ext/gpu_engine_cuda.cu"],
        extra_compile_args={"nvcc": cuda_flags},
        language="c++",
    ))
    
    # Use custom build class
    cmdclass = {"build_ext": CrayonBuildExt}
    log("Using custom CUDA build class")
    
else:
    cmdclass = {}
    if not FORCE_CPU:
        log("Skipping CUDA extension - not available")

# ============================================================================
# SETUP
# ============================================================================

setup(
    name="xerv-crayon",
    version=VERSION,
    packages=find_packages("src"),
    package_dir={"": "src"},
    include_package_data=True,
    ext_modules=ext_modules,
    cmdclass=cmdclass,
    python_requires=">=3.8",
    zip_safe=False,
)