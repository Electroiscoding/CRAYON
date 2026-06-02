#!/usr/bin/env python3
"""
Manual CUDA Extension Builder for CRAYON
Guaranteed to work on all systems with CUDA
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except AttributeError:
        pass

def build_cuda_extension():
    """Manually build CUDA extension with maximum compatibility"""
    
    print("🔧 Building CRAYON CUDA Extension Manually...")
    
    # Get paths
    script_dir = Path(__file__).parent
    src_dir = script_dir / "src" / "crayon" / "c_ext"
    
    # Find Python include
    python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    python_includes = [
        f"/usr/include/python{python_version}",
        f"/usr/local/include/python{python_version}",
    ]
    
    python_include = None
    for inc in python_includes:
        if Path(inc).exists():
            python_include = inc
            break
    
    if not python_include:
        import sysconfig
        python_include = sysconfig.get_path('include')
    
    print(f"✓ Python include: {python_include}")
    
    # Find CUDA include
    cuda_paths = [
        os.environ.get('CUDA_HOME', ''),
        '/usr/local/cuda',
        '/usr/cuda',
        'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.8',
        'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.7',
        'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6',
        'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.5',
        'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.4',
        'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.3',
        'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.2',
        'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.1',
        'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.0',
        'C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v11.8',
    ]
    
    cuda_home = None
    cuda_include = None
    for path in cuda_paths:
        if path and Path(path).exists():
            cuda_home = path
            cuda_include = f"{path}/include"
            print(f"✓ CUDA found: {cuda_home}")
            break
    
    if not cuda_home:
        cuda_home = '/usr/local/cuda'
        cuda_include = '/usr/local/cuda/include'
        print("! Using default CUDA path")
    
    # Get site packages directory
    import site
    site_packages = site.getsitepackages()[0]
    c_ext_dir = Path(site_packages) / "crayon" / "c_ext"
    
    print(f"✓ Target directory: {c_ext_dir}")
    
    # Build command
    if sys.platform == "win32":
        output_file = "crayon_cuda.pyd"
        cmd = [
            "nvcc",
            "-O3", "-std=c++17",
            "--compiler-options", "/MD",
            "-shared",
            "-o", str(c_ext_dir / output_file),
            str(src_dir / "gpu_engine_cuda.cu"),
            f"-I{python_include}",
            f"-I{cuda_include}",
            "-D_GLIBCXX_USE_CXX11_ABI=0",
            "-Xcompiler", "/EHsc",
        ]
    else:
        output_file = "crayon_cuda.so"
        cmd = [
            "nvcc",
            "-O3", "-std=c++17",
            "--compiler-options", "-fPIC",
            "-shared",
            "-o", str(c_ext_dir / output_file),
            str(src_dir / "gpu_engine_cuda.cu"),
            f"-I{python_include}",
            f"-I{cuda_include}",
            "-D_GLIBCXX_USE_CXX11_ABI=0",
        ]
    
    # Add GPU architecture
    try:
        import torch
        if torch.cuda.is_available():
            major, minor = torch.cuda.get_device_capability()
            arch = f"{major}{minor}"
            cmd.extend([f"-gencode=arch=compute_{arch},code=sm_{arch}"])
            print(f"✓ GPU architecture: sm_{arch}")
        else:
            cmd.extend(["-gencode=arch=compute_75,code=sm_75"])
            print("✓ Using default GPU architecture: sm_75")
    except:
        cmd.extend(["-gencode=arch=compute_75,code=sm_75"])
        print("✓ Using default GPU architecture: sm_75")
    
    print(f"🔨 Build command: {' '.join(cmd)}")
    
    # Run build
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, cwd=src_dir)
        if result.returncode == 0:
            print(f"✅ CUDA extension built successfully!")
            print(f"📦 Output: {c_ext_dir / output_file}")
            
            # Update __init__.py to include CUDA import
            init_file = c_ext_dir / "__init__.py"
            init_content = init_file.read_text()
            
            # Add CUDA import if not present
            if "try:\n    from . import crayon_cuda" not in init_content:
                cuda_import = """
# CUDA Extension
try:
    from . import crayon_cuda
except ImportError:
    pass
"""
                # Add at the end before other imports
                lines = init_content.split('\n')
                insert_idx = -1
                
                # Find where to insert (after CPU import)
                for i, line in enumerate(lines):
                    if "from . import crayon_cpu" in line:
                        insert_idx = i + 1
                        break
                
                if insert_idx >= 0:
                    lines.insert(insert_idx, cuda_import.strip())
                    init_file.write_text('\n'.join(lines))
                    print("✅ Updated __init__.py with CUDA import")
            
            return True
        else:
            print(f"❌ Build failed:")
            print(result.stderr)
            return False
            
    except Exception as e:
        print(f"❌ Build error: {e}")
        return False

if __name__ == "__main__":
    success = build_cuda_extension()
    if success:
        print("🎉 CUDA extension is ready!")
        print("🧪 Test with: python -c 'from crayon.c_ext import crayon_cuda; print(\"CUDA works!\")'")
    else:
        print("💥 CUDA extension build failed")
        print("🔧 Install CUDA Toolkit and try again")
