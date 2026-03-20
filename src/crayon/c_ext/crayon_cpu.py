"""
CPU Backend Wrapper for Crayon
================================

This module wraps the compiled C++ extension for CPU tokenization.
Falls back to pure Python implementation if extension is not available.
"""

import sys
import os
import importlib.util

def _load_cpu_extension():
    """Load compiled CPU extension with platform-specific naming."""
    
    # Determine extension filename based on platform
    if sys.platform == "win32":
        # Windows: look for .pyd files with version info
        search_dirs = [
            os.path.dirname(__file__),  # Current directory
            os.path.join(os.path.dirname(__file__), "compiled"),  # Compiled subdirectory
        ]
        
        ext_file = None
        search_dir = None
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                ext_files = [f for f in os.listdir(search_dir) 
                           if f.startswith('crayon_cpu') and f.endswith('.pyd')]
                if ext_files:
                    ext_file = ext_files[0]
                    break
        
        if not ext_file:
            raise ImportError("No compiled CPU extension found (.pyd file)")
    else:
        # Linux/macOS: look for .so files
        search_dirs = [
            os.path.dirname(__file__),  # Current directory
            os.path.join(os.path.dirname(__file__), "compiled"),  # Compiled subdirectory
        ]
        
        ext_file = None
        search_dir = None
        for search_dir in search_dirs:
            if os.path.exists(search_dir):
                ext_files = [f for f in os.listdir(search_dir) 
                           if f.startswith('crayon_cpu') and f.endswith('.so')]
                if ext_files:
                    ext_file = ext_files[0]
                    break
        
        if not ext_file:
            # Try to find any .so file as last resort
            for search_dir in search_dirs:
                if os.path.exists(search_dir):
                    so_files = [f for f in os.listdir(search_dir) if f.endswith('.so')]
                    if so_files:
                        ext_file = so_files[0]
                        print(f"🔍 Found .so file: {ext_file}")
                        break
            
            if not ext_file:
                raise ImportError("No compiled CPU extension found (.so file)")
    
    # Load extension
    ext_path = os.path.join(search_dir, ext_file)
    print(f"🔍 Loading CPU extension from: {ext_path}")
    
    try:
        # Try direct import first
        ext_dir = os.path.dirname(ext_path)
        if ext_dir not in sys.path:
            sys.path.insert(0, ext_dir)
        
        # Remove .pyd/.so extension for module name
        module_name = os.path.splitext(ext_file)[0]
        
        spec = importlib.util.spec_from_file_location(module_name, ext_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Could not create spec for {ext_path}")
        
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        print(f"✅ Successfully loaded compiled extension: {ext_file}")
        return mod
    except Exception as e:
        # Try alternative loading method
        try:
            import importlib.machinery
            loader = importlib.machinery.ExtensionFileLoader(module_name, ext_path)
            spec = importlib.util.spec_from_file_location(module_name, ext_path, loader=loader)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            print(f"✅ Successfully loaded compiled extension (alt method): {ext_file}")
            return mod
        except Exception as e2:
            raise ImportError(f"Failed to load extension {ext_path}: {e}\nAlternative method failed: {e2}")

# Try to load the compiled extension
try:
    _cpu_ext = _load_cpu_extension()
    print("✓ Using compiled C++ extension for maximum performance")
except ImportError as e:
    print(f"⚠ Compiled extension not available: {e}")
    print("🔄 Falling back to pure Python implementation (slower but functional)")
    
    # Load pure Python fallback
    try:
        from . import crayon_cpu_fallback as _cpu_ext
        print("✓ Pure Python fallback loaded successfully")
    except ImportError as fallback_error:
        raise ImportError(
            f"Failed to load both compiled extension and pure Python fallback:\n"
            f"Extension error: {e}\n"
            f"Fallback error: {fallback_error}\n"
            "This suggests a corrupted installation. Try reinstalling with:\n"
            "  pip install --force-reinstall xerv-crayon"
        )

# Export the required functions
tokenize = _cpu_ext.tokenize
load_dat = _cpu_ext.load_dat

# Export hardware info if available
if hasattr(_cpu_ext, 'get_hardware_info'):
    get_hardware_info = _cpu_ext.get_hardware_info
else:
    def get_hardware_info():
        return "CPU Backend [Unknown]"

__all__ = ['tokenize', 'load_dat', 'get_hardware_info']
