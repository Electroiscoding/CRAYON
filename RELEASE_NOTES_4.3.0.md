# XERV CRAYON v4.3.0 Release Notes
## Release Date: February 1, 2026

---

## 🚀 Critical Fix: ROCm/HIP Compilation

This release fixes a **critical build failure** on AMD ROCm systems that prevented installation on machines with AMD GPUs.

### Root Cause
The previous build system used `g++` to compile the ROCm engine, but the HIP kernel code (`__global__`, `blockIdx`, `threadIdx`, `hipLaunchKernelGGL`) **requires the `hipcc` compiler**.

Python 3.13's stricter `setuptools` ignored previous compiler override hacks, causing the build to fail with errors like:
```
error: 'blockIdx' was not declared in this scope
error: 'hipLaunchKernelGGL' was not declared in this scope
```

### Solution
1. **Renamed `rocm_engine.cpp` → `rocm_engine.hip`**: The `.hip` extension is the proper file type for HIP source files.

2. **Custom `CrayonBuildExt` class**: A new build extension that explicitly invokes `hipcc` for `.hip` files, bypassing setuptools' default compiler selection.

3. **Production-grade error handling**: All HIP API calls now use proper error checking with `hipGetErrorString()`.

4. **Fixed license format**: Updated `pyproject.toml` to use SPDX license expression (plain string) instead of deprecated table format.

---

## 📦 What's New

### Build System
- ✅ **ROCm/HIP now compiles correctly** with hipcc
- ✅ **Python 3.10-3.13** fully supported
- ✅ **Fixed deprecation warnings** in pyproject.toml
- ✅ **Clean source distribution** with .hip files included

### ROCm Engine (`rocm_engine.hip`)
- ✅ **Proper device initialization** with context creation
- ✅ **Memory cleanup** in module destructor
- ✅ **Bounds checking** in kernels to prevent invalid memory access
- ✅ **Consistent API** with CUDA engine (returns tuple with metadata)

### Colab Notebook
- ✅ **Updated to v4.3.0**
- ✅ **Added hipcc detection** for ROCm environments
- ✅ **Better output formatting** with Quick Start guide

---

## 🔧 Installation

### From PyPI (Recommended)
```bash
pip install xerv-crayon
```

### From Source (For Development)
```bash
git clone https://github.com/Electroiscoding/CRAYON.git
cd CRAYON
pip install -v .
```

### Force Build for Specific Backend
```bash
# CPU only (skip all GPU backends)
CRAYON_FORCE_CPU=1 pip install xerv-crayon

# Force ROCm environment variables
ROCM_HOME=/opt/rocm pip install xerv-crayon
```

---

## 📊 Supported Backends

| Backend | Hardware | Compiler | Status |
|---------|----------|----------|--------|
| CPU | x86_64 (AVX2/512) | g++/MSVC | ✅ Always built |
| CUDA | NVIDIA GPU (SM 7.0+) | nvcc (PyTorch) | ✅ Auto-detected |
| ROCm | AMD GPU (gfx9+) | hipcc | ✅ **Fixed in v4.3.0** |

---

## 🧪 Verification

After installation, verify your backends:
```python
import crayon

print(f"Version: {crayon.get_version()}")
print(f"Backends: {crayon.check_backends()}")

# Quick test
vocab = crayon.CrayonVocab(device="auto")
vocab.load_profile("lite")
tokens = vocab.tokenize("Hello, world!")
print(f"Tokens: {tokens}")
```

---

## 🐛 Bug Fixes

- **Fixed**: ROCm engine compilation failure on Python 3.13+
- **Fixed**: `hipLaunchKernelGGL` not found error
- **Fixed**: License classifier deprecation warnings
- **Fixed**: Uninitialized variable warnings in CPU engine

---

## 📖 Full Changelog

See [CHANGELOG.md](./CHANGELOG.md) for complete history.

---

## 🙏 Credits

Thanks to the community for reporting the ROCm build issue and providing detailed error logs that helped identify the root cause.
