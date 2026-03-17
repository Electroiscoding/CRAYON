# CRAYON Installation Guide

## Quick Installation

### Option 1: Install from PyPI (Recommended)
```bash
pip install xerv-crayon
```

### Option 2: If PyPI Installation Fails

#### Method A: Force Wheel Installation
```bash
pip install --only-binary=:all: xerv-crayon
```

#### Method B: Install from GitHub (Latest)
```bash
pip install git+https://github.com/Electroiscoding/CRAYON.git
```

#### Method C: Manual Build (Advanced)
```bash
git clone https://github.com/Electroiscoding/CRAYON.git
cd CRAYON
pip install -e .
```

## Troubleshooting

### If you get build errors:
1. **Install Visual Studio Build Tools** (Windows):
   - Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/
   - Select "C++ build tools" during installation

2. **Install CUDA Toolkit** (for GPU support):
   - Download from: https://developer.nvidia.com/cuda-downloads
   - Set environment variable: `set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x`

3. **Force CPU-only installation**:
   ```bash
   set CRAYON_FORCE_CPU=1
   pip install xerv-crayon
   ```

### Python Version Requirements
- **Minimum**: Python 3.8
- **Recommended**: Python 3.10+
- **Tested**: 3.8, 3.9, 3.10, 3.11, 3.12, 3.13

## Quick Test

```python
from crayon import CrayonVocab

# Auto-detects hardware (CUDA if available, else CPU)
tokenizer = CrayonVocab(device="auto")
tokenizer.load_profile("standard")
tokens = tokenizer.tokenize("Hello, world!")
print(f"Device: {tokenizer.device}")
print(f"Tokens: {tokens}")
```

## Features

- ✅ **Automatic Hardware Detection**: CUDA, ROCm, or CPU
- ✅ **Seamless Fallback**: CPU if GPU unavailable
- ✅ **Detailed Error Messages**: Actionable debugging info
- ✅ **Cross-Platform**: Windows, Linux, macOS
- ✅ **Multiple Python Versions**: 3.8+ support
- ✅ **High Performance**: AVX2/AVX-512 CPU, GPU acceleration
