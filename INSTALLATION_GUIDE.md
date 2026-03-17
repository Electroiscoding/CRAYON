# CRAYON Installation Guide

## Quick Install (CPU Only)

```bash
pip install xerv-crayon
```

## CUDA Installation (NVIDIA GPUs)

### Prerequisites
1. **NVIDIA GPU** with CUDA support (Pascal architecture or newer)
2. **CUDA Toolkit** 12.1+ recommended
3. **PyTorch with CUDA support**

### Step 1: Install CUDA Toolkit
Download and install from: https://developer.nvidia.com/cuda-downloads

**Windows:**
- Install to default location: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x`
- Add to PATH: `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x\bin`

**Linux:**
```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
```

### Step 2: Install PyTorch CUDA
```bash
# Uninstall CPU-only version first
pip uninstall torch torchvision torchaudio

# Install CUDA version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Step 3: Install CRAYON with CUDA
```bash
# Development install (recommended)
git clone https://github.com/Electroiscoding/CRAYON.git
cd CRAYON
pip install -e . --verbose

# Or production install
pip install xerv-crayon --verbose
```

### Step 4: Verify Installation
```python
from crayon.core.vocabulary import CrayonVocab

# Should show green message if CUDA is available
vocab = CrayonVocab(device="auto")
print(f"Active device: {vocab.device}")
```

## ROCm Installation (AMD GPUs)

### Prerequisites
1. **AMD GPU** with ROCm support
2. **ROCm Toolkit** 5.4+ recommended

### Installation
```bash
# Set ROCm environment
export ROCM_HOME=/opt/rocm
export HIP_VISIBLE_DEVICES=0

# Install CRAYON
pip install -e . --verbose
```

## Troubleshooting

### CUDA Extension Not Compiled

If you see:
```
WARNING:crayon.vocab:CUDA extension not compiled. Falling back to CPU.
```

Run this diagnostic:
```python
from crayon.core.vocabulary import CrayonVocab
vocab = CrayonVocab(device="cpu")  # Initialize first
print(vocab._get_cuda_import_error())  # Get detailed fix instructions
```

### Common Issues

#### 1. "NVCC not found"
**Solution:** Install CUDA Toolkit and add to PATH

#### 2. "PyTorch CUDA not available" 
**Solution:** Install CUDA version of PyTorch:
```bash
pip uninstall torch
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

#### 3. "CUDA_HOME not set"
**Solution:** Set environment variable:
- **Windows:** `CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.x`
- **Linux:** `export CUDA_HOME=/usr/local/cuda`

#### 4. Build fails with "out of memory"
**Solution:** Limit build jobs:
```bash
export MAX_JOBS=1
pip install -e . --verbose
```

### Forced Builds

If you have CUDA installed but no GPU, force build:
```bash
# Windows
set CRAYON_FORCE_CUDA=1
pip install -e . --force-reinstall

# Linux/Mac
export CRAYON_FORCE_CUDA=1
pip install -e . --force-reinstall
```

### Generic Wheel Build (for distribution)
```bash
export CRAYON_GENERIC_BUILD=1
python -m build
```

## Performance Verification

```python
import time
from crayon.core.vocabulary import CrayonVocab

# Test with different backends
for device in ["cpu", "cuda"]:
    try:
        vocab = CrayonVocab(device=device)
        vocab.load_profile("lite")
        
        start = time.time()
        tokens = vocab.tokenize("Hello world! " * 1000)
        elapsed = time.time() - start
        
        print(f"{device.upper()}: {elapsed:.6f}s for {len(tokens)} tokens")
    except Exception as e:
        print(f"{device.upper()}: {e}")
```

## Getting Help

- **Issues:** https://github.com/Electroiscoding/CRAYON/issues
- **Discussions:** https://github.com/Electroiscoding/CRAYON/discussions
- **Documentation:** https://github.com/Electroiscoding/CRAYON#readme

## Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `CRAYON_DEVICE` | Force device selection | `cuda`, `cpu`, `rocm` |
| `CRAYON_FORCE_CUDA` | Force CUDA build | `1` |
| `CRAYON_FORCE_ROCM` | Force ROCm build | `1` |
| `CRAYON_FORCE_CPU` | CPU-only build | `1` |
| `CRAYON_GENERIC_BUILD` | Build for all GPU archs | `1` |
| `CRAYON_PROFILE_DIR` | Custom profile directory | `/path/to/profiles` |
| `MAX_JOBS` | Limit build parallelism | `1` |
