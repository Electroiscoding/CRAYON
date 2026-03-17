# CRAYON CUDA Testing Guide for Google Colab T4

## Quick Setup Commands

Run these cells in sequence in Google Colab (with T4 GPU runtime):

```bash
# Cell 1: Check GPU
!nvidia-smi
!nvcc --version
```

```bash
# Cell 2: Install PyTorch CUDA
!pip uninstall torch torchvision torchaudio -y
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")
```

```bash
# Cell 3: Install CRAYON with CUDA
!pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ xerv-crayon[cuda]

# Verify installation
!python -c "import crayon; print('CRAYON installed')"
```

```python
# Cell 4: Test CUDA functionality
import logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

from crayon.core.vocabulary import CrayonVocab

print("=== CRAYON CUDA Test ===")

# Auto-detection (should pick CUDA)
vocab = CrayonVocab(device="auto")
print(f"Device: {vocab.device}")

# Load profile
vocab.load_profile("lite")
print(f"Profile loaded: {len(vocab)} tokens")

# Test tokenization
text = "Hello, world! This is CUDA-accelerated tokenization."
tokens = vocab.tokenize(text)
print(f"Text: {text}")
print(f"Tokens: {tokens}")
print(f"Count: {len(tokens)}")
```

```python
# Cell 5: Performance benchmark
import time

def benchmark(vocab, text, runs=5):
    times = []
    for _ in range(runs):
        start = time.time()
        tokens = vocab.tokenize(text)
        times.append(time.time() - start)
    avg_time = sum(times) / len(times)
    return avg_time, len(tokens)

# Test texts
texts = [
    "Hello world",
    "Hello world! " * 10,
    "Hello world! " * 100,
    "Hello world! " * 1000,
]

# CPU comparison
vocab_cpu = CrayonVocab(device="cpu")
vocab_cpu.load_profile("lite")

print("=== Performance Comparison ===")
for i, text in enumerate(texts):
    print(f"\nTest {i+1}: {len(text)} chars")

    # CPU
    cpu_time, cpu_tokens = benchmark(vocab_cpu, text)
    print(f"  CPU:  {cpu_time:.6f}s ({cpu_tokens} tokens)")

    # CUDA
    cuda_time, cuda_tokens = benchmark(vocab, text)
    print(f"  CUDA: {cuda_time:.6f}s ({cuda_tokens} tokens)")

    # Speedup
    speedup = cpu_time / cuda_time if cuda_time > 0 else 0
    print(f"  Speedup: {speedup:.2f}x")
```

```python
# Cell 6: Batch processing test
batch_texts = [
    "def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2)",
    "class NeuralNetwork(nn.Module): def __init__(self): super().__init__()",
    "import torch; model = torch.nn.Sequential(torch.nn.Linear(10, 5), torch.nn.ReLU())",
] * 50  # Large batch

print(f"Batch size: {len(batch_texts)}")

# CUDA batch
start = time.time()
batch_tokens = vocab.tokenize(batch_texts)
cuda_batch_time = time.time() - start

# CPU batch
start = time.time()
batch_tokens_cpu = vocab_cpu.tokenize(batch_texts)
cpu_batch_time = time.time() - start

print(f"CPU batch:  {cpu_batch_time:.4f}s")
print(f"CUDA batch: {cuda_batch_time:.4f}s")
print(f"Speedup: {cpu_batch_time/cuda_batch_time:.2f}x")
```

## Expected Results on T4

- **Device Detection**: Should automatically select "cuda"
- **Hardware**: NVIDIA T4, ~16GB VRAM, Compute Capability 7.5
- **Performance**: 2-5x speedup on single texts, 5-10x on batches
- **Memory**: Efficient GPU utilization

## Troubleshooting

If CUDA doesn't work, run this diagnostic:

```python
# Get detailed error information
vocab = CrayonVocab(device="cpu")  # Initialize first
print(vocab._get_cuda_import_error())
```

Common fixes:
1. **PyTorch not CUDA**: Reinstall with `cu121` wheels
2. **CUDA_HOME**: Colab usually has this set correctly
3. **GPU runtime**: Ensure "GPU" is selected in runtime settings

## Colab-Specific Notes

- **Free T4 GPU**: Limited to ~12 hours, may disconnect
- **Memory**: ~16GB GPU RAM, ~25GB system RAM
- **CUDA**: Pre-installed CUDA 12.2, but we use 12.1 for compatibility
- **PyTorch**: Must be CUDA-enabled version

## Alternative: Use Development Version

```bash
# Install directly from GitHub
!pip install git+https://github.com/Electroiscoding/CRAYON.git

# Force CUDA build if needed
!CRAYON_FORCE_CUDA=1 pip install git+https://github.com/Electroiscoding/CRAYON.git
```

This guide tests the CRAYON improvements made to fix CUDA extension issues and provide better error messaging.
