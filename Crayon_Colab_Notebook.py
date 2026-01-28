"""
XERV CRAYON V4.2.4 - Omni-Backend Tokenizer
=============================================
Copy this ENTIRE script into a Google Colab cell and run it.

IMPORTANT: Enable GPU runtime first:
Runtime -> Change runtime type -> GPU (T4/V100/A100)
"""

import subprocess
import sys
import os
import time

print("=" * 70)
print("XERV CRAYON INSTALLATION V4.2.4")
print("=" * 70)

# Step 1: GPU Detection
print("\n[1/7] Detecting GPU hardware...")
try:
    result = subprocess.run(["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10)
    if result.returncode == 0:
        gpu_info = result.stdout.strip()
        print(f"      GPU: {gpu_info}")
        has_gpu = True
    else:
        print("      No NVIDIA GPU detected")
        has_gpu = False
except:
    print("      No NVIDIA GPU detected")
    has_gpu = False

# Step 2: NVCC Detection
print("\n[2/7] Checking CUDA compiler...")
nvcc_check = subprocess.run(["which", "nvcc"], capture_output=True, text=True)
if nvcc_check.returncode == 0:
    nvcc_path = nvcc_check.stdout.strip()
    print(f"      NVCC: {nvcc_path}")
    has_nvcc = True
else:
    print("      NVCC not found")
    has_nvcc = False

# Step 3: Ensure PyTorch is installed (required for CUDAExtension)
print("\n[3/7] Checking PyTorch...")
try:
    import torch
    print(f"      PyTorch {torch.__version__}")
    print(f"      CUDA available: {torch.cuda.is_available()}")
except ImportError:
    print("      Installing PyTorch...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "torch"])
    import torch
    print(f"      PyTorch {torch.__version__} installed")

# Step 4: Clean ALL Caches
print("\n[4/7] Cleaning ALL caches...")
os.system("pip uninstall -y xerv-crayon crayon 2>/dev/null")
os.system("pip cache purge 2>/dev/null")
os.system("rm -rf /tmp/crayon* ~/.cache/pip 2>/dev/null")
print("      Done")

# Step 5: Fresh Clone
print("\n[5/7] Cloning from GitHub...")
timestamp = int(time.time())
clone_dir = f"/tmp/crayon_{timestamp}"
os.system(f"git clone --depth 1 https://github.com/Electroiscoding/CRAYON.git {clone_dir}")

version_check = subprocess.run(["grep", "-m1", "__version__", f"{clone_dir}/src/crayon/__init__.py"],
                               capture_output=True, text=True)
print(f"      {version_check.stdout.strip()}")

# Step 6: Build and Install
print("\n[6/7] Building with CUDA support (this takes ~2 min)...")
print("-" * 70)

env = os.environ.copy()
env["CUDA_HOME"] = "/usr/local/cuda"

result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-v", "--no-cache-dir", "--no-build-isolation", clone_dir],
    env=env
)

print("-" * 70)

# Step 7: Verify
print("\n[7/7] Verifying installation...")

for key in list(sys.modules.keys()):
    if "crayon" in key:
        del sys.modules[key]

import crayon
print(f"\n      Version: {crayon.get_version()}")
backends = crayon.check_backends()
print(f"      Backends: {backends}")

if backends.get("cuda"):
    print("      CUDA: READY", "\u2705")
elif has_gpu and has_nvcc:
    print("      WARNING: GPU detected but CUDA not compiled!")
    print("      Check build output above for nvcc errors")

print("\n" + "=" * 70)
print("TOKENIZER TEST")
print("=" * 70)

from crayon import CrayonVocab

vocab = CrayonVocab(device="auto")
vocab.load_profile("lite")

info = vocab.get_info()
print(f"\nDevice: {info['device'].upper()}")
print(f"Backend: {info['backend']}")
print(f"Vocabulary: {vocab.vocab_size:,} tokens")

text = "Hello, Crayon!"
tokens = vocab.tokenize(text)
print(f"\nTest: '{text}' -> {tokens}")

print("\n" + "=" * 70)
print("BENCHMARKS")  
print("=" * 70)

base_text = "The quick brown fox jumps over the lazy dog."

print("\n--- Throughput ---")
for batch_size in [1000, 10000, 50000]:
    batch = [base_text] * batch_size
    vocab.tokenize(batch[:10])
    start = time.time()
    results = vocab.tokenize(batch)
    duration = time.time() - start
    total_tokens = sum(len(r) for r in results)
    print(f"{batch_size:>8}: {batch_size/duration:>12,.0f} docs/sec | {total_tokens/duration:>14,.0f} tokens/sec")

if vocab.device != "cpu":
    print(f"\n--- GPU Stress Test ---")
    for batch_size in [100000, 500000]:
        batch = [base_text] * batch_size
        start = time.time()
        results = vocab.tokenize(batch)
        duration = time.time() - start
        print(f"{batch_size:>8}: {batch_size/duration:>12,.0f} docs/sec in {duration:.3f}s")

vocab.close()
print("\n" + "=" * 70)
print("DONE!")
print("=" * 70)
