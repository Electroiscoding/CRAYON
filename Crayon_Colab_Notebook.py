"""
XERV CRAYON V4.2.3 - Omni-Backend Tokenizer
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
print("XERV CRAYON INSTALLATION V4.2.3")
print("=" * 70)

# Step 1: GPU Detection
print("\n[1/6] Detecting GPU hardware...")
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
print("\n[2/6] Checking CUDA compiler...")
nvcc_check = subprocess.run(["which", "nvcc"], capture_output=True, text=True)
if nvcc_check.returncode == 0:
    nvcc_path = nvcc_check.stdout.strip()
    print(f"      NVCC: {nvcc_path}")
    nvcc_v = subprocess.run([nvcc_path, "--version"], capture_output=True, text=True)
    for line in nvcc_v.stdout.split("\n"):
        if "release" in line.lower():
            print(f"      {line.strip()}")
    has_nvcc = True
else:
    print("      NVCC not found")
    has_nvcc = False

# Step 3: Clean ALL Caches
print("\n[3/6] Cleaning ALL caches...")
os.system("pip uninstall -y xerv-crayon crayon 2>/dev/null")
os.system("pip cache purge 2>/dev/null")
os.system("rm -rf /tmp/crayon /tmp/crayon_build ~/.cache/pip 2>/dev/null")
print("      Done")

# Step 4: Fresh Clone with timestamp to avoid caching
print("\n[4/6] Cloning from GitHub (fresh)...")
timestamp = int(time.time())
clone_dir = f"/tmp/crayon_{timestamp}"
os.system(f"git clone --depth 1 https://github.com/Electroiscoding/CRAYON.git {clone_dir}")

# Verify version in cloned repo
version_check = subprocess.run(["grep", "__version__", f"{clone_dir}/src/crayon/__init__.py"],
                               capture_output=True, text=True)
print(f"      Cloned version: {version_check.stdout.strip()}")

# Step 5: Install with verbose output and no cache
print("\n[5/6] Building and installing...")
print("-" * 70)

result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-v", "--no-cache-dir", "--no-build-isolation", clone_dir],
    env={**os.environ, "CUDA_HOME": "/usr/local/cuda"}
)

print("-" * 70)

# Step 6: Verify Installation
print("\n[6/6] Verifying installation...")

# Force reimport
if "crayon" in sys.modules:
    del sys.modules["crayon"]
for key in list(sys.modules.keys()):
    if key.startswith("crayon"):
        del sys.modules[key]

import crayon
print(f"\n      Crayon Version: {crayon.get_version()}")
backends = crayon.check_backends()
print(f"      Backends: {backends}")

if backends.get("cuda"):
    print("      CUDA backend: READY")
elif has_gpu and has_nvcc:
    print("\n      WARNING: GPU + NVCC detected but CUDA backend not available!")
    print("      Check the build output above for errors.")

print("\n" + "=" * 70)
print("INITIALIZATION")
print("=" * 70)

from crayon import CrayonVocab

vocab = CrayonVocab(device="auto")
vocab.load_profile("lite")

info = vocab.get_info()
print(f"\nActive Device: {info['device'].upper()}")
print(f"Backend: {info['backend']}")
print(f"Vocabulary: {vocab.vocab_size:,} tokens")

# Quick test
text = "Hello, Crayon tokenizer!"
tokens = vocab.tokenize(text)
print(f"\nTest: '{text}' -> {len(tokens)} tokens")

print("\n" + "=" * 70)
print("BENCHMARKS")
print("=" * 70)

import time

base_text = "The quick brown fox jumps over the lazy dog."

print("\n--- Batch Throughput ---")
for batch_size in [1000, 10000, 50000]:
    batch = [base_text] * batch_size
    vocab.tokenize(batch[:10])
    start = time.time()
    results = vocab.tokenize(batch)
    duration = time.time() - start
    total_tokens = sum(len(r) for r in results)
    print(f"{batch_size:>8}: {batch_size/duration:>12,.0f} docs/sec | {total_tokens/duration:>14,.0f} tokens/sec")

if vocab.device != "cpu":
    print(f"\n--- GPU Stress Test ({vocab.device.upper()}) ---")
    for batch_size in [100000, 500000]:
        batch = [base_text] * batch_size
        start = time.time()
        results = vocab.tokenize(batch)
        duration = time.time() - start
        total_tokens = sum(len(r) for r in results)
        print(f"{batch_size:>8}: {batch_size/duration:>12,.0f} docs/sec in {duration:.3f}s")

vocab.close()
print("\nDone!")
