"""
XERV CRAYON V4.2.4 - Production Omni-Backend Tokenizer
=======================================================
Copy this ENTIRE script into a Google Colab cell and run it.

IMPORTANT: Enable GPU runtime first:
Runtime -> Change runtime type -> GPU (T4/V100/A100)

This version uses PyTorch's CUDAExtension for reliable CUDA compilation.
"""

import subprocess
import sys
import os
import time

print("=" * 70)
print("XERV CRAYON V4.2.4 INSTALLATION")
print("=" * 70)

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

print("\n[2/7] Checking CUDA compiler...")
nvcc_check = subprocess.run(["which", "nvcc"], capture_output=True, text=True)
if nvcc_check.returncode == 0:
    print(f"      NVCC: {nvcc_check.stdout.strip()}")
else:
    print("      NVCC not found")

print("\n[3/7] Ensuring PyTorch with CUDA...")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "torch"], capture_output=True)

import torch
print(f"      PyTorch: {torch.__version__}")
print(f"      CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"      CUDA version: {torch.version.cuda}")

print("\n[4/7] Cleaning caches...")
os.system("pip uninstall -y xerv-crayon crayon 2>/dev/null")
os.system("pip cache purge 2>/dev/null")
os.system("rm -rf /tmp/crayon* ~/.cache/pip 2>/dev/null")
print("      Done")

print("\n[5/7] Cloning from GitHub...")
timestamp = int(time.time())
clone_dir = f"/tmp/crayon_{timestamp}"
os.system(f"git clone --depth 1 https://github.com/Electroiscoding/CRAYON.git {clone_dir}")

version_check = subprocess.run(["grep", "-m1", "__version__", f"{clone_dir}/src/crayon/__init__.py"],
                               capture_output=True, text=True)
print(f"      Source version: {version_check.stdout.strip()}")

print("\n[6/7] Building with PyTorch CUDAExtension...")
print("-" * 70)

build_env = os.environ.copy()
build_env["CUDA_HOME"] = "/usr/local/cuda"

result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-v", "--no-cache-dir", "--no-build-isolation", clone_dir],
    env=build_env
)

print("-" * 70)

print("\n[7/7] Verifying installation...")

for key in list(sys.modules.keys()):
    if "crayon" in key:
        del sys.modules[key]

import crayon
print(f"\n      Installed version: {crayon.get_version()}")
backends = crayon.check_backends()
print(f"      Backends: {backends}")

if backends.get("cuda"):
    print("      CUDA backend: READY", "")
else:
    if has_gpu:
        print("      CUDA backend: NOT AVAILABLE (check build logs above)")
    else:
        print("      CUDA backend: NOT AVAILABLE (no GPU)")

print("\n" + "=" * 70)
print("TOKENIZER INITIALIZATION")
print("=" * 70)

from crayon import CrayonVocab

vocab = CrayonVocab(device="auto")
vocab.load_profile("lite")

info = vocab.get_info()
print(f"\nActive Device: {info['device'].upper()}")
print(f"Backend: {info['backend']}")
print(f"Vocabulary: {vocab.vocab_size:,} tokens")

text = "Hello, Crayon tokenizer!"
tokens = vocab.tokenize(text)
print(f"\nQuick Test: '{text}'")
print(f"Tokens: {tokens}")
print(f"Count: {len(tokens)}")

print("\n" + "=" * 70)
print("PERFORMANCE BENCHMARKS")
print("=" * 70)

base_text = "The quick brown fox jumps over the lazy dog."

print("\n--- Latency (single string) ---")
iterations = 10000
for _ in range(100):
    vocab.tokenize(base_text)
start = time.perf_counter()
for _ in range(iterations):
    vocab.tokenize(base_text)
elapsed = time.perf_counter() - start
print(f"Latency: {(elapsed/iterations)*1e6:.2f} us/call")
print(f"Calls/sec: {iterations/elapsed:,.0f}")

print("\n--- Batch Throughput ---")
print(f"{'Batch':>10} | {'Docs/sec':>14} | {'Tokens/sec':>16}")
print("-" * 48)

for batch_size in [1000, 10000, 50000]:
    batch = [base_text] * batch_size
    vocab.tokenize(batch[:10])
    start = time.time()
    results = vocab.tokenize(batch)
    duration = time.time() - start
    total_tokens = sum(len(r) for r in results)
    print(f"{batch_size:>10,} | {batch_size/duration:>14,.0f} | {total_tokens/duration:>16,.0f}")

if vocab.device != "cpu":
    print(f"\n--- GPU Stress Test ({vocab.device.upper()}) ---")
    for batch_size in [100000, 500000]:
        batch = [base_text] * batch_size
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        start = time.time()
        results = vocab.tokenize(batch)
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        duration = time.time() - start
        total_tokens = sum(len(r) for r in results)
        print(f"{batch_size:>10,}: {batch_size/duration:>12,.0f} docs/sec | {duration:.3f}s")

print("\n" + "=" * 70)
print("ENCODE/DECODE VERIFICATION")
print("=" * 70)

test_cases = [
    "Hello, world!",
    "The quick brown fox.",
    "def forward(self, x): return x",
]

all_passed = True
for text in test_cases:
    tokens = vocab.tokenize(text)
    decoded = vocab.decode(tokens)
    passed = text == decoded
    all_passed = all_passed and passed
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] '{text}' -> {len(tokens)} tokens")

print(f"\nAll tests: {'PASSED' if all_passed else 'FAILED'}")

vocab.close()
print("\nDone!")
