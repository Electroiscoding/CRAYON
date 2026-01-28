"""
XERV CRAYON V4.2.3 - Omni-Backend Tokenizer
=============================================
Copy this ENTIRE script into a Google Colab cell and run it.

IMPORTANT: Enable GPU runtime first:
Runtime -> Change runtime type -> GPU (T4/V100/A100)
"""

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 1: CLEAN INSTALL FROM SOURCE (FORCES CUDA COMPILATION)
# ═══════════════════════════════════════════════════════════════════════════════

import subprocess
import sys
import os

print("=" * 70)
print("XERV CRAYON INSTALLATION")
print("=" * 70)

print("\n[1/5] Detecting GPU hardware...")
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

print("\n[2/5] Checking CUDA compiler...")
nvcc_check = subprocess.run(["which", "nvcc"], capture_output=True, text=True)
if nvcc_check.returncode == 0:
    nvcc_path = nvcc_check.stdout.strip()
    print(f"      NVCC found: {nvcc_path}")
    nvcc_version = subprocess.run([nvcc_path, "--version"], capture_output=True, text=True)
    for line in nvcc_version.stdout.split("\n"):
        if "release" in line.lower():
            print(f"      {line.strip()}")
else:
    print("      NVCC not found - CUDA backend will not be available")

print("\n[3/5] Removing old installations...")
os.system("pip uninstall -y xerv-crayon crayon 2>/dev/null")
os.system("rm -rf /tmp/crayon ~/.cache/pip/wheels/*crayon* 2>/dev/null")

print("\n[4/5] Cloning latest source from GitHub...")
os.system("rm -rf /tmp/crayon")
clone_result = os.system("git clone --depth 1 https://github.com/Electroiscoding/CRAYON.git /tmp/crayon")
if clone_result != 0:
    print("      ERROR: Git clone failed!")
    sys.exit(1)

print("\n[5/5] Building and installing (with CUDA compilation)...")
print("      This may take 1-2 minutes on first run...")
print("-" * 70)

build_result = subprocess.run(
    [sys.executable, "-m", "pip", "install", "-v", "--no-build-isolation", "/tmp/crayon"],
    capture_output=False
)

print("-" * 70)

if build_result.returncode != 0:
    print("\nERROR: Installation failed!")
    sys.exit(1)

print("\n" + "=" * 70)
print("INSTALLATION COMPLETE")
print("=" * 70)

import crayon
print(f"\nCrayon Version: {crayon.get_version()}")
backends = crayon.check_backends()
print(f"Available Backends: {backends}")

if has_gpu and not backends.get("cuda"):
    print("\nWARNING: GPU detected but CUDA backend not available!")
    print("Check the build output above for CUDA compilation errors.")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 2: INITIALIZE AND TEST
# ═══════════════════════════════════════════════════════════════════════════════

from crayon import CrayonVocab

print("\n" + "=" * 70)
print("TOKENIZER TEST")
print("=" * 70)

vocab = CrayonVocab(device="auto")
vocab.load_profile("lite")

info = vocab.get_info()
print(f"\nActive Device: {info['device'].upper()}")
print(f"Backend: {info['backend']}")
print(f"Vocabulary Size: {vocab.vocab_size:,} tokens")

text = "Hello, world! Crayon is a high-performance tokenizer."
tokens = vocab.tokenize(text)
print(f"\nTest Input: {text}")
print(f"Tokens: {tokens}")
print(f"Token Count: {len(tokens)}")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 3: BENCHMARKS
# ═══════════════════════════════════════════════════════════════════════════════

import time

print("\n" + "=" * 70)
print("PERFORMANCE BENCHMARKS")
print("=" * 70)

base_text = "The quick brown fox jumps over the lazy dog."

print("\n--- Latency Test (Single String) ---")
iterations = 10000
for _ in range(100):
    vocab.tokenize(base_text)
start = time.perf_counter()
for _ in range(iterations):
    vocab.tokenize(base_text)
elapsed = time.perf_counter() - start
print(f"Latency: {(elapsed/iterations)*1e6:.2f} microseconds/call")
print(f"Throughput: {iterations/elapsed:,.0f} calls/second")

print("\n--- Batch Throughput Test ---")
print(f"{'Batch Size':>12} | {'Docs/sec':>14} | {'Tokens/sec':>16}")
print("-" * 50)

for batch_size in [100, 1000, 10000, 50000]:
    batch = [base_text] * batch_size
    vocab.tokenize(batch[:10])
    
    start = time.time()
    results = vocab.tokenize(batch)
    duration = time.time() - start
    
    total_tokens = sum(len(r) for r in results)
    docs_sec = batch_size / duration
    toks_sec = total_tokens / duration
    print(f"{batch_size:>12,} | {docs_sec:>14,.0f} | {toks_sec:>16,.0f}")

if vocab.device != "cpu":
    print(f"\n--- GPU Stress Test ({vocab.device.upper()}) ---")
    for batch_size in [100000, 500000]:
        batch = [base_text] * batch_size
        start = time.time()
        results = vocab.tokenize(batch)
        duration = time.time() - start
        total_tokens = sum(len(r) for r in results)
        print(f"{batch_size:>12,} docs in {duration:.3f}s = {batch_size/duration:,.0f} docs/sec, {total_tokens/duration:,.0f} tokens/sec")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 4: ROUND-TRIP VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════════

print("\n" + "=" * 70)
print("ENCODE/DECODE VERIFICATION")
print("=" * 70)

test_strings = [
    "Hello, Crayon!",
    "The quick brown fox jumps over the lazy dog.",
    "def forward(self, x): return torch.relu(x)",
]

all_passed = True
for s in test_strings:
    tokens = vocab.tokenize(s)
    decoded = vocab.decode(tokens)
    passed = s == decoded
    all_passed = all_passed and passed
    status = "PASS" if passed else "FAIL"
    print(f"[{status}] '{s[:40]}...' -> {len(tokens)} tokens")

print(f"\nAll tests passed: {all_passed}")

vocab.close()
print("\nDone!")
