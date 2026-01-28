"""
XERV CRAYON V4.2.2 - Omni-Backend Tokenizer
============================================
Copy this entire file into Google Colab and run all cells.
Works on CPU, NVIDIA GPU (T4/V100/A100), and AMD GPU.

IMPORTANT: Enable GPU runtime for best performance:
Runtime -> Change runtime type -> GPU
"""

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 1: INSTALL CRAYON (ALWAYS BUILDS FROM SOURCE FOR GPU SUPPORT)
# ═══════════════════════════════════════════════════════════════════════════════

import subprocess
import sys
import os

print("Detecting hardware...")
try:
    result = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                           capture_output=True, text=True, timeout=10)
    if result.returncode == 0:
        gpu_name = result.stdout.strip()
        print(f"GPU Found: {gpu_name}")
    else:
        gpu_name = None
        print("No NVIDIA GPU detected")
except:
    gpu_name = None
    print("No NVIDIA GPU detected")

print("Installing Crayon from source (with GPU compilation if available)...")
os.system("rm -rf /tmp/crayon 2>/dev/null")
os.system("git clone --depth 1 https://github.com/Electroiscoding/CRAYON.git /tmp/crayon")
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-build-isolation", "/tmp/crayon"])

import crayon
print(f"Crayon v{crayon.get_version()} installed")
backends = crayon.check_backends()
print(f"Available backends: {backends}")

if gpu_name and not backends.get("cuda"):
    print("WARNING: GPU detected but CUDA backend not available.")
    print("This may be due to compilation issues. Check build logs above.")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 2: INITIALIZE TOKENIZER
# ═══════════════════════════════════════════════════════════════════════════════

from crayon import CrayonVocab

vocab = CrayonVocab(device="auto")
vocab.load_profile("lite")

info = vocab.get_info()
print(f"Active Device: {info['device'].upper()}")
print(f"Backend: {info['backend']}")
print(f"Vocabulary Size: {vocab.vocab_size:,} tokens")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 3: BASIC TOKENIZATION
# ═══════════════════════════════════════════════════════════════════════════════

text = "Hello, world! Crayon is a high-performance tokenizer."
tokens = vocab.tokenize(text)

print(f"Input: {text}")
print(f"Tokens: {tokens}")
print(f"Token Count: {len(tokens)}")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 4: LATENCY BENCHMARK (SINGLE STRING)
# ═══════════════════════════════════════════════════════════════════════════════

import time

text = "The quick brown fox jumps over the lazy dog."
iterations = 10000

for _ in range(100):
    vocab.tokenize(text)

start = time.perf_counter()
for _ in range(iterations):
    vocab.tokenize(text)
elapsed = time.perf_counter() - start

latency_us = (elapsed / iterations) * 1_000_000
print(f"Single-String Latency: {latency_us:.2f} microseconds")
print(f"Calls per Second: {iterations / elapsed:,.0f}")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 5: BATCH THROUGHPUT BENCHMARK
# ═══════════════════════════════════════════════════════════════════════════════

base_text = "The quick brown fox jumps over the lazy dog."

print("\nBatch Throughput Results:")
print("-" * 60)

for batch_size in [100, 1000, 10000, 50000]:
    batch = [base_text] * batch_size
    
    vocab.tokenize(batch[:10])
    
    start = time.time()
    results = vocab.tokenize(batch)
    duration = time.time() - start
    
    total_tokens = sum(len(r) for r in results)
    docs_per_sec = batch_size / duration
    tokens_per_sec = total_tokens / duration
    
    print(f"Batch {batch_size:>6}: {docs_per_sec:>12,.0f} docs/sec | {tokens_per_sec:>14,.0f} tokens/sec")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 6: GPU STRESS TEST (IF AVAILABLE)
# ═══════════════════════════════════════════════════════════════════════════════

if vocab.device != "cpu":
    print(f"\nGPU Stress Test ({vocab.device.upper()}):")
    print("-" * 60)
    
    for batch_size in [10000, 50000, 100000]:
        batch = [base_text] * batch_size
        
        start = time.time()
        results = vocab.tokenize(batch)
        duration = time.time() - start
        
        total_tokens = sum(len(r) for r in results)
        print(f"Batch {batch_size:>6}: {batch_size/duration:>12,.0f} docs/sec | {total_tokens/duration:>14,.0f} tokens/sec in {duration:.3f}s")
else:
    print("\nGPU stress test skipped (running on CPU)")
    print("To enable GPU: Runtime -> Change runtime type -> GPU")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 7: ENCODE/DECODE ROUND-TRIP
# ═══════════════════════════════════════════════════════════════════════════════

original = "Hello, Crayon!"
tokens = vocab.tokenize(original)
decoded = vocab.decode(tokens)

print(f"\nRound-Trip Test:")
print(f"  Original: {original}")
print(f"  Tokens:   {tokens}")
print(f"  Decoded:  {decoded}")
print(f"  Match:    {original == decoded}")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 8: CLEANUP
# ═══════════════════════════════════════════════════════════════════════════════

vocab.close()
print("\nDone!")
