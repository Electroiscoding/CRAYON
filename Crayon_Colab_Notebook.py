"""
XERV CRAYON V4.2.0 - Omni-Backend Tokenizer
============================================
Copy this entire file into Google Colab and run all cells.
Works on CPU, NVIDIA GPU (T4/V100/A100), and AMD GPU.
"""

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 1: INSTALL CRAYON (WITH AUTOMATIC GPU DETECTION)
# ═══════════════════════════════════════════════════════════════════════════════

import subprocess
import sys
import os

def detect_gpu():
    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, timeout=10)
        return result.returncode == 0
    except:
        return False

has_gpu = detect_gpu()
print(f"GPU Detected: {has_gpu}")

if has_gpu:
    print("Building from source with CUDA support...")
    os.system("rm -rf /tmp/crayon 2>/dev/null")
    result = os.system("git clone --depth 1 https://github.com/Electroiscoding/CRAYON.git /tmp/crayon 2>/dev/null")
    if result == 0:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "--no-build-isolation", "/tmp/crayon"])
    else:
        print("Git clone failed, installing from TestPyPI (CPU only)")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", 
            "-i", "https://test.pypi.org/simple/", 
            "--extra-index-url", "https://pypi.org/simple/", "xerv-crayon"])
else:
    print("Installing pre-built CPU version...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", 
        "-i", "https://test.pypi.org/simple/", 
        "--extra-index-url", "https://pypi.org/simple/", "xerv-crayon"])

import crayon
print(f"Crayon v{crayon.get_version()} installed")
print(f"Available backends: {crayon.check_backends()}")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 2: INITIALIZE TOKENIZER (AUTO-DETECTS GPU)
# ═══════════════════════════════════════════════════════════════════════════════

from crayon import CrayonVocab

vocab = CrayonVocab(device="auto")
vocab.load_profile("lite")

info = vocab.get_info()
print(f"Device: {info['device'].upper()}")
print(f"Backend: {info['backend']}")
print(f"Vocab Size: {vocab.vocab_size:,} tokens")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 3: BASIC TOKENIZATION
# ═══════════════════════════════════════════════════════════════════════════════

text = "Hello, world! Crayon is a high-performance tokenizer."
tokens = vocab.tokenize(text)

print(f"Input: {text}")
print(f"Tokens: {tokens}")
print(f"Count: {len(tokens)}")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 4: BATCH TOKENIZATION
# ═══════════════════════════════════════════════════════════════════════════════

batch = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning powers modern AI systems.",
    "def forward(self, x): return torch.relu(x)",
]

batch_tokens = vocab.tokenize(batch)

for i, (text, toks) in enumerate(zip(batch, batch_tokens)):
    print(f"[{i+1}] {text[:40]}... -> {len(toks)} tokens")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 5: LATENCY BENCHMARK
# ═══════════════════════════════════════════════════════════════════════════════

import time

text = "Crayon optimizes tokenization at the silicon level with AVX2 SIMD."
iterations = 10000

for _ in range(100):
    _ = vocab.tokenize(text)

start = time.perf_counter()
for _ in range(iterations):
    _ = vocab.tokenize(text)
elapsed = time.perf_counter() - start

latency_us = (elapsed / iterations) * 1_000_000
print(f"Latency: {latency_us:.2f} us/call")
print(f"Throughput: {iterations / elapsed:,.0f} calls/sec")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 6: BATCH THROUGHPUT BENCHMARK
# ═══════════════════════════════════════════════════════════════════════════════

base_text = "The quick brown fox jumps over the lazy dog."

for batch_size in [100, 1000, 10000]:
    batch = [base_text] * batch_size
    
    _ = vocab.tokenize(batch[:10])
    
    start = time.time()
    results = vocab.tokenize(batch)
    duration = time.time() - start
    
    total_tokens = sum(len(r) for r in results)
    
    print(f"Batch {batch_size:>5}: {batch_size/duration:>10,.0f} docs/sec | {total_tokens/duration:>12,.0f} tokens/sec")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 7: GPU STRESS TEST (100K DOCUMENTS)
# ═══════════════════════════════════════════════════════════════════════════════

if vocab.device != "cpu":
    batch_size = 100_000
    batch = ["The quick brown fox jumps over the lazy dog."] * batch_size
    
    print(f"Processing {batch_size:,} documents on {vocab.device.upper()}...")
    
    start = time.time()
    results = vocab.tokenize(batch)
    duration = time.time() - start
    
    total_tokens = sum(len(r) for r in results)
    
    print(f"Duration: {duration:.4f}s")
    print(f"Throughput: {batch_size/duration:,.0f} docs/sec")
    print(f"Token Rate: {total_tokens/duration:,.0f} tokens/sec")
else:
    print("Skipping GPU stress test (running on CPU)")
    print("Enable GPU: Runtime -> Change runtime type -> GPU")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 8: PROFILE SWITCHING
# ═══════════════════════════════════════════════════════════════════════════════

code = "def forward(self, x): return torch.matmul(x, w)"

tokens_lite = vocab.tokenize(code)
print(f"[LITE] {len(tokens_lite)} tokens")

try:
    with vocab.using_profile("code"):
        tokens_code = vocab.tokenize(code)
        print(f"[CODE] {len(tokens_code)} tokens")
except FileNotFoundError:
    print("[CODE] Profile not available")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 9: ENCODE/DECODE ROUND-TRIP
# ═══════════════════════════════════════════════════════════════════════════════

original = "Hello, Crayon!"
tokens = vocab.tokenize(original)
decoded = vocab.decode(tokens)

print(f"Original: {original}")
print(f"Tokens: {tokens}")
print(f"Decoded: {decoded}")
print(f"Match: {original == decoded}")

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 10: CLEANUP
# ═══════════════════════════════════════════════════════════════════════════════

vocab.close()
print("Done!")
