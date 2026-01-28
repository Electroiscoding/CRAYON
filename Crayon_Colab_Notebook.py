"""
XERV CRAYON V4.2.9 - Production Omni-Backend Tokenizer
=======================================================
Copy this ENTIRE script into a Google Colab cell and run it.

IMPORTANT: Enable GPU runtime first:
Runtime -> Change runtime type -> GPU (T4/V100/A100)
"""

import subprocess
import sys
import os
import time

print("=" * 70)
print("XERV CRAYON V4.2.9 INSTALLATION")
print("=" * 70)

# ... (rest of the script is same until Verification)
# 6. Verification
print("\n[6/7] Verifying installation...")
# Reset module cache
for key in list(sys.modules.keys()):
    if "crayon" in key:
        del sys.modules[key]

try:
    import crayon
    print(f"      Success! Installed version: {crayon.get_version()}")
    backends = crayon.check_backends()
    print(f"      Backends: {backends}")
except ImportError as e:
    print(f"      FATAL: Could not import crayon: {e}")
    sys.exit(1)


# 7. Benchmarks
print("\n" + "=" * 70)
print("BENCHMARKS & TESTING")
print("=" * 70)

from crayon import CrayonVocab

vocab = CrayonVocab(device="auto")
vocab.load_profile("lite")
print(f"\nActive Device: {vocab.device.upper()}")

# USE CORRECT API
info = vocab.get_info()
print(f"Backend: {info['backend']}")

if vocab.device == "cpu" and backends.get("cuda"):
    print("NOTE: Running on CPU but CUDA is available. Use device='cuda' to force.")

# Throughput test
text = "The quick brown fox jumps over the lazy dog."
batch_sizes = [1000, 10000, 50000]
print("\nBatch Throughput:")
for bs in batch_sizes:
    batch = [text] * bs
    # Warmup
    vocab.tokenize(batch[:10]) 
    
    start = time.time()
    res = vocab.tokenize(batch)
    dur = time.time() - start
    
    toks = sum(len(x) for x in res)
    print(f"  {bs:>8,} docs: {bs/dur:>12,.0f} docs/sec | {toks/dur:>14,.0f} tokens/sec")

print("\nDone!")
