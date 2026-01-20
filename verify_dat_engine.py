"""
XERV CRAYON V2.0 - Production Verification Script
Verifies the DAT engine with actual trained vocabularies.
"""
import sys
import os
import json

# Add paths
sys.path.insert(0, os.path.join(os.getcwd(), "build", "lib.win-amd64-cpython-313"))
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

import time
import tempfile
import mmap

from crayon.c_ext.dat_builder import DATBuilder
from crayon.c_ext import crayon_fast

print("=" * 70)
print("XERV CRAYON V2.0 - HYPER-PRODUCTION DAT ENGINE VERIFICATION")
print("=" * 70)

# Load the trained vocabulary (lite version for speed)
vocab_path = os.path.join(os.getcwd(), "trained_vocab_lite.json")
if not os.path.exists(vocab_path):
    # Fallback to full vocab
    vocab_path = os.path.join(os.getcwd(), "trained_vocab.json")

print(f"Loading vocabulary from: {vocab_path}")

with open(vocab_path, 'r', encoding='utf-8') as f:
    vocab_data = json.load(f)

# Handle both list and dict formats
if isinstance(vocab_data, list):
    vocab = vocab_data
elif isinstance(vocab_data, dict):
    vocab = [k for k, v in sorted(vocab_data.items(), key=lambda x: x[1])]
else:
    raise ValueError("Unknown vocab format")

print(f"Vocabulary Size: {len(vocab):,} tokens")

# Build DAT
builder = DATBuilder()
builder.build(vocab)

# Save to temp file  
dat_path = os.path.join(tempfile.gettempdir(), "trained_vocab.dat")
builder.save(dat_path)

print(f"DAT Nodes: {builder.size:,}")
print(f"DAT File Size: {os.path.getsize(dat_path)/1024:.1f} KB")

# Load via mmap (zero-copy)
fh = open(dat_path, 'rb')
mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
size = crayon_fast.load_dat(mm)
print(f"Loaded into C++ engine: {size:,} nodes")

# Build id_to_token for decoding
id_to_token = {i: t for i, t in enumerate(vocab)}

# Test tokenization
test_texts = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning and artificial intelligence are transforming industries.",
    "def hello_world():\n    print('Hello, World!')",
]

print("-" * 70)
print("TOKENIZATION SAMPLES:")
print("-" * 70)

for text in test_texts:
    tokens = crayon_fast.tokenize(text)
    # Decode first few tokens
    decoded = [id_to_token.get(t, f"[{t}]") for t in tokens[:10]]
    print(f"Input: \"{text[:50]}...\"" if len(text) > 50 else f"Input: \"{text}\"")
    print(f"Tokens ({len(tokens)}): {tokens[:10]}...")
    print(f"Decoded: {decoded}")
    print()

# Benchmark with substantial text
benchmark_text = " ".join(test_texts) * 5000
text_size_kb = len(benchmark_text) / 1024
text_size_mb = len(benchmark_text) / 1024 / 1024

print("=" * 70)
print(f"BENCHMARK: {text_size_mb:.2f} MB of text")
print("=" * 70)

# Warmup
_ = crayon_fast.tokenize(benchmark_text[:1000])

# Actual benchmark
start = time.perf_counter()
result = crayon_fast.tokenize(benchmark_text)
elapsed = time.perf_counter() - start

tokens_per_sec = len(result) / elapsed
mb_per_sec = text_size_mb / elapsed

print(f"Tokens generated: {len(result):,}")
print(f"Time: {elapsed*1000:.2f} ms")
print(f"Throughput: {tokens_per_sec:,.0f} tokens/sec")
print(f"Throughput: {mb_per_sec:.2f} MB/sec")
print("=" * 70)

if tokens_per_sec > 1_000_000:
    print("STATUS: ✅ HYPER-PRODUCTION READY (>1M tokens/sec)")
elif tokens_per_sec > 500_000:
    print("STATUS: ✅ PRODUCTION READY (>500K tokens/sec)")
else:
    print("STATUS: ⚠️ Performance below target")

# Cleanup
try:
    crayon_fast.load_dat(b'CRAY' + b'\x02\x00\x00\x00' + b'\x00\x00\x00\x00')
except:
    pass
mm.close()
fh.close()
os.unlink(dat_path)
