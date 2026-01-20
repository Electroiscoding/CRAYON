"""
XERV CRAYON V2.0 - Quick Benchmark Suite
Benchmarks the DAT Engine with smaller vocabularies for fast results.
"""
import sys
import os
import json
import time
import tempfile
import mmap
import logging

# Suppress verbose logging
logging.getLogger().setLevel(logging.WARNING)

# Add paths
sys.path.insert(0, os.path.join(os.getcwd(), "build", "lib.win-amd64-cpython-313"))
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

from crayon.c_ext.dat_builder import DATBuilder
from crayon.c_ext import crayon_fast

def load_vocab_from_json(path: str) -> list:
    """Load vocabulary from JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return data
    elif isinstance(data, dict):
        return [k for k, v in sorted(data.items(), key=lambda x: x[1])]
    else:
        raise ValueError(f"Unknown vocab format in {path}")

def benchmark_vocab(name: str, vocab: list, test_text: str, iterations: int = 5) -> dict:
    """Benchmark a vocabulary with the DAT engine."""
    # Suppress builder logging
    import logging
    logging.getLogger().setLevel(logging.CRITICAL)
    
    # Build DAT
    builder = DATBuilder()
    build_start = time.perf_counter()
    builder.build(vocab)
    build_time = time.perf_counter() - build_start
    
    # Save to temp file
    dat_path = os.path.join(tempfile.gettempdir(), f"bench_{name}.dat")
    builder.save(dat_path)
    dat_size = os.path.getsize(dat_path)
    
    # Load via mmap
    fh = open(dat_path, 'rb')
    mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
    
    load_start = time.perf_counter()
    size = crayon_fast.load_dat(mm)
    load_time = time.perf_counter() - load_start
    
    # Warmup
    _ = crayon_fast.tokenize(test_text[:1000])
    
    # Benchmark
    text_bytes = len(test_text.encode('utf-8'))
    total_tokens = 0
    total_time = 0.0
    
    for _ in range(iterations):
        start = time.perf_counter()
        tokens = crayon_fast.tokenize(test_text)
        elapsed = time.perf_counter() - start
        total_tokens += len(tokens)
        total_time += elapsed
    
    avg_time = total_time / iterations
    avg_tokens = total_tokens / iterations
    
    tokens_per_sec = avg_tokens / avg_time
    mb_per_sec = (text_bytes / 1024 / 1024) / avg_time
    
    # Cleanup
    try:
        crayon_fast.load_dat(b'CRAY' + b'\x02\x00\x00\x00' + b'\x00\x00\x00\x00')
    except:
        pass
    mm.close()
    fh.close()
    os.unlink(dat_path)
    
    return {
        'name': name,
        'vocab_size': len(vocab),
        'dat_nodes': size,
        'dat_size_kb': dat_size / 1024,
        'build_time_ms': build_time * 1000,
        'load_time_ms': load_time * 1000,
        'tokens_generated': int(avg_tokens),
        'time_ms': avg_time * 1000,
        'tokens_per_sec': tokens_per_sec,
        'mb_per_sec': mb_per_sec,
    }

def main():
    print("=" * 80)
    print("XERV CRAYON V2.0 - QUICK BENCHMARK SUITE")
    print("=" * 80)
    print()
    
    # Smaller vocabs first (quick to compile)
    vocab_files = [
        ("science", "trained_vocab_science.json"),
        ("code", "trained_vocab_code.json"),
        ("multilingual", "trained_vocab_multilingual.json"),
        ("arts_commerce", "trained_vocab_arts_commerce.json"),
        ("lite_5k", "trained_vocab_lite.json", 5000),  # First 5k tokens only
    ]
    
    # Test text
    benchmark_text = """The quick brown fox jumps over the lazy dog. Machine learning and artificial 
intelligence are transforming industries. def fibonacci(n): return n if n <= 1 else fibonacci(n-1) + fibonacci(n-2).
The Schrödinger equation describes quantum behavior. class DataProcessor: pass. """ * 5000
    
    text_size_mb = len(benchmark_text) / 1024 / 1024
    
    print(f"Benchmark Text Size: {text_size_mb:.2f} MB")
    print(f"Iterations per vocab: 5")
    print("-" * 80)
    print()
    
    results = []
    
    for entry in vocab_files:
        if len(entry) == 3:
            name, filename, limit = entry
        else:
            name, filename = entry
            limit = None
            
        filepath = os.path.join(os.getcwd(), filename)
        if not os.path.exists(filepath):
            print(f"[SKIP] {name}: File not found")
            continue
        
        print(f"[BENCH] {name}...", end=" ", flush=True)
        try:
            vocab = load_vocab_from_json(filepath)
            if limit:
                vocab = vocab[:limit]
            
            result = benchmark_vocab(name, vocab, benchmark_text)
            results.append(result)
            
            print(f"✓ {result['vocab_size']:,} tokens | {result['tokens_per_sec']:,.0f} tok/s | {result['mb_per_sec']:.2f} MB/s")
        except Exception as e:
            print(f"✗ ERROR: {e}")
    
    # Summary table
    print()
    print("=" * 80)
    print("BENCHMARK RESULTS SUMMARY")
    print("=" * 80)
    print()
    print(f"{'Profile':<20} | {'Vocab':>8} | {'Tokens/sec':>15} | {'MB/sec':>8} | {'Build':>10}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['name']:<20} | {r['vocab_size']:>8,} | {r['tokens_per_sec']:>15,.0f} | {r['mb_per_sec']:>8.2f} | {r['build_time_ms']:>9.0f}ms")
    
    print("-" * 80)
    print()
    
    # Markdown table for README
    print("=" * 80)
    print("MARKDOWN TABLE FOR README.md")
    print("=" * 80)
    print()
    print("| Profile | Vocab Size | Tokens/sec | MB/sec | DAT Size | Status |")
    print("| :--- | ---: | ---: | ---: | ---: | :---: |")
    
    for r in results:
        status = "✅" if r['tokens_per_sec'] > 500000 else "⚠️"
        print(f"| **`{r['name']}`** | {r['vocab_size']:,} | **{r['tokens_per_sec']:,.0f}** | {r['mb_per_sec']:.2f} | {r['dat_size_kb']:.0f} KB | {status} |")
    
    print()
    print("=" * 80)

if __name__ == "__main__":
    main()
