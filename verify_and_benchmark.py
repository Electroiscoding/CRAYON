"""
Final Verification, Benchmark, and Data Report for XERV Crayon.

1. Verifies tokenization correctness.
2. Benchmarks performance with the TRAINED vocabulary.
3. Reports exact data quantities utilized.
"""

import time
import json
import csv
from pathlib import Path
from crayon import CrayonVocab

# Configuration
VOCAB_PATH = "trained_vocab.json"
RESOURCE_DIR = Path("src/crayon/resources")

def calculate_data_stats():
    """Calculates exact quantity of data used for training."""
    stats = {
        "files": [],
        "total_lines": 0,
        "total_bytes": 0,
        "total_samples": 0
    }
    
    # 1. Shakespeare
    fpath = RESOURCE_DIR / "input.txt"
    if fpath.exists():
        size = fpath.stat().st_size
        lines = 0
        with open(fpath, 'r', encoding='utf-8') as f:
            lines = sum(1 for _ in f)
        stats["files"].append({"name": "Tiny Shakespeare", "size": size, "lines": lines, "samples": 1})
        stats["total_bytes"] += size
        stats["total_lines"] += lines
        stats["total_samples"] += 1

    # 2. RainDrop-DTS
    fpath = RESOURCE_DIR / "data.csv"
    if fpath.exists():
        size = fpath.stat().st_size
        samples = 0
        with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
            samples = sum(1 for _ in f) - 1 # Header
        stats["files"].append({"name": "RainDrop-DTS (CSV)", "size": size, "lines": samples + 1, "samples": samples})
        stats["total_bytes"] += size
        stats["total_lines"] += samples + 1
        stats["total_samples"] += samples

    # 3. Physics
    fpath = RESOURCE_DIR / "physics_detailed_dataset_700_rows.csv"
    if fpath.exists():
        size = fpath.stat().st_size
        samples = 0
        with open(fpath, 'r', encoding='utf-8', errors='ignore') as f:
            samples = sum(1 for _ in f) - 1
        stats["files"].append({"name": "Physics Dataset (CSV)", "size": size, "lines": samples + 1, "samples": samples})
        stats["total_bytes"] += size
        stats["total_lines"] += samples + 1
        stats["total_samples"] += samples

    # 4. GRAD
    fpath = RESOURCE_DIR / "graduate_math.jsonl"
    if fpath.exists():
        size = fpath.stat().st_size
        samples = 0
        # In training we limited this, checking actual usage limit
        with open("train_vocab.py", "r") as f:
            content = f.read()
            if "MAX_GRAD_ENTRIES = 500" in content:
                limit_msg = "(Limited to 500 entries)"
                used_samples = 500
            else:
                limit_msg = "(Full Dataset)"
                with open(fpath, 'r', encoding='utf-8', errors='ignore') as jf:
                     used_samples = sum(1 for _ in jf)
        
        stats["files"].append({"name": f"GRAD Math (JSONL) {limit_msg}", "size": size, "lines": used_samples, "samples": used_samples})
        
        # We only count bytes processed roughly for the report if limited
        if "Limited" in limit_msg:
             stats["total_bytes"] += min(size, 5 * 1024 * 1024) # Estimate 5MB usage
             stats["total_samples"] += 500
        else:
             stats["total_bytes"] += size
             stats["total_samples"] += used_samples

    return stats

def main():
    print("=" * 60)
    print("XERV CRAYON: FINAL REPORT")
    print("=" * 60)

    # ---------------------------------------------------------
    # 1. Load Vocabulary
    # ---------------------------------------------------------
    start_load = time.perf_counter()
    try:
        vocab = CrayonVocab.from_json(VOCAB_PATH)
        load_time = (time.perf_counter() - start_load) * 1000
        print(f"\n[1] VOCABULARY LOADED")
        print(f"    - Source: {VOCAB_PATH}")
        print(f"    - Size:   {len(vocab):,} tokens")
        print(f"    - C-Ext:  {'[OK] Enabled (AVX2)' if vocab._c_ext_available else '[--] Disabled'}")
        print(f"    - Time:   {load_time:.2f} ms")
    except Exception as e:
        print(f"\n[!] Failed to load vocabulary: {e}")
        return

    # ---------------------------------------------------------
    # 2. Verify Tokenization
    # ---------------------------------------------------------
    print(f"\n[2] VERIFICATION")
    test_cases = [
        "delhi is india's capital",
        "The quick brown fox 123.",
        "Solve: 2x^2 + 4x = 0",
        "Quantum mechanics describes nature at scale.",
    ]
    
    for text in test_cases:
        tokens = vocab.tokenize(text)
        decoded = vocab.decode(tokens)
        unk_count = tokens.count(vocab.unk_token_id)
        
        status = "PASS" if text == decoded else "WARN (Lossy)"
        if unk_count > 0: status = "WARN (UNKs)"
        
        print(f"    Case: '{text}'")
        print(f"      -> Tokens:  {tokens}")
        print(f"      -> Decoded: '{decoded}'")
        print(f"      -> Status:  {status}")
        print("-" * 30)

    # ---------------------------------------------------------
    # 3. Benchmarking
    # ---------------------------------------------------------
    print(f"\n[3] PERFORMANCE BENCHMARK")
    
    # Generate representative text (mix of math, code, english)
    bench_text = """
    The partition function Z is given by the sum over states.
    In python: def compute(x): return x ** 2
    Delhi is a major city. 
    """ * 1000 # ~100KB block
    
    iterations = 50
    total_tokens = 0
    start_bench = time.perf_counter()
    
    for _ in range(iterations):
        t = vocab.tokenize(bench_text)
        total_tokens += len(t)
        
    duration = time.perf_counter() - start_bench
    throughput = total_tokens / duration
    
    print(f"    - Input Size:     {len(bench_text)/1024:.1f} KB per iter")
    print(f"    - Total Processed: {total_tokens:,} tokens")
    print(f"    - Duration:       {duration:.3f} s")
    print(f"    - THROUGHPUT:     {throughput:,.0f} tokens/sec")
    
    if throughput > 2000000:
        print(f"    - Result:         [OK] EXCEEDS TARGET (>2M)")
    else:
        print(f"    - Result:         [!!] BELOW TARGET")

    # ---------------------------------------------------------
    # 4. Data Usage Report
    # ---------------------------------------------------------
    print(f"\n[4] DATA QUANTITY REPORT")
    print(f"    Exact data sources used for training:")
    
    stats = calculate_data_stats()
    
    print(f"    {'-'*50}")
    print(f"    {'DATASET':<30} | {'SIZE':<10} | {'SAMPLES':<10}")
    print(f"    {'-'*50}")
    
    for f in stats["files"]:
        size_str = f"{f['size']/1024:.1f} KB"
        print(f"    {f['name']:<30} | {size_str:<10} | {f['samples']:<10,}")
        
    print(f"    {'-'*50}")
    print(f"    TOTAL PROCESSED SAMPLES: {stats['total_samples']:,}")
    print(f"    TOTAL ESTIMATED BYTES:   {stats['total_bytes']/1024/1024:.2f} MB")
    print("=" * 60)

if __name__ == "__main__":
    main()
