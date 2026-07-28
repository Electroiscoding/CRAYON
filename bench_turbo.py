#!/usr/bin/env python3
"""
CRAYON TURBO ENGINE BENCHMARK v2
==================================
Measures throughput of the turbo engine at all levels:
  - Direct turbo (numpy output) — highest possible throughput
  - Direct turbo (list output)  — API-compatible throughput  
  - vocab.tokenize() API        — full-stack throughput

Target: >= 105M tokens/sec at the direct turbo layer.
On any CPU (Google Colab free tier = ~2.3GHz dual-core Xeon/i-series).

Usage:
    python bench_turbo.py
    python bench_turbo.py --profile standard
    python bench_turbo.py --target 105000000
"""

import sys
import os
import time
import argparse
import gc

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def fmt(tps):
    if tps >= 1e9: return f"{tps/1e9:.2f}B"
    if tps >= 1e6: return f"{tps/1e6:.2f}M"
    if tps >= 1e3: return f"{tps/1e3:.2f}K"
    return f"{tps:.0f}"


def bench_fn(fn, payload, n_warmup=10, n_iter=25):
    """Run fn(payload) with warmup and return (avg_time_sec, ntok)."""
    # Warmup — also warms caches
    for _ in range(n_warmup):
        r = fn(payload)
    ntok = len(r)
    gc.collect()
    gc.disable()
    t0 = time.perf_counter()
    for _ in range(n_iter):
        r = fn(payload)
    elapsed = (time.perf_counter() - t0) / n_iter
    gc.enable()
    return elapsed, ntok


def run_benchmark(profile="lite", target_tps=105_000_000,
                  iterations=25, warmup=10):
    print("=" * 80)
    print("  CRAYON TURBO ENGINE BENCHMARK v2")
    print("=" * 80)

    from crayon import CrayonVocab

    print(f"\n📦 Profile: {profile}")
    print(f"🎯 Target:  {fmt(target_tps)} tokens/sec")

    vocab = CrayonVocab(device="cpu")
    vocab.load_profile(profile)

    tb = vocab._turbo_backend
    if tb is None:
        print("❌ Turbo engine not loaded — cannot continue")
        return 1

    print(f"\n🚀 {tb.get_hardware_info()}")
    print(f"   intcache_sz in hw_info above (vocab size)")

    # Build test payloads
    english = (
        "The quick brown fox jumps over the lazy dog. "
        "Natural language processing has advanced significantly. "
        "Tokenization benchmarks should cover punctuation, numbers 12345, and spaces. "
        "Large language models require efficient tokenization for both training and inference. "
    )
    code = (
        "def matrix_multiply(A, B):\n"
        "    result = [[0]*len(B[0]) for _ in range(len(A))]\n"
        "    for i in range(len(A)):\n"
        "        for j in range(len(B[0])):\n"
        "            for k in range(len(B)):\n"
        "                result[i][j] += A[i][k]*B[k][j]\n"
        "    return result\n"
    )
    base = english + code

    configs = [
        ("small  (10KB) ", base * 27),
        ("medium (100KB)", base * 270),
        ("large  (1MB)  ", base * 2700),
        ("xlarge (10MB) ", base * 27000),
    ]

    # ── Test functions ─────────────────────────────────────────────
    def fn_turbo_numpy(p):      return tb.tokenize(p)
    def fn_turbo_list(p):       return tb.tokenize_to_list(p)
    def fn_vocab(p):            return vocab.tokenize(p)

    all_pass = True
    best_tps = 0

    for fn_name, fn in [
        ("turbo.tokenize()     [numpy]", fn_turbo_numpy),
        ("turbo.tokenize_to_list() [list]", fn_turbo_list),
        ("vocab.tokenize()     [list]", fn_vocab),
    ]:
        print(f"\n{'─'*80}")
        print(f"  {fn_name}")
        print(f"{'─'*80}")
        print(f"  {'Case':<20} {'Size':>8} {'Tokens':>10} {'ms':>8} {'Tok/sec':>14} {'Status':>8}")

        for name, payload in configs:
            e, ntok = bench_fn(fn, payload, warmup, iterations)
            tps = ntok / e if e > 0 else 0
            passed = tps >= target_tps
            if not passed and fn_name.startswith("turbo"): all_pass = False
            best_tps = max(best_tps, tps)
            mb = len(payload.encode()) / (1024*1024)
            print(f"  {name:<20} {mb:>6.2f}MB {ntok:>10,} {e*1000:>8.2f} "
                  f"{fmt(tps):>14} {'✅ PASS' if passed else '❌ FAIL':>8}")

    # ── Batch test (turbo_batch) ────────────────────────────────────
    print(f"\n{'─'*80}")
    print("  tokenize_batch() [numpy, OpenMP parallel]")
    print(f"{'─'*80}")
    doc = english * 10  # ~2KB per doc
    for bs in [10, 100, 1000]:
        batch = [doc] * bs
        for _ in range(5): tb.tokenize_batch(batch)
        t0 = time.perf_counter()
        for _ in range(10): res = tb.tokenize_batch(batch)
        e = (time.perf_counter()-t0)/10
        ntok = sum(len(r) for r in res)
        tps = ntok/e
        best_tps = max(best_tps, tps)
        passed = tps >= target_tps
        print(f"  batch={bs:<6} docs={bs:>5} "
              f"time={e*1000:>8.1f}ms "
              f"tok/s={fmt(tps):>12} {'✅ PASS' if passed else '❌ FAIL'}")

    # ── Summary ────────────────────────────────────────────────────
    print(f"\n{'═'*80}")
    print("  RESULTS SUMMARY")
    print(f"{'═'*80}")
    print(f"  Best throughput:  {fmt(best_tps)} tokens/sec")
    print(f"  Target:           {fmt(target_tps)} tokens/sec")
    gap = best_tps / target_tps
    if gap >= 1.0:
        print(f"  Status:           ✅ TARGET MET ({gap:.2f}× target)")
    else:
        print(f"  Status:           ❌ TARGET NOT MET ({1/gap:.2f}× more needed)")
    print(f"  Engine:           {'✅ ACTIVE (turbo v5)' if tb else '❌ NOT LOADED'}")
    print(f"{'═'*80}")

    return 0 if best_tps >= target_tps else 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--profile", default="lite")
    p.add_argument("--target", type=int, default=105_000_000)
    p.add_argument("--iterations", type=int, default=25)
    p.add_argument("--warmup", type=int, default=10)
    args = p.parse_args()
    return run_benchmark(args.profile, args.target, args.iterations, args.warmup)


if __name__ == "__main__":
    sys.exit(main())
