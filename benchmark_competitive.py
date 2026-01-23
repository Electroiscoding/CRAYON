"""
XERV CRAYON V2.0 - Competitive Benchmark Against All Major Tokenizers
======================================================================
100% HONEST. NO SUGARCOATING. DATA-DRIVEN.

Compares against:
- OpenAI tiktoken (GPT-4, GPT-3.5)
- HuggingFace tokenizers (BERT, GPT-2, LLaMA, T5)

All metrics: Tokens/sec, MB/sec, Load Time, Avg Time per Iteration
"""

import sys
import os
import time
import mmap
from datetime import datetime
import json

# Add paths
sys.path.insert(0, os.path.join(os.getcwd(), "build", "lib.win-amd64-cpython-313"))
sys.path.insert(0, os.path.join(os.getcwd(), "src"))

# Configuration
ITERATIONS = 10
WARMUP = 2

# Test text - realistic mixed content
BASE_TEXT = """T
def matrix_multiply(A, B):
    # Standard O(n^3) matrix multiplication
    result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
    return result
"""

TEST_TEXT = BASE_TEXT * 100  # ~62KB

print("=" * 100)
print("XERV CRAYON V2.0 - COMPETITIVE TOKENIZER BENCHMARK")
print("100% HONEST. NO SUGARCOATING. DATA-DRIVEN.")
print("=" * 100)
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Test Text Size: {len(TEST_TEXT):,} bytes ({len(TEST_TEXT)/1024:.1f} KB)")
print(f"Iterations: {ITERATIONS} (+ {WARMUP} warmup)")
print("=" * 100)
print()

results = []

def benchmark_tokenizer(name, tokenize_fn, load_fn=None, vocab_size=None):
    """Benchmark a tokenizer with all metrics."""
    print(f"[BENCH] {name}...", end=" ", flush=True)
    
    try:
        # Measure load time if provided
        load_time_ms = 0
        if load_fn:
            start = time.perf_counter()
            load_fn()
            load_time_ms = (time.perf_counter() - start) * 1000
        
        # Warmup
        for _ in range(WARMUP):
            _ = tokenize_fn(TEST_TEXT)
        
        # Benchmark iterations
        times = []
        token_counts = []
        
        for _ in range(ITERATIONS):
            start = time.perf_counter()
            tokens = tokenize_fn(TEST_TEXT)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            token_counts.append(len(tokens) if hasattr(tokens, '__len__') else len(list(tokens)))
        
        avg_time = sum(times) / len(times)
        min_time = min(times)
        max_time = max(times)
        avg_tokens = sum(token_counts) / len(token_counts)
        total_tokens = int(avg_tokens)  # Token count for this text
        
        text_bytes = len(TEST_TEXT.encode('utf-8'))
        tokens_per_sec = avg_tokens / avg_time
        mb_per_sec = (text_bytes / 1024 / 1024) / avg_time
        
        result = {
            "name": name,
            "status": "OK",
            "vocab_size": vocab_size or "N/A",
            "avg_tokens": avg_tokens,
            "token_count": total_tokens,
            "load_time_ms": load_time_ms,
            "avg_time_ms": avg_time * 1000,
            "min_time_ms": min_time * 1000,
            "max_time_ms": max_time * 1000,
            "tokens_per_sec": tokens_per_sec,
            "mb_per_sec": mb_per_sec,
        }
        
        print(f"[OK] {tokens_per_sec:,.0f} tok/s | {total_tokens:,} tokens | {avg_time*1000:.2f}ms | Load: {load_time_ms:.2f}ms")
        return result
        
    except Exception as e:
        print(f"[FAIL] ERROR: {e}")
        return {"name": name, "status": "FAIL", "error": str(e)}

# ============================================================================
# 1. XERV CRAYON (Lite Profile - 50k vocab)
# ============================================================================
print("\n" + "="*50)
print("XERV CRAYON")
print("="*50)

try:
    from crayon import CrayonVocab
    
    def load_crayon_lite():
        global _crayon_vocab
        _crayon_vocab = CrayonVocab.load_profile("code")
    
    load_crayon_lite()  # Pre-load for benchmark
    
    if _crayon_vocab.fast_mode:
        print("  [OK] Loaded with AVX2 engine")
    else:
        print("  [WARN] Loaded in fallback mode (no AVX2)")
    
    results.append(benchmark_tokenizer(
        "CRAYON (lite, 50k)",
        lambda text: _crayon_vocab.tokenize(text),
        load_fn=load_crayon_lite,
        vocab_size=50000
    ))
        
except ImportError as e:
    print(f"  CRAYON not available: {e}")
except Exception as e:
    print(f"  CRAYON load error: {e}")

# ============================================================================
# 2. OpenAI tiktoken
# ============================================================================
print("\n" + "="*50)
print("OpenAI tiktoken")
print("="*50)

try:
    import tiktoken
    
    # GPT-4 / GPT-3.5-turbo (cl100k_base)
    def load_tiktoken_cl100k():
        global _enc_cl100k
        _enc_cl100k = tiktoken.get_encoding("cl100k_base")
    
    load_tiktoken_cl100k()
    results.append(benchmark_tokenizer(
        "tiktoken (cl100k/GPT-4)",
        lambda text: _enc_cl100k.encode(text),
        load_fn=load_tiktoken_cl100k,
        vocab_size=100000
    ))
    
    # GPT-3 (p50k_base)
    def load_tiktoken_p50k():
        global _enc_p50k
        _enc_p50k = tiktoken.get_encoding("p50k_base")
    
    load_tiktoken_p50k()
    results.append(benchmark_tokenizer(
        "tiktoken (p50k/GPT-3)",
        lambda text: _enc_p50k.encode(text),
        load_fn=load_tiktoken_p50k,
        vocab_size=50000
    ))
    
except ImportError:
    print("  tiktoken not installed. Run: pip install tiktoken")

# ============================================================================
# 3. HuggingFace Tokenizers
# ============================================================================
print("\n" + "="*50)
print("HuggingFace Tokenizers")
print("="*50)

try:
    from transformers import AutoTokenizer
    import warnings
    warnings.filterwarnings("ignore")
    
    # GPT-2 (BPE, 50k vocab)
    try:
        def load_gpt2():
            global _gpt2_tok
            _gpt2_tok = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
        
        load_gpt2()
        results.append(benchmark_tokenizer(
            "HF GPT-2 (BPE)",
            lambda text: _gpt2_tok.encode(text),
            load_fn=load_gpt2,
            vocab_size=50257
        ))
    except Exception as e:
        print(f"  GPT-2 failed: {e}")
    
    # BERT (WordPiece, 30k vocab)
    try:
        def load_bert():
            global _bert_tok
            _bert_tok = AutoTokenizer.from_pretrained("bert-base-uncased", use_fast=True)
        
        load_bert()
        results.append(benchmark_tokenizer(
            "HF BERT (WordPiece)",
            lambda text: _bert_tok.encode(text),
            load_fn=load_bert,
            vocab_size=30522
        ))
    except Exception as e:
        print(f"  BERT failed: {e}")
    
    # T5 (SentencePiece, 32k vocab)
    try:
        def load_t5():
            global _t5_tok
            _t5_tok = AutoTokenizer.from_pretrained("t5-small", use_fast=True)
        
        load_t5()
        results.append(benchmark_tokenizer(
            "HF T5 (SentencePiece)",
            lambda text: _t5_tok.encode(text),
            load_fn=load_t5,
            vocab_size=32000
        ))
    except Exception as e:
        print(f"  T5 failed: {e}")
    
    # LLaMA (if available)
    try:
        def load_llama():
            global _llama_tok
            _llama_tok = AutoTokenizer.from_pretrained("huggyllama/llama-7b", use_fast=True)
        
        load_llama()
        results.append(benchmark_tokenizer(
            "HF LLaMA (SP-BPE)",
            lambda text: _llama_tok.encode(text),
            load_fn=load_llama,
            vocab_size=32000
        ))
    except Exception as e:
        print(f"  LLaMA skipped (needs auth)")
        
except ImportError:
    print("  transformers not installed. Run: pip install transformers")

# ============================================================================
# RESULTS SUMMARY
# ============================================================================
print()
print("=" * 100)
print("RESULTS SUMMARY (Real Tokenizers Only - Sorted by Tokens/sec)")
print("=" * 100)
print()

ok_results = [r for r in results if r.get("status") == "OK"]
ok_results.sort(key=lambda x: x["tokens_per_sec"], reverse=True)

print(f"{'Tokenizer':<28} | {'Vocab':>8} | {'Tokens':>10} | {'Tokens/sec':>14} | {'MB/sec':>8} | {'Load Time':>10} | {'Avg Time':>10}")
print("-" * 110)

for r in ok_results:
    vocab = f"{r['vocab_size']:,}" if isinstance(r['vocab_size'], int) else r['vocab_size']
    token_count = f"{r['token_count']:,}" if 'token_count' in r else "N/A"
    print(f"{r['name']:<28} | {vocab:>8} | {token_count:>10} | {r['tokens_per_sec']:>14,.0f} | {r['mb_per_sec']:>8.2f} | {r['load_time_ms']:>9.2f}ms | {r['avg_time_ms']:>9.2f}ms")

print("-" * 100)

# ============================================================================
# MATPLOTLIB VISUALIZATION - BAR CHART + HISTOGRAM
# ============================================================================
print()
print("Generating visualizations...")

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    import numpy as np
    
    names = [r['name'] for r in ok_results]
    tokens_per_sec = [r['tokens_per_sec'] for r in ok_results]
    times_ms = [r['avg_time_ms'] for r in ok_results]
    load_times = [r['load_time_ms'] for r in ok_results]
    
    colors = ['#2ecc71' if 'CRAYON' in name else '#3498db' for name in names]
    
    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Chart 1: Tokens/sec (Bar Chart)
    ax1 = axes[0, 0]
    bars1 = ax1.barh(names, tokens_per_sec, color=colors)
    ax1.set_xlabel('Tokens per Second', fontsize=11)
    ax1.set_title('Tokenization Speed\n(Higher is Better)', fontsize=13, fontweight='bold')
    ax1.ticklabel_format(style='plain', axis='x')
    for bar, val in zip(bars1, tokens_per_sec):
        ax1.text(val + max(tokens_per_sec)*0.01, bar.get_y() + bar.get_height()/2, 
                f'{val:,.0f}', va='center', fontsize=9)
    
    # Chart 2: Avg Time (Bar Chart)
    ax2 = axes[0, 1]
    bars2 = ax2.barh(names, times_ms, color=colors)
    ax2.set_xlabel('Time (milliseconds)', fontsize=11)
    ax2.set_title('Tokenization Time\n(Lower is Better)', fontsize=13, fontweight='bold')
    for bar, val in zip(bars2, times_ms):
        ax2.text(val + max(times_ms)*0.01, bar.get_y() + bar.get_height()/2, 
                f'{val:.2f}ms', va='center', fontsize=9)
    
    # Chart 3: Tokens/sec Histogram
    ax3 = axes[1, 0]
    x_pos = np.arange(len(names))
    bars3 = ax3.bar(x_pos, tokens_per_sec, color=colors, edgecolor='black', linewidth=0.5)
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels([n.replace(' ', '\n') for n in names], fontsize=8, rotation=0)
    ax3.set_ylabel('Tokens per Second', fontsize=11)
    ax3.set_title('Speed Comparison (Histogram)\n(Higher is Better)', fontsize=13, fontweight='bold')
    ax3.ticklabel_format(style='plain', axis='y')
    for bar, val in zip(bars3, tokens_per_sec):
        ax3.text(bar.get_x() + bar.get_width()/2, val + max(tokens_per_sec)*0.02, 
                f'{val/1e6:.1f}M', ha='center', va='bottom', fontsize=9)
    
    # Chart 4: Load Time Histogram
    ax4 = axes[1, 1]
    bars4 = ax4.bar(x_pos, load_times, color=colors, edgecolor='black', linewidth=0.5)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels([n.replace(' ', '\n') for n in names], fontsize=8, rotation=0)
    ax4.set_ylabel('Load Time (ms)', fontsize=11)
    ax4.set_title('Load Time Comparison (Histogram)\n(Lower is Better)', fontsize=13, fontweight='bold')
    for bar, val in zip(bars4, load_times):
        ax4.text(bar.get_x() + bar.get_width()/2, val + max(load_times)*0.02, 
                f'{val:.1f}ms', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    fig_path = "benchmark_comparison.png"
    plt.savefig(fig_path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f"[OK] Saved: {fig_path}")
    plt.close()
    
except ImportError:
    print("matplotlib not installed. Run: pip install matplotlib")
except Exception as e:
    print(f"Visualization error: {e}")

# ============================================================================
# SAVE RESULTS TO MARKDOWN
# ============================================================================
print()
print("Saving results...")

with open("BENCHMARK_RESULTS.md", "w", encoding="utf-8") as f:
    f.write("# XERV Crayon V2.0 - Competitive Benchmark Results\n\n")
    f.write("**100% HONEST. NO SUGARCOATING. DATA-DRIVEN.**\n\n")
    f.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write(f"**Test Text Size:** {len(TEST_TEXT):,} bytes ({len(TEST_TEXT)/1024:.1f} KB)\n\n")
    f.write(f"**Iterations:** {ITERATIONS} (+ {WARMUP} warmup)\n\n")
    f.write("---\n\n")
    
    f.write("## Results (Real Tokenizers Only - Sorted by Speed)\n\n")
    f.write("| Tokenizer | Vocab Size | Token Count | Tokens/sec | MB/sec | Load Time | Avg Time | Min Time | Max Time |\n")
    f.write("| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n")
    
    for r in ok_results:
        vocab = f"{r['vocab_size']:,}" if isinstance(r['vocab_size'], int) else r['vocab_size']
        token_count = f"{r['token_count']:,}" if 'token_count' in r else "N/A"
        f.write(f"| **{r['name']}** | {vocab} | {token_count} | {r['tokens_per_sec']:,.0f} | {r['mb_per_sec']:.2f} | {r['load_time_ms']:.2f}ms | {r['avg_time_ms']:.2f}ms | {r['min_time_ms']:.2f}ms | {r['max_time_ms']:.2f}ms |\n")
    
    f.write("\n---\n\n")
    f.write("## Visualization\n\n")
    f.write("![Benchmark Comparison](benchmark_comparison.png)\n\n")
    
    f.write("---\n\n")
    f.write("## Speed Comparison\n\n")
    
    if ok_results:
        crayon_result = next((r for r in ok_results if 'CRAYON' in r['name']), None)
        if crayon_result:
            f.write("| Tokenizer | Speed vs CRAYON |\n")
            f.write("| :--- | ---: |\n")
            for r in ok_results:
                ratio = crayon_result['tokens_per_sec'] / r['tokens_per_sec']
                if 'CRAYON' in r['name']:
                    f.write(f"| **{r['name']}** | **baseline** |\n")
                elif ratio > 1:
                    f.write(f"| {r['name']} | {ratio:.1f}x slower |\n")
                else:
                    f.write(f"| {r['name']} | {1/ratio:.1f}x faster |\n")
    
    f.write("\n---\n\n")
    f.write("## Tokenizers Tested\n\n")
    f.write("| Tokenizer | Type | Vocab Size | Source |\n")
    f.write("| :--- | :--- | ---: | :--- |\n")
    f.write("| CRAYON (lite) | DAT + C++ | 50,000 | Custom engine |\n")
    f.write("| tiktoken cl100k | BPE | 100,000 | OpenAI GPT-4 |\n")
    f.write("| tiktoken p50k | BPE | 50,000 | OpenAI GPT-3 |\n")
    f.write("| HF GPT-2 | BPE (Rust) | 50,257 | HuggingFace |\n")
    f.write("| HF BERT | WordPiece | 30,522 | HuggingFace |\n")
    f.write("| HF T5 | SentencePiece | 32,000 | HuggingFace |\n")
    
    f.write("\n---\n\n")
    f.write("## Reproducibility\n\n")
    f.write("```bash\n")
    f.write("pip install tiktoken transformers matplotlib\n")
    f.write("python benchmark_competitive.py\n")
    f.write("```\n")

print("[OK] Saved: BENCHMARK_RESULTS.md")

# Save JSON
with open("benchmark_results.json", "w") as f:
    json.dump({
        "date": datetime.now().isoformat(),
        "test_text_bytes": len(TEST_TEXT),
        "iterations": ITERATIONS,
        "results": ok_results
    }, f, indent=2)

print("[OK] Saved: benchmark_results.json")

print()
print("=" * 100)
print("BENCHMARK COMPLETE")
print("=" * 100)
