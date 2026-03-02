"""
XERV Crayon CLI - Command Line Interface
=========================================
Provides command-line tools for benchmarking and vocabulary management.
"""
import sys
import time
import argparse


def run_benchmark():
    """Run a quick benchmark of the Crayon tokenizer."""
    parser = argparse.ArgumentParser(
        prog='crayon-benchmark',
        description='XERV Crayon Tokenizer Benchmark Tool'
    )
    parser.add_argument(
        '--profile', '-p',
        default='lite',
        choices=['lite', 'standard'],
        help='Vocabulary profile to use (default: lite)'
    )
    parser.add_argument(
        '--iterations', '-n',
        type=int,
        default=10,
        help='Number of benchmark iterations (default: 10)'
    )
    parser.add_argument(
        '--text', '-t',
        default=None,
        help='Custom text to tokenize (default: built-in test text)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("XERV CRAYON TOKENIZER BENCHMARK")
    print("=" * 60)
    
    try:
        from crayon import CrayonVocab
    except ImportError as e:
        print(f"[ERROR] Failed to import crayon: {e}")
        print("Make sure xerv-crayon is properly installed.")
        sys.exit(1)
    
    # Load vocabulary
    print(f"\n[INFO] Loading profile: {args.profile}")
    start = time.perf_counter()
    
    try:
        vocab = CrayonVocab.load_profile(args.profile)
    except Exception as e:
        print(f"[ERROR] Failed to load profile: {e}")
        sys.exit(1)
    
    load_time = (time.perf_counter() - start) * 1000
    
    if vocab.fast_mode:
        print(f"[OK] Loaded with AVX2 engine ({load_time:.2f}ms)")
    else:
        print(f"[WARN] Loaded in fallback mode ({load_time:.2f}ms)")
    
    # Prepare test text
    if args.text:
        test_text = args.text
    else:
        test_text = """
def matrix_multiply(A, B):
    # Standard O(n^3) matrix multiplication
    result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
    return result

The quick brown fox jumps over the lazy dog. 
Machine learning models require efficient tokenization for optimal performance.
""" * 100  # Repeat for meaningful benchmark
    
    text_size = len(test_text.encode('utf-8'))
    print(f"\n[INFO] Test text size: {text_size:,} bytes ({text_size/1024:.1f} KB)")
    print(f"[INFO] Iterations: {args.iterations}")
    
    # Warmup
    print("\n[INFO] Warming up...")
    for _ in range(2):
        _ = vocab.tokenize(test_text)
    
    # Benchmark
    print("[INFO] Running benchmark...")
    times = []
    token_counts = []
    
    for i in range(args.iterations):
        start = time.perf_counter()
        tokens = vocab.tokenize(test_text)
        elapsed = time.perf_counter() - start
        times.append(elapsed)
        token_counts.append(len(tokens))
    
    # Calculate metrics
    avg_time = sum(times) / len(times)
    min_time = min(times)
    max_time = max(times)
    avg_tokens = sum(token_counts) / len(token_counts)
    tokens_per_sec = avg_tokens / avg_time
    mb_per_sec = (text_size / 1024 / 1024) / avg_time
    
    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    print(f"  Profile:        {args.profile}")
    print(f"  Token Count:    {int(avg_tokens):,}")
    print(f"  Tokens/sec:     {tokens_per_sec:,.0f}")
    print(f"  MB/sec:         {mb_per_sec:.2f}")
    print(f"  Avg Time:       {avg_time*1000:.2f}ms")
    print(f"  Min Time:       {min_time*1000:.2f}ms")
    print(f"  Max Time:       {max_time*1000:.2f}ms")
    print("=" * 60)
    
    return 0


def main():
    """Main entry point."""
    return run_benchmark()


if __name__ == '__main__':
    sys.exit(main())
