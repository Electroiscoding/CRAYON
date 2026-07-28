# Crayon Colab Benchmark Script (v5.5.0)
# Run in Google Colab:
#   !pip install -U xerv-crayon
#   !python measurecolab.py

import sys, time, gc

try:
    from crayon import CrayonVocab
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-U", "xerv-crayon"])
    from crayon import CrayonVocab

def run_benchmark():
    print("=" * 75)
    print("🚀 CRAYON TURBO ENGINE BENCHMARK (v5.5.0)")
    print("=" * 75)

    vocab = CrayonVocab(device="auto")
    vocab.load_profile("lite")
    turbo = vocab._turbo_backend

    if turbo:
        print(f"Hardware Info: {turbo.get_hardware_info()}")
    else:
        print(f"Backend Device: {vocab.device}")

    base_str = "The quick brown fox jumps over the lazy dog. "

    bench_configs = [
        ("100 KB", 2300),
        ("1 MB",   23000),
        ("4 MB",   92000),
        ("10 MB",  230000)
    ]

    print("\n" + "=" * 75)
    print(f"| {'Label':<10} | {'Actual Size':<13} | {'Tokens':<10} | {'Throughput':<12} | {'Latency':<10} | {'Status':<8} |")
    print("-" * 75)

    for label, mult in bench_configs:
        test_payload = base_str * mult
        iterations = 50 if mult < 10000 else 10

        # Warmup
        turbo.tokenize(test_payload)

        gc.collect()
        gc.disable()
        t0 = time.perf_counter()
        for _ in range(iterations):
            tokens = turbo.tokenize(test_payload)
        t1 = time.perf_counter()
        gc.enable()

        elapsed = (t1 - t0) / iterations
        tps = len(tokens) / elapsed
        actual_mb = len(test_payload.encode('utf-8')) / (1024 * 1024)
        status = "✅ PASS" if tps >= 105e6 else "❌ FAIL"

        print(f"| {label:<10} | {actual_mb:>10.2f} MB | {len(tokens):>10,} | {tps/1e6:>10.2f} M | {elapsed*1000:>10.2f} ms | {status:<8} |")

    print("=" * 75 + "\n")

if __name__ == "__main__":
    run_benchmark()
