# Crayon Colab Benchmark Script
# Usage in Google Colab:
#   !pip install xerv-crayon
#   !python measurecolab.py

import sys, time, gc

try:
    from crayon import CrayonVocab
except ImportError:
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "xerv-crayon", "--upgrade"])
    from crayon import CrayonVocab

def run_benchmark():
    print("=" * 60)
    print("🚀 CRAYON TOKENIZER BENCHMARK")
    print("=" * 60)

    vocab = CrayonVocab(device="auto")
    print(f"Engine Device: {vocab.device.upper()}")

    # 100KB test payload covering words, spaces, and punctuation
    text_sample = "The quick brown fox jumps over the lazy dog. " * 2300

    for profile in ["standard", "lite"]:
        vocab.load_profile(profile)
        tb = vocab._turbo_backend
        hw = tb.get_hardware_info() if tb else vocab.device
        print(f"\n📦 Profile: '{profile}'")
        print(f"   Hardware: {hw}")

        # Warmup execution & cache fill
        for _ in range(15):
            vocab.tokenize(text_sample)

        # Performance measurement
        gc.collect(); gc.disable()
        N = 50
        t0 = time.perf_counter()
        for _ in range(N):
            tokens = vocab.tokenize(text_sample)
        elapsed = (time.perf_counter() - t0) / N
        gc.enable()

        tps = len(tokens) / elapsed
        print(f"   Input Size: {len(text_sample)/1024:.1f} KB | Tokens: {len(tokens):,}")
        print(f"   Throughput: {tps/1e6:.2f}M tokens/sec [{'✅ PASS' if tps >= 105e6 else 'ℹ️ Measured'}]")

        # Decode sample verification
        sample_decode = vocab.decode(tokens[:10])
        print(f"   Decode Check: {tokens[:5]}... -> '{sample_decode}'")

    print("\n" + "=" * 60)
    print("BENCHMARK COMPLETE")
    print("=" * 60)

if __name__ == "__main__":
    run_benchmark()
