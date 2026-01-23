try:
    import tiktoken
    import transformers
    import crayon
    from importlib.metadata import version
    if version("xerv-crayon") < "2.0.3":
        raise ImportError("Old version detected")
except ImportError:
    import subprocess
    import sys
    import os
    print("Installing xerv-crayon>=2.0.3...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--upgrade", "xerv-crayon>=2.0.3", "tiktoken", "transformers"])
    print("Installation complete. PLEASE RESTART RUNTIME (Runtime > Restart Session) and run this cell again.")
    # Attempt to kill own process to force restart in some envs
    # os.kill(os.getpid(), 9)
    sys.exit(0)

import time
import os
import tiktoken
from transformers import GPT2TokenizerFast, LlamaTokenizerFast
from crayon import CrayonVocab

print("\nPREPARING BENCHMARK DATA...")
text_chunk = """
def matrix_multiply(A, B):
    result = [[0 for _ in range(len(B[0]))] for _ in range(len(A))]
    for i in range(len(A)):
        for j in range(len(B[0])):
            for k in range(len(B)):
                result[i][j] += A[i][k] * B[k][j]
    return result

The quick brown fox jumps over the lazy dog. 
Scientific research requires precision, data, and reproducible results.
""" * 500
test_text = text_chunk * 20
bytes_size = len(test_text.encode('utf-8'))
print(f"Test Text Size: {bytes_size / 1024 / 1024:.2f} MB")

print("\nLOADING TOKENIZERS...")

# 1. CRAYON
start = time.perf_counter()
crayon_vocab = CrayonVocab.load_profile("lite")
crayon_load = (time.perf_counter() - start) * 1000
print(f"CRAYON Loaded: {crayon_load:.2f}ms (AVX2: {crayon_vocab.fast_mode})")

# 2. TikToken
start = time.perf_counter()
enc_gpt4 = tiktoken.get_encoding("cl100k_base")
tiktoken_load = (time.perf_counter() - start) * 1000
print(f"TikToken Loaded: {tiktoken_load:.2f}ms")

# 3. HF GPT-2
try:
    start = time.perf_counter()
    hf_gpt2 = GPT2TokenizerFast.from_pretrained("gpt2")
    hf_load = (time.perf_counter() - start) * 1000
    print(f"HF GPT-2 Loaded: {hf_load:.2f}ms")
except:
    hf_gpt2 = None

print("\nRUNNING BENCHMARK (5 Iterations)...")

def run_bench(name, func):
    times = []
    tokens = 0
    for _ in range(5):
        s = time.perf_counter()
        res = func(test_text)
        e = time.perf_counter()
        times.append(e - s)
        tokens = len(res)
    
    avg_time = sum(times) / len(times)
    tps = tokens / avg_time
    return tps, avg_time, tokens

results = []

# CRAYON
tps, avg, toks = run_bench("CRAYON (lite)", lambda t: crayon_vocab.tokenize(t))
results.append(("CRAYON", tps, avg, toks))
print(f"CRAYON: {tps:,.0f} tokens/sec")

# TikToken
tps, avg, toks = run_bench("TikToken (GPT-4)", lambda t: enc_gpt4.encode(t))
results.append(("TikToken", tps, avg, toks))
print(f"TikToken: {tps:,.0f} tokens/sec")

# HF
if hf_gpt2:
    tps, avg, toks = run_bench("HF GPT-2", lambda t: hf_gpt2.encode(t))
    results.append(("HF GPT-2", tps, avg, toks))
    print(f"HF GPT-2: {tps:,.0f} tokens/sec")

print("\n" + "="*65)
print(f"{'TOKENIZER':<20} | {'TOKENS/SEC':>15} | {'SPEEDUP':>10} | {'TIME':>8}")
print("-" * 65)

base_tps = results[0][1]

for name, tps, avg, _ in results:
    multiplier = base_tps / tps
    if name == "CRAYON":
        x_factor = "1.0x"
    else:
        x_factor = f"{multiplier:.1f}x"
        
    print(f"{name:<20} | {tps:,.0f} | {x_factor:>10} | {avg*1000:.2f}ms")
print("="*65)
mn = min(results, key=lambda x: x[2])
mx = max(results, key=lambda x: x[2])
print(f"\nWINNER: {mn[0]} is {mx[2]/mn[2]:.1f}x FASTER than {mx[0]}")
