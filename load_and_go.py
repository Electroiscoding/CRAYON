"""
XERV Crayon - Load & Go Inference Mode Demo

This demonstrates the instant "inference only" workflow:
1. LOAD: Load pre-trained vocabulary from file
2. INIT: Auto-compile SIMD trie (milliseconds)
3. GO: Tokenize at >2M tokens/sec

No training phase required - just load and tokenize!
"""

import json
import time
from crayon import CrayonVocab


def load_and_go():
    print("=" * 60)
    print("XERV Crayon - Load & Go Inference Mode")
    print("=" * 60)
    
    # 1. LOAD: Load your pre-trained vocabulary
    print("\n[1] Loading vocabulary from vocab.json...")
    start = time.perf_counter()
    
    with open("vocab.json", "r") as f:
        token_list = json.load(f)
    
    load_time = (time.perf_counter() - start) * 1000
    print(f"    Loaded {len(token_list)} tokens in {load_time:.2f}ms")
    
    # 2. INIT: Auto-compile SIMD trie (instant)
    print("\n[2] Initializing C-Engine (auto-compiling SIMD trie)...")
    start = time.perf_counter()
    
    vocab = CrayonVocab(token_list)
    
    init_time = (time.perf_counter() - start) * 1000
    print(f"    C-Extension enabled: {vocab._c_ext_available}")
    print(f"    Trie compiled in {init_time:.2f}ms")
    
    # 3. GO: Tokenize immediately
    print("\n[3] Tokenizing...")
    text = "User just wants to tokenize and go!"
    
    start = time.perf_counter()
    tokens = vocab.tokenize(text)
    tokenize_time = (time.perf_counter() - start) * 1000000  # microseconds
    
    print(f"    Input:  '{text}'")
    print(f"    Tokens: {tokens}")
    print(f"    Decoded: {[vocab.id_to_token.get(i, '<UNK>') for i in tokens]}")
    print(f"    Time: {tokenize_time:.2f}us")
    
    # Benchmark throughput
    print("\n[4] Throughput Benchmark (1000 iterations)...")
    test_text = text * 100  # Make it longer
    
    start = time.perf_counter()
    for _ in range(1000):
        _ = vocab.tokenize(test_text)
    elapsed = time.perf_counter() - start
    
    total_chars = len(test_text) * 1000
    chars_per_sec = total_chars / elapsed
    print(f"    Throughput: {chars_per_sec:,.0f} chars/sec")
    print(f"    Estimated: ~{chars_per_sec/4:,.0f} tokens/sec")
    
    print("\n" + "=" * 60)
    print("[OK] Load & Go complete! Ready for production inference.")
    print("=" * 60)


if __name__ == "__main__":
    load_and_go()
