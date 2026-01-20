
import time
import sys
import os
from pathlib import Path

# Add src to sys.path
current_dir = Path(os.getcwd())
src_path = current_dir / "src"
sys.path.append(str(src_path))

from crayon.core.vocabulary import CrayonVocab
from crayon.core.profiles import PROFILES

def benchmark_profile(name, text, iterations=5):
    try:
        vocab = CrayonVocab.load_profile(name)
        
        # Warmup
        vocab.tokenize(text[:1000])
        
        total_chars = len(text)
        total_bytes = len(text.encode('utf-8'))
        
        start = time.time()
        for _ in range(iterations):
            vocab.tokenize(text)
        end = time.time()
        
        avg_time = (end - start) / iterations
        num_tokens = len(vocab.tokenize(text))
        
        tps = num_tokens / avg_time
        mbps = (total_bytes / avg_time) / (1024*1024)
        
        engine_type = "DAT (C++)" if vocab._c_ext_available else "Python (Slow)"
        
        return {
            "name": name.upper(),
            "tps": tps,
            "mbps": mbps,
            "time": avg_time,
            "vocab_size": len(vocab),
            "engine": engine_type
        }
    except Exception as e:
        return {"name": name.upper(), "error": str(e)}

def main():
    print("="*80)
    print("XERV CRAYON: DOUBLE-ARRAY TRIE BENCHMARK")
    print("="*80)
    
    # Use Shakespeare or large text
    text = ""
    res_path = current_dir / "src" / "crayon" / "resources" / "input.txt"
    if res_path.exists():
        with open(res_path, 'r', encoding='utf-8') as f:
            text = f.read()
    else:
        text = "The quick brown fox jumps over the lazy dog. " * 30000

    print(f"Dataset Size: {len(text)/1024/1024:.2f} MB")
    print("-" * 100)
    print(f"{'PROFILE':<15} | {'VOCAB':<8} | {'TOKENS/SEC':<15} | {'MB/SEC':<8} | {'ENGINE':<10}")
    print("-" * 100)
    
    results = []
    # Quick Check on Lite Only First
    res = benchmark_profile("lite", text)
    if "error" in res:
         print(f"{res['name']:<15} | ERROR: {res['error']}")
    else:
         print(f"{res['name']:<15} | {res['vocab_size']:<8} | {res['tps']:<15,.0f} | {res['mbps']:<8.2f} | {res['engine']:<10}")

    print("-" * 100)

if __name__ == "__main__":
    main()
