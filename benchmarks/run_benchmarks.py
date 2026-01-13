import os
import sys
import json

# Ensure benchmarks directory is in path for micro_bench import
script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, script_dir)

from crayon.core.vocabulary import CrayonVocab
from micro_bench import CrayonBenchmark

def main():
    print("=" * 60)
    print("XERV Crayon Benchmark Suite")
    print("=" * 60)
    
    # 1. Setup Vocabulary (Synthetic for demo)
    print("\n[1] Generating Synthetic Vocabulary...")
    vocab_tokens = ["the", "of", "and", "in", "to", "a", "with", "is", " "] + \
                   [f"word{i}" for i in range(50000)]
    vocab = CrayonVocab(vocab_tokens)
    
    print(f"    Vocabulary size: {len(vocab):,} tokens")
    print(f"    C-Extension enabled: {vocab._c_ext_available}")
    
    # 2. Setup Dummy Corpora
    os.makedirs("temp_bench_data", exist_ok=True)
    corpus_path = "temp_bench_data/synthetic.txt"
    with open(corpus_path, "w", encoding="utf-8") as f:
        # 10MB of text
        f.write((" ".join(vocab_tokens[:100]) + " ") * 20000)
        
    corpora = {"synthetic_10mb": corpus_path}
    
    # 3. Run Benchmarks
    print("\n[2] Running Corpus Benchmarks...")
    bench = CrayonBenchmark(vocab, corpora)
    results = bench.run_benchmarks(iterations=5)
    
    # 4. Report
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(json.dumps(results, indent=2))
    
    # 5. C vs Python comparison
    print("\n[3] Running C Extension vs Python Comparison...")
    comparison_text = " ".join(vocab_tokens[:100]) * 1000
    comparison = bench.run_c_vs_python_comparison(comparison_text, iterations=10)
    
    print("\nC Extension vs Python Fallback:")
    print(json.dumps(comparison, indent=2))
    
    if 'c_extension' in comparison and 'python_fallback' in comparison:
        speedup = comparison['python_fallback']['mean_time'] / comparison['c_extension']['mean_time']
        print(f"\n>>> C Extension Speedup: {speedup:.2f}x")
    
    # Cleanup
    os.remove(corpus_path)
    os.rmdir("temp_bench_data")
    
    print("\n[Done] Benchmark complete.")

if __name__ == "__main__":
    main()