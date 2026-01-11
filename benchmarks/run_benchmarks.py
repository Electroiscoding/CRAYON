import os
import sys
import json

# Ensure src is in path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))

from crayon.core.vocabulary import CrayonVocab
from benchmarks.micro_bench import CrayonBenchmark

def main():
    print("Initializing Crayon Benchmark Suite...")
    
    # 1. Setup Vocabulary (Synthetic for demo)
    print("Generating Synthetic Vocabulary...")
    vocab_tokens = ["the", "of", "and", "in", "to", "a", "with", "is"] + \
                   [f"word{i}" for i in range(50000)]
    vocab = CrayonVocab(vocab_tokens)
    
    # 2. Setup Dummy Corpora
    os.makedirs("temp_bench_data", exist_ok=True)
    corpus_path = "temp_bench_data/synthetic.txt"
    with open(corpus_path, "w", encoding="utf-8") as f:
        # 10MB of text
        f.write((" ".join(vocab_tokens[:100]) + " ") * 20000)
        
    corpora = {"synthetic_10mb": corpus_path}
    
    # 3. Run
    bench = CrayonBenchmark(vocab, corpora)
    results = bench.run_benchmarks(iterations=5)
    
    # 4. Report
    print("\n" + "="*40)
    print("BENCHMARK RESULTS")
    print("="*40)
    print(json.dumps(results, indent=2))
    
    # Cleanup
    os.remove(corpus_path)
    os.rmdir("temp_bench_data")

if __name__ == "__main__":
    main()