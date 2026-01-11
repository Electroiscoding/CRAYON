import time
import tracemalloc
import statistics
from typing import Dict, List, Any
from src.crayon.core.vocabulary import CrayonVocab
# Note: psutil dependency assumed for full implementation, omitted here for stdlib purity if needed
# but specified in paper.

class CrayonBenchmark:
    """
    Comprehensive micro-benchmark suite for tokenizer performance evaluation.
    """
    
    def __init__(self, tokenizer: CrayonVocab, test_corpora: Dict[str, str]):
        self.tokenizer = tokenizer
        self.corpora = test_corpora
        self.results: Dict[str, Any] = {}

    def run_benchmarks(self, iterations: int = 5) -> Dict:
        """Execute full benchmark suite."""
        for name, path in self.corpora.items():
            self.results[name] = self._run_corpus_bench(path, iterations)
        return self.results

    def _run_corpus_bench(self, path: str, iterations: int) -> Dict:
        """Run single corpus benchmark."""
        with open(path, 'r', encoding='utf-8') as f:
            text = f.read() # Load into RAM for micro-bench (throughput focus)
            
        times = []
        peak_mem = []
        
        for _ in range(iterations):
            tracemalloc.start()
            start = time.perf_counter()
            
            tokens = self.tokenizer.tokenize(text)
            
            end = time.perf_counter()
            _, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            
            times.append(end - start)
            peak_mem.append(peak / 1024 / 1024) # MB
            
        total_tokens = len(tokens) # from last run
        
        return {
            "throughput_mean": total_tokens / statistics.mean(times),
            "latency_ms_per_mb": (statistics.mean(times) * 1000) / (len(text.encode('utf-8')) / 1e6),
            "memory_peak_mb": statistics.mean(peak_mem)
        }