import time
import tracemalloc
import statistics
from typing import Dict, List, Any
from crayon.core.vocabulary import CrayonVocab

class CrayonBenchmark:
    """
    Comprehensive micro-benchmark suite for tokenizer performance evaluation.
    
    Measures throughput, latency, and memory usage across different configurations.
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
            text = f.read()  # Load into RAM for micro-bench (throughput focus)
            
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
            peak_mem.append(peak / 1024 / 1024)  # MB
            
        total_tokens = len(tokens)  # from last run
        
        return {
            "throughput_mean": total_tokens / statistics.mean(times),
            "latency_ms_per_mb": (statistics.mean(times) * 1000) / (len(text.encode('utf-8')) / 1e6),
            "memory_peak_mb": statistics.mean(peak_mem),
            "c_ext_enabled": self.tokenizer._c_ext_available
        }

    def run_c_vs_python_comparison(self, text: str, iterations: int = 10) -> Dict:
        """Compare C extension vs Python fallback performance."""
        results = {}
        
        # Test with C extension (if available)
        if self.tokenizer._c_ext_available:
            times = []
            for _ in range(iterations):
                start = time.perf_counter()
                _ = self.tokenizer.tokenize(text)
                times.append(time.perf_counter() - start)
            results['c_extension'] = {
                'mean_time': statistics.mean(times),
                'std_dev': statistics.stdev(times) if len(times) > 1 else 0
            }
        
        # Test with Python fallback
        original_available = self.tokenizer._c_ext_available
        original_trie = self.tokenizer._c_trie
        
        self.tokenizer._c_ext_available = False
        self.tokenizer._c_trie = None
        
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            _ = self.tokenizer.tokenize(text)
            times.append(time.perf_counter() - start)
        results['python_fallback'] = {
            'mean_time': statistics.mean(times),
            'std_dev': statistics.stdev(times) if len(times) > 1 else 0
        }
        
        # Restore C extension
        self.tokenizer._c_ext_available = original_available
        self.tokenizer._c_trie = original_trie
        
        return results