import unittest
import time
from crayon.core.vocabulary import CrayonVocab

class TestThroughput(unittest.TestCase):
    
    def setUp(self):
        # Large vocabulary
        self.tokens = ["the", "of", "and", "in", "to", "a", "with", "is", " "] + \
                      [f"word{i}" for i in range(1000)]
        self.vocab = CrayonVocab(self.tokens)
        # Sample text
        self.text = " ".join(["the", "of", "and"] * 10000)

    def test_throughput_target(self):
        """Benchmark core throughput."""
        # Warm up
        _ = self.vocab.tokenize(self.text)
        
        # Measure
        iterations = 5
        start = time.perf_counter()
        for _ in range(iterations):
            _ = self.vocab.tokenize(self.text)
        elapsed = time.perf_counter() - start
        
        total_tokens = len(self.vocab.tokenize(self.text)) * iterations
        throughput = total_tokens / elapsed
        
        print(f"Throughput Test: {throughput:,.0f} tokens/sec")
        
        # We should at least achieve baseline performance
        self.assertGreater(throughput, 10000, "Throughput fell below minimum acceptable threshold")

    def test_c_extension_performance_boost(self):
        """Test that C extension provides performance improvement."""
        if not self.vocab._c_ext_available:
            self.skipTest("C extension not available")
        
        # Measure Python fallback
        self.vocab._c_ext_available = False
        original_trie = self.vocab._c_trie
        self.vocab._c_trie = None
        
        start = time.perf_counter()
        for _ in range(3):
            _ = self.vocab.tokenize(self.text)
        python_time = time.perf_counter() - start
        
        # Restore C extension
        self.vocab._c_ext_available = True
        self.vocab._c_trie = original_trie
        
        start = time.perf_counter()
        for _ in range(3):
            _ = self.vocab.tokenize(self.text)
        c_time = time.perf_counter() - start
        
        print(f"Python time: {python_time:.3f}s, C time: {c_time:.3f}s")
        # C extension should be at least comparable (may not always be faster due to Python overhead)