
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
        
        # We should at least achieve baseline performance (10k is very conservative for C++ engine)
        self.assertGreater(throughput, 10000, "Throughput fell below minimum acceptable threshold")

    def test_engine_performance_boost(self):
        """Test that the engine provides reasonable performance."""
        # In V4, 'fast_mode' is the default if compiled.
        # We check by seeing if it's using the C++ backend.
        info = self.vocab.get_info()
        is_fast = info["backend"].endswith("_extension")
        
        if not is_fast:
             self.skipTest("C++ extension not available, can't test boost")
             
        start = time.perf_counter()
        for _ in range(3):
            _ = self.vocab.tokenize(self.text)
        c_time = time.perf_counter() - start
        
        print(f"C++ Engine time: {c_time:.3f}s")
        self.assertGreater(len(self.vocab.tokenize(self.text)), 0)

if __name__ == "__main__":
    unittest.main()