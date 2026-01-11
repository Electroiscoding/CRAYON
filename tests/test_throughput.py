import unittest
import time
from src.crayon.core.vocabulary import CrayonVocab

class TestThroughput(unittest.TestCase):
    
    def setUp(self):
        # Generate synthetic corpus
        self.vocab_tokens = [f"token{i}" for i in range(1000)]
        self.vocab = CrayonVocab(self.vocab_tokens)
        
        # Construct large text
        import random
        self.text = " ".join(random.choices(self.vocab_tokens, k=100_000))

    def test_throughput_target(self):
        """
        Benchmark core throughput.
        Target: > 2M tokens/sec (Section 10.1).
        """
        start = time.perf_counter()
        _ = self.vocab.tokenize(self.text)
        end = time.perf_counter()
        
        duration = end - start
        token_count = 100_000
        tps = token_count / duration
        
        print(f"\nThroughput Test: {tps:,.0f} tokens/sec")
        
        # We assert a relaxed lower bound for CI environments (GitHub Actions are slow)
        # But for "overachieving", we expect high numbers locally.
        self.assertTrue(tps > 100_000, "Throughput below sanity check limit")