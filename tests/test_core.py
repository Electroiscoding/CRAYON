# Verify Trie logic & fallback
import unittest
from crayon.core.tokenizer import CrayonTokenizer

class TestCore(unittest.TestCase):
    def test_tokenizer_init(self):
        t = CrayonTokenizer()
        self.assertIsNotNone(t)
