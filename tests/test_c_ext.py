import unittest
import sys
from crayon.core.vocabulary import CrayonVocab

# Check availability
try:
    from crayon.c_ext import _core
    C_EXT_AVAILABLE = True
except ImportError:
    C_EXT_AVAILABLE = False

class TestCExtension(unittest.TestCase):
    
    def setUp(self):
        self.vocab_list = ["apple", "app", "application", "banana", "band", "b"]
        self.vocab = CrayonVocab(self.vocab_list, unk_token="<UNK>")

    @unittest.skipUnless(C_EXT_AVAILABLE, "C extension not compiled")
    def test_c_extension_loaded(self):
        """Verify C module is importable and exposes correct functions."""
        self.assertTrue(hasattr(_core, "crayon_tokenize_fast"))
        self.assertTrue(hasattr(_core, "build_trie"))

    @unittest.skipUnless(C_EXT_AVAILABLE, "C extension not compiled")
    def test_c_trie_built(self):
        """Verify C-Trie is automatically built during vocabulary initialization."""
        self.assertTrue(self.vocab._c_ext_available)
        self.assertIsNotNone(self.vocab._c_trie)

    @unittest.skipUnless(C_EXT_AVAILABLE, "C extension not compiled")
    def test_longest_match_correctness(self):
        """Compare C-ext output against Python reference implementation."""
        text = "appleband"
        
        # Get tokens (uses C extension if available)
        tokens = self.vocab.tokenize(text)
        
        self.assertEqual(len(tokens), 2)
        # "apple" (id 0) + "band" (id 4)
        expected_ids = [self.vocab.token_to_id["apple"], self.vocab.token_to_id["band"]]
        self.assertEqual(tokens, expected_ids)

    @unittest.skipUnless(C_EXT_AVAILABLE, "C extension not compiled")
    def test_segfault_resilience(self):
        """Verify robust handling of empty strings and edge cases."""
        # Empty string
        self.assertEqual(self.vocab.tokenize(""), [])
        # Unicode string (should fall back to UNK for unknown chars)
        result = self.vocab.tokenize("café")
        self.assertTrue(len(result) > 0)

    @unittest.skipUnless(C_EXT_AVAILABLE, "C extension not compiled")
    def test_c_ext_vs_python_fallback(self):
        """Ensure C extension and Python fallback produce identical results."""
        text = "applicationbanana"
        
        # Force Python path by temporarily disabling C extension
        original_c_trie = self.vocab._c_trie
        original_available = self.vocab._c_ext_available
        
        self.vocab._c_ext_available = False
        self.vocab._c_trie = None
        python_result = self.vocab.tokenize(text)
        
        # Restore C extension
        self.vocab._c_ext_available = original_available
        self.vocab._c_trie = original_c_trie
        c_result = self.vocab.tokenize(text)
        
        # Results should be identical
        self.assertEqual(python_result, c_result)