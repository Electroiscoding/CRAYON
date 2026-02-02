
import unittest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from crayon.core.vocabulary import CrayonVocab
from crayon.core.primitives import TokenMetadata

class TestCoreTokenization(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.tokens = ["un", "fortunate", "ly", "unfortunate", "man"]
        # Building the vocab will assign IDs: 
        # un:0, fortunate:1, ly:2, unfortunate:3, man:4
        cls.vocab = CrayonVocab(cls.tokens)

    def test_longest_match_priority(self):
        """
        Verify that the tokenizer strictly prefers the longest match.
        'unfortunately' -> 'unfortunate' + 'ly' (if 'unfortunately' not in vocab)
        """
        text = "unfortunately"
        ids = self.vocab.tokenize(text)
        
        # 'unfortunate' (3) and 'ly' (2)
        # resolved_tokens = [self.vocab.id_to_token[i] for i in ids]
        self.assertEqual(ids, [3, 2])
        
        # Verify decoding
        decoded = self.vocab.decode(ids)
        self.assertEqual(decoded, "unfortunately")

    def test_unknown_token_fallback(self):
        """Verify fallback for unknown bytes/tokens."""
        text = "x"  # 'x' is unknown
        ids = self.vocab.tokenize(text)
        
        # Engine hardcoded fallback is ID 1
        self.assertIn(1, ids)

    def test_metadata_memory_layout(self):
        """Verify primitives use slots."""
        meta = TokenMetadata(token_id=1, frequency=100, average_length=5.5)
        # Frozen dataclasses raise FrozenInstanceError (Python 3.10+) or TypeError
        with self.assertRaises((AttributeError, TypeError)):
            meta.new_attr = 1  # Should fail due to __slots__ and frozen=True

    def test_vocabulary_contains(self):
        """Test vocabulary membership checks."""
        self.assertIn("unfortunate", self.vocab)
        self.assertNotIn("nonexistent", self.vocab)

    def test_vocabulary_size(self):
        """Test vocabulary size."""
        self.assertEqual(len(self.vocab), 5)

if __name__ == "__main__":
    unittest.main()